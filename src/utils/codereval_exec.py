from __future__ import annotations

import hashlib
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Optional, Tuple


TRAVIS_ROOT = "/home/travis/builds/repos"


def _split_header_and_body(source: str) -> tuple[str, str]:
    lines = source.splitlines(True)
    header = []
    i = 0

    if i < len(lines) and lines[i].startswith("#!"):
        header.append(lines[i])
        i += 1

    if i < len(lines) and "coding" in lines[i]:
        header.append(lines[i])
        i += 1

    while i < len(lines) and (
        lines[i].strip() == "" or lines[i].lstrip().startswith("#")
    ):
        header.append(lines[i])
        i += 1

    if i < len(lines) and lines[i].lstrip().startswith(('"""', "'''")):
        quote = '"""' if lines[i].lstrip().startswith('"""') else "'''"
        first = lines[i]
        header.append(first)
        i += 1
        if first.count(quote) < 2:
            while i < len(lines) and quote not in lines[i]:
                header.append(lines[i])
                i += 1
            if i < len(lines):
                header.append(lines[i])
                i += 1

    while i < len(lines) and (
        lines[i].strip() == "" or lines[i].lstrip().startswith("#")
    ):
        header.append(lines[i])
        i += 1

    while i < len(lines) and lines[i].lstrip().startswith(
        "from __future__ import"
    ):
        header.append(lines[i])
        i += 1

    return "".join(header), "".join(lines[i:])


def prepare_codereval_script(
    script_path: str,
    repos_root: str,
    repo_root: str,
    project_root: str,
    logical_script_path: Optional[str] = None,
) -> str:
    """Patch a CoderEval validation script using the GroundTruth environment."""
    source_path = Path(script_path).resolve()
    logical_path = Path(logical_script_path or script_path).resolve()
    source = source_path.read_text(encoding="utf-8", errors="replace")
    patched_source = source.replace(TRAVIS_ROOT, str(Path(repos_root).resolve()))

    digest = hashlib.md5(
        f"{logical_path}:{os.getpid()}:{os.urandom(8).hex()}".encode()
    ).hexdigest()[:12]
    temp_dir = (
        Path(tempfile.gettempdir())
        / f"codereval_exec_{digest}"
        / logical_path.parent.name
    )
    temp_dir.mkdir(parents=True, exist_ok=True)
    output_path = temp_dir / logical_path.name

    repo_root_path = Path(repo_root).resolve()
    project_root_path = Path(project_root).resolve()
    bootstrap = (
        "import os, sys\n"
        f'REPO_ROOT = r"{repo_root_path}"\n'
        f'SCRIPT_PATH = r"{logical_path}"\n'
        f'PROJECT_ROOT = r"{project_root_path}"\n'
        "__file__ = SCRIPT_PATH\n"
        "sys.argv[0] = SCRIPT_PATH\n"
        "SCRIPT_DIR = os.path.dirname(SCRIPT_PATH)\n"
        "sys.path.insert(0, SCRIPT_DIR)\n"
        "sys.path.insert(0, REPO_ROOT)\n"
        "sys.path.insert(0, os.path.join(REPO_ROOT, 'src'))\n"
        "krake_root = os.path.join(REPO_ROOT, 'krake')\n"
        "if os.path.isdir(os.path.join(krake_root, 'krake')):\n"
        "    sys.path.insert(0, krake_root)\n"
        "dateutil_dir = os.path.join(REPO_ROOT, 'dateutil')\n"
        "if os.path.isdir(dateutil_dir):\n"
        "    sys.path.insert(0, dateutil_dir)\n"
        "os.chdir(REPO_ROOT)\n"
        "try:\n"
        "    _src = open(SCRIPT_PATH, 'r', encoding='utf-8', "
        "errors='replace').read()\n"
        "    if 'repos/' in _src or 'repos\\\\' in _src:\n"
        "        if os.path.isdir(os.path.join(PROJECT_ROOT, 'repos')):\n"
        "            os.chdir(PROJECT_ROOT)\n"
        "except Exception:\n"
        "    pass\n"
    )

    header, body = _split_header_and_body(patched_source)
    output = header
    if output and not output.endswith("\n"):
        output += "\n"
    output += "\n" + bootstrap + "\n" + body
    output_path.write_text(output, encoding="utf-8")
    return str(output_path)


def build_codereval_env(repo_root: str) -> dict:
    """Build the same environment variables used by GroundTruth.py."""
    env = dict(os.environ)
    repo = Path(repo_root).resolve()
    additions = [str(repo), str(repo / "src")]
    existing = [p for p in env.get("PYTHONPATH", "").split(os.pathsep) if p]
    env["PYTHONPATH"] = os.pathsep.join(
        additions + [p for p in existing if p not in additions]
    )
    return env


def run_codereval_script(
    script_path: str,
    repos_root: str,
    repo_root: str,
    project_root: str,
    timeout_sec: int = 30,
    python_exe: Optional[str] = None,
    logical_script_path: Optional[str] = None,
) -> Tuple[bool, str]:
    """Execute a validation script under the GroundTruth execution contract."""
    prepared_path = prepare_codereval_script(
        script_path=script_path,
        repos_root=repos_root,
        repo_root=repo_root,
        project_root=project_root,
        logical_script_path=logical_script_path,
    )
    # Interpreter resolution: explicit python_exe > environment variable CODEREVAL_PYTHON > current sys.executable.
    # When GT / main.py / pipeline_v2 share the same executor, CODEREVAL_PYTHON can pin the interpreter
    # to the codereval_eval environment, avoiding dependency inconsistencies caused by "which env it was launched from".
    interpreter = python_exe or os.environ.get("CODEREVAL_PYTHON") or sys.executable
    env = build_codereval_env(repo_root)
    process = None
    try:
        process = subprocess.Popen(
            [interpreter, prepared_path],
            cwd=str(Path(repo_root).resolve()),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        output, _ = process.communicate(timeout=timeout_sec)
        return (
            process.returncode == 0,
            "" if process.returncode == 0 else (output or "").strip(),
        )
    except subprocess.TimeoutExpired:
        if process is not None:
            process.kill()
            process.communicate()
        return False, f"timeout>{timeout_sec}s"
    except Exception as exc:
        return False, f"exception: {exc!r}"
    finally:
        prepared = Path(prepared_path)
        try:
            prepared.unlink()
            prepared.parent.rmdir()
            prepared.parent.parent.rmdir()
        except OSError:
            pass
