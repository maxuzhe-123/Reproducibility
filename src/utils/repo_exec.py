# # repo_exec.py
# from __future__ import annotations
#
# import os
# import os, re, shutil, subprocess, tempfile, sys
# # repo_exec.py
# PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))  # go from src/utils back to the project root
#
# from typing import Optional
#
# REQ_BASE = [
#     "pip",
#     "setuptools",
#     "wheel",
#     "pytest",
#     "pytest-asyncio",
#     "pytest-mock",
#     "freezegun",
#     "mock",
# ]
#
# def _venv_python(venv_dir: str) -> str:
#     return os.path.join(venv_dir, "bin", "python")
#
# def _ensure_repo_venv(repo_id: str) -> str:
#     venv_dir = os.path.join(PROJECT_ROOT, ".repo_venvs", repo_id)
#     py = _venv_python(venv_dir)
#     if os.path.exists(py):
#         return venv_dir
#
#     os.makedirs(os.path.dirname(venv_dir), exist_ok=True)
#     subprocess.check_call([sys.executable, "-m", "venv", venv_dir])
#     subprocess.check_call([py, "-m", "pip", "install", "-U", "pip"])
#     subprocess.check_call([py, "-m", "pip", "install", "-U"] + REQ_BASE)
#     return venv_dir
#
# def run_tests_in_dir(workdir: str, test_cmd: str, timeout_sec: int = 600, repo_id: str = "default"):
#     env = os.environ.copy()
#     env["PYTHONNOUSERSITE"] = "1"
#
#     # Key point: make the repo's src-layout / local packages importable by pytest
#     src_path = os.path.join(workdir, "src")
#     env["PYTHONPATH"] = src_path + os.pathsep + workdir + os.pathsep + env.get("PYTHONPATH", "")
#
#     venv_dir = _ensure_repo_venv(repo_id)
#     py = _venv_python(venv_dir)
#
#     # Run with python -m pytest from the repo venv (do not use shell=True)
#     cmd = test_cmd.strip()
#     if cmd.startswith("pytest "):
#         args = cmd.split()[1:]
#     elif cmd == "pytest":
#         args = []
#     else:
#         # If the command does not start with pytest, splitting it shell-style is risky
#         # test_cmd should always be written as "pytest ..."
#         raise ValueError(f"test_cmd must start with pytest, got: {test_cmd}")
#
#     full_cmd = [py, "-m", "pytest"] + args
#
#     p = subprocess.run(
#         full_cmd,
#         cwd=workdir,
#         capture_output=True,
#         text=True,
#         timeout=timeout_sec,
#         env=env,
#     )
#     ok = (p.returncode == 0)
#     out = (p.stdout or "") + "\n" + (p.stderr or "")
#     return ok, out
#
# def eval_one_sample_in_tmp(
#     repo_path: str,
#     relpath: str,
#     entry_point: str,
#     code: str,
#     start_line: int | None,
#     end_line: int | None,
#     test_cmd: str,
#     timeout_sec: int = 600,
# ):
#     tmp_root = tempfile.mkdtemp(prefix="codereval_repo_")
#     repo_tmp = os.path.join(tmp_root, "repo")
#     print("cwd:", os.getcwd())
#     print("repo_path(abs):", os.path.abspath(repo_path), "exists?", os.path.exists(repo_path))
#     # Add this before copytree in eval_one_sample_in_tmp
#     if not os.path.isabs(repo_path):
#         repo_path = os.path.join(PROJECT_ROOT, repo_path)
#
#     print("fixed repo_path:", repo_path, "exists?", os.path.exists(repo_path))
#     try:
#         shutil.copytree(repo_path, repo_tmp)  # do not use dirs_exist_ok
#         ok_apply, apply_err = apply_patch_to_repo(
#             repo_root=repo_tmp,
#             relpath=relpath,
#             entry_point=entry_point,
#             new_func_code=code,
#             start_line=start_line,
#             end_line=end_line,
#         )
#         if not ok_apply:
#             return False, f"apply_patch_failed: {apply_err}"
#         # ok, out = run_tests_in_dir(repo_tmp, test_cmd, timeout_sec=timeout_sec)
#         repo_id = os.path.basename(os.path.normpath(repo_path))  # e.g. neo4j---neo4j-python-driver
#         ok, out = run_tests_in_dir(repo_tmp, test_cmd, timeout_sec=timeout_sec, repo_id=repo_id)
#         return ok, out
#     finally:
#         print("tmp_root:", tmp_root)
#         print("repo_tmp:", repo_tmp)
#         shutil.rmtree(tmp_root, ignore_errors=True)
#
#
# def _resolve_repo_file(repo_root: str, relpath: str) -> str | None:
#     """
#     Resolve the actual file path inside a repo for CoderEval tasks.
#
#     CoderEval's relpath may omit 'src/' while many repos use a src-layout.
#     We try common candidates in order.
#     """
#     candidates = [
#         relpath,
#         os.path.join("src", relpath),
#     ]
#
#     for rp in candidates:
#         p = os.path.join(repo_root, rp)
#         if os.path.exists(p):
#             return p
#     return None
#
# def _line_indent(s: str) -> int:
#     return len(s) - len(s.lstrip(" "))
#
# def _find_def_line_idx(lines: list[str], entry_point: str) -> int | None:
#     pat = re.compile(rf"^\s*def\s+{re.escape(entry_point)}\s*\(")
#     for i, ln in enumerate(lines):
#         if pat.match(ln):
#             return i
#     return None
#
# def _find_block_end_by_indent(lines: list[str], def_idx: int) -> int:
#     """
#     Find end of a Python def block by indentation.
#     Returns end index as Python slice end (exclusive).
#     """
#     def_indent = _line_indent(lines[def_idx])
#     i = def_idx + 1
#     while i < len(lines):
#         ln = lines[i]
#         # keep empty lines and comments inside block scanning
#         if ln.strip() == "" or ln.lstrip().startswith("#"):
#             i += 1
#             continue
#         cur_indent = _line_indent(ln)
#         # If indentation <= def indentation, block ended
#         if cur_indent <= def_indent:
#             break
#         i += 1
#     return i  # exclusive
#
# def apply_patch_to_repo(
#     repo_root: str,
#     relpath: str,
#     entry_point: str,
#     new_func_code: str,
#     start_line: int | None = None,
#     end_line: int | None = None,
# ):
#     """
#     Patch target function safely.
#
#     Strategy:
#     1) Locate file (relpath or src/relpath)
#     2) Locate def entry_point
#     3) Prefer indent-based end boundary (robust across versions)
#     4) If start_line/end_line provided, use them ONLY if they fully cover the real function block.
#        Otherwise fallback to indent-based range to avoid leaving dangling indented lines.
#     """
#     target = _resolve_repo_file(repo_root, relpath)
#     if target is None:
#         return False, f"target file not found (tried relpath and src/relpath): {relpath}"
#
#     with open(target, "r", encoding="utf-8", errors="ignore") as f:
#         lines = f.read().splitlines()
#
#     def_idx = _find_def_line_idx(lines, entry_point)
#     if def_idx is None:
#         return False, f"cannot find def {entry_point} in {relpath}"
#
#     # real end by indentation (exclusive)
#     real_end = _find_block_end_by_indent(lines, def_idx)
#
#     # candidate range from dataset (1-based inclusive)
#     use_dataset_range = False
#     if isinstance(start_line, int) and isinstance(end_line, int) and start_line >= 1 and end_line >= start_line:
#         ds_s = start_line - 1
#         ds_e = end_line  # exclusive
#         # Only accept dataset range if it covers the whole real block
#         # (i.e., ds_s <= def_idx and ds_e >= real_end)
#         if ds_s <= def_idx and ds_e >= real_end:
#             use_dataset_range = True
#
#     if use_dataset_range:
#         s = start_line - 1
#         e = end_line
#     else:
#         s = def_idx
#         e = real_end
#
#     out_lines = lines[:s] + new_func_code.splitlines() + lines[e:]
#     with open(target, "w", encoding="utf-8") as f:
#         f.write("\n".join(out_lines) + "\n")
#
#     return True, ""
#
