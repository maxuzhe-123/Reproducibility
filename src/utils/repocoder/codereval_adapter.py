from __future__ import annotations

import ast
import json
import os
import re
import tempfile
import textwrap
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from src.datasets.codereval import build_validate_file_map
from src.utils.codereval_exec import run_codereval_script
from src.utils.parse import extract_code


def _clean_code(code: str) -> str:
    if not code:
        return ""
    lines = [line for line in code.splitlines() if not line.strip().startswith("#")]
    code = "\n".join(lines)
    code = re.sub(r"'''[\s\S]*?'''", "", code)
    code = re.sub(r'"""[\s\S]*?"""', "", code)
    return code


def _strip_leading_docstring(code: str) -> str:
    stripped = code.lstrip()
    for quote in ('"""', "'''"):
        if stripped.startswith(quote):
            close = stripped.find(quote, len(quote))
            if close != -1:
                return stripped[close + len(quote):].lstrip("\n")
    return code


@contextmanager
def _temporary_cwd(path: Path):
    old = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(old)


class CoderEvalRepoCoderAdapter:
    """
    Adapter between the current CoderEval runner style and the original
    RepoCoder RG1 pipeline in src/utils/repocoder.
    """

    def __init__(self, cfg: Dict[str, Any]):
        self.cfg = cfg
        self.project_root = Path(cfg.get("project_root") or Path.cwd()).resolve()

        ce_cfg = cfg.get("codereval", {}) or {}
        rag_cfg = cfg.get("rag", {}) or {}

        self.repos_root = Path(ce_cfg.get("repos_root", "repos")).resolve()
        self.index_path = Path(
            ce_cfg.get("index_path", "src/datasets/CoderEval/CoderEval4Python.json")
        ).resolve()
        self.dataset_path = Path(
            ce_cfg.get("dataset_path", "src/datasets/CoderEval-ALL.jsonl")
        ).resolve()

        prompt_cfg = cfg.get("prompt", {}) or {}
        self.rg1_prompt_path = Path(
            prompt_cfg.get("rag_prompt_path", "prompts/rg-one-gram-ws-20-ss-2.jsonl")
        ).resolve()

        self.window_sizes = list(rag_cfg.get("window_sizes", [20]))
        self.slice_sizes = list(rag_cfg.get("slice_sizes", [2]))
        self.regenerate_prompts = bool(rag_cfg.get("regenerate_prompts", False))

        self.id2rag_prompt: Dict[str, str] = {}
        self.id2validate_file: Dict[str, str] = {}
        self.id2project: Dict[str, str] = {}

    def load_tasks(self, num_tasks: Optional[int] = None) -> List[Dict[str, Any]]:
        if not self.dataset_path.exists():
            raise FileNotFoundError(f"CoderEval dataset not found: {self.dataset_path}")

        tasks: List[Dict[str, Any]] = []
        with self.dataset_path.open("r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    tasks.append(json.loads(line))
        return tasks[: int(num_tasks)] if num_tasks is not None else tasks

    def build_validate_file_map(self) -> None:
        self.id2validate_file = build_validate_file_map(
            str(self.index_path),
            str(self.repos_root),
        )
        # _id -> project: standalone tasks use this to locate the local repository so that `import <repo package>` resolves to the
        # package of the same name inside the repository (consistent with GroundTruth).
        try:
            records = json.loads(
                self.index_path.read_text(encoding="utf-8")
            ).get("RECORDS", [])
            self.id2project = {str(r["_id"]): r.get("project", "") for r in records}
        except Exception:
            self.id2project = {}

    def prepare_rg1_prompts(self) -> None:
        repos = [p.name for p in self.repos_root.glob("*---*") if p.is_dir()]
        if not repos:
            raise FileNotFoundError(f"No repos found under: {self.repos_root}")

        if self.regenerate_prompts or not self.rg1_prompt_path.exists():
            try:
                from src.utils.repocoder.repo_utils import CONSTANTS, FilePathBuilder
                from src.utils.repocoder.run_pipeline import run_RG1_and_oracle_method
            except ModuleNotFoundError as exc:
                raise ModuleNotFoundError(
                    "RepoCoder prompt generation requires the original repocoder "
                    "dependencies. Install the missing dependency or provide an "
                    f"existing RG1 prompt file at {self.rg1_prompt_path}."
                ) from exc

            self.rg1_prompt_path.parent.mkdir(parents=True, exist_ok=True)
            with _temporary_cwd(self.project_root):
                FilePathBuilder.repo_base_dir = str(self.repos_root.relative_to(self.project_root))
                FilePathBuilder.codereval_benchmark = str(self.dataset_path.relative_to(self.project_root))
                run_RG1_and_oracle_method(
                    CONSTANTS.codereval_benchmark,
                    repos,
                    self.window_sizes,
                    self.slice_sizes,
                )

        self.id2rag_prompt = self._load_rg1_prompts(self.rg1_prompt_path)

    def _load_rg1_prompts(self, path: Path) -> Dict[str, str]:
        if not path.exists():
            raise FileNotFoundError(f"RG1 prompt file not found: {path}")

        id2prompt: Dict[str, str] = {}
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                item = json.loads(line)
                task_id = item["metadata"]["task_id"]
                id2prompt[task_id.split("/")[-1]] = item["prompt"]
        return id2prompt

    def get_effective_prompt(self, task: Dict[str, Any], rag_mode: str) -> str:
        if rag_mode == "rg1":
            task_id = task["metadata"]["task_id"]
            return self.id2rag_prompt.get(task_id.split("/")[-1], task.get("prompt", ""))
        return task.get("prompt", "")

    def extract_and_fix_code(self, raw: str, task: Dict[str, Any]) -> str:
        code = extract_code(raw)
        md = task.get("metadata", {})
        function_name = md.get("function_name") or md.get("name", "")
        signature = md.get("signature", "")

        if signature:
            canonical_sig = signature if signature.startswith("def ") else f"def {signature}"
            if not canonical_sig.rstrip().endswith(":"):
                canonical_sig = canonical_sig.rstrip() + ":"
        elif function_name:
            canonical_sig = f"def {function_name}():"
        else:
            canonical_sig = None

        if not code:
            return f"{canonical_sig}\n    pass" if canonical_sig else ""

        if function_name and re.match(r"^\s*def\s+", code):
            match = re.match(r"^\s*def\s+(\w+)", code)
            if match and match.group(1) != function_name:
                real = re.search(
                    rf"^def\s+{re.escape(function_name)}\s*\(",
                    raw,
                    re.MULTILINE,
                )
                if real:
                    tail = raw[real.start():]
                    for stop in ("```", "Stop.", "[DONE]", "[END]"):
                        if stop in tail:
                            tail = tail[:tail.index(stop)]
                    code = textwrap.dedent(tail).strip()
                else:
                    code = ""

        if not code or not re.match(r"^\s*def\s+", code):
            body = _strip_leading_docstring(code) if code else ""
            found = None
            for j, line in enumerate(body.splitlines()):
                if re.match(r"^\s*def\s+", line):
                    found = j
                    break
            if found is not None:
                code = "\n".join(body.splitlines()[found:])
            elif canonical_sig:
                indented = "\n".join(("    " + ln) if ln.strip() else "" for ln in body.splitlines())
                code = canonical_sig + "\n" + (indented if indented.strip() else "    pass")
            else:
                return code

        if canonical_sig:
            lines = code.splitlines()
            for j, line in enumerate(lines):
                if re.match(r"^\s*def\s+", line):
                    lines[j] = canonical_sig
                    break
            code = "\n".join(lines)

        lines = code.splitlines()
        seen_target = False
        cut_at = None
        for j, line in enumerate(lines):
            if re.match(r"^def\s+", line):
                if not seen_target:
                    seen_target = True
                else:
                    cut_at = j
                    break
        if cut_at is not None:
            lines = lines[:cut_at]

        lines = [line for line in lines if not re.match(r"^(return|yield|raise)\b", line)]
        return "\n".join(lines).rstrip()

    def _infer_repo_dir_from_validate_path(self, validate_path: str) -> Optional[Path]:
        vp = Path(validate_path).resolve()
        try:
            rel = vp.relative_to(self.repos_root)
        except ValueError:
            return None
        return (self.repos_root / rel.parts[0]).resolve() if rel.parts else None

    def eval_one(self, task: Dict[str, Any], generated_code: str, timeout_sec: int = 30) -> Tuple[bool, str]:
        task_id = task["metadata"]["task_id"]
        sample_id = str(task.get("_id") or task_id.split("/")[-1])
        validate_path = self.id2validate_file.get(sample_id)
        if not validate_path:
            return False, f"missing_validate_file_for_id={sample_id}"

        validate_path = str(Path(validate_path).resolve())
        if not os.path.exists(validate_path):
            return False, f"validate_file_not_found: {validate_path}"

        file_content = Path(validate_path).read_text(encoding="utf-8", errors="ignore")
        file_lines = file_content.split("\n")
        try:
            ast_file = ast.parse(file_content)
        except Exception as exc:
            return False, f"ast_parse_failed: {exc!r}"

        target_name = (
            task["metadata"].get("function_name")
            or task["metadata"].get("name")
            or task["metadata"].get("signature", "")
        )
        if "(" in str(target_name):
            target_name = str(target_name).split("(", 1)[0].strip()

        new_data = ""
        start_indent = 0
        for node in ast.walk(ast_file):
            if not isinstance(node, ast.FunctionDef):
                continue
            if node.name != target_name and node.name != f"_{target_name}":
                continue
            end_line = getattr(node, "end_lineno", None)
            if end_line is None:
                return False, "missing_end_lineno_in_ast"
            start_line = node.lineno
            indent_line = file_lines[start_line - 1]
            start_indent = len(indent_line) - len(indent_line.lstrip(" "))
            head = "\n".join(file_lines[:start_line - 1])
            tail = "\n".join(file_lines[end_line:])
            new_data = head + "\n<insert generated code here>\n" + tail + "\n"
            break

        if not new_data:
            return False, f"function_not_found_in_validate_file: {target_name}"

        code = _clean_code(generated_code).rstrip() + "\n"
        code_lines = code.split("\n")
        code_indent = len(code_lines[0]) - len(code_lines[0].lstrip(" ")) if code_lines and code_lines[0].strip() else 0
        if start_indent > code_indent:
            pad = " " * (start_indent - code_indent)
            code = "\n".join((pad + ln) if ln.strip() else ln for ln in code_lines)

        out_data = new_data.replace("<insert generated code here>", code)
        validate_dir = Path(validate_path).parent
        temp_dir = Path(tempfile.mkdtemp(prefix="codereval_candidate_"))
        out_py = temp_dir / Path(validate_path).name
        out_py.write_text(out_data, encoding="utf-8")

        repo_dir = self._infer_repo_dir_from_validate_path(validate_path)
        if repo_dir is not None and repo_dir.name == "standalone":
            # Prefer the dataset's project field to locate the real repository (the task_id prefix is "CoderEval", not the repository name).
            project = self.id2project.get(sample_id, "")
            repo_name = project.replace("/", "---") if project else task_id.split("/", 1)[0]
            candidate_repo_dir = self.repos_root / repo_name
            if candidate_repo_dir.is_dir():
                repo_dir = candidate_repo_dir.resolve()
        if repo_dir is None:
            repo_dir = validate_dir
        ok, err = run_codereval_script(
            script_path=str(out_py),
            repos_root=str(self.repos_root),
            repo_root=str(repo_dir),
            project_root=str(self.project_root),
            timeout_sec=int(timeout_sec),
            logical_script_path=validate_path,
        )
        try:
            out_py.unlink()
            temp_dir.rmdir()
        except OSError:
            pass
        return ok, err
