"""
src/utils/PythonExec.py
------------------------
CoderEval evaluation tool.

Two usages:

1) Command line (compatible with the original CoderEval toolchain):
   python src/utils/PythonExec.py <preds.jsonl> <n> \\
       [--repos-root <path>] [--index <path>] [--timeout <sec>] [--output <out.jsonl>]

2) Module call (used by run_codereval_simple):
   from src.utils.PythonExec import run_pythonexec
   stats = run_pythonexec(
       preds_jsonl  = "runs/.../preds.jsonl",
       index_json   = "src/datasets/CoderEval/CoderEval4Python.json",
       repos_root   = "repos",
       n            = 10,
       output_jsonl = "runs/.../pythonexec_results.jsonl",
       timeout      = 30,
   )

preds.jsonl format (one JSON object per line):
  {"_id": "<task_id>", "generate_results": ["def foo(): ...", ...], ...}
  - "generate_results" is a list; each element is a complete function definition string (model output)
  - other fields (task_id, level, preds, ...) are ignored

Returns:
  {
    "total":       <int>,   # number of processed tasks
    "pass_k":      [p@1, p@2, ..., p@n],   # pass@k (k=1..n), normalized to [0,1]
    "by_level":    {"self": [p@1,...], ...}
  }
"""

import sys
import os
import json
import ast
import argparse
import numpy as np
import multiprocessing
import tempfile
from pathlib import Path
from typing import Dict, List, Tuple, Optional

try:
    from src.utils.codereval_exec import run_codereval_script
except ModuleNotFoundError:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from src.utils.codereval_exec import run_codereval_script


# ============================================================
# The Process class of the original CoderEval (unchanged, kept for backward compatibility)
# ============================================================
class Process(multiprocessing.Process):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._pconn, self._cconn = multiprocessing.Pipe()
        self._exception = None

    def run(self):
        try:
            multiprocessing.Process.run(self)
            self._cconn.send(None)
        except Exception as exception:
            self._cconn.send(exception)

    def join(self, timeout):
        super().join(timeout)
        if self.is_alive():
            self.terminate()
        super().join()

    @property
    def exception(self):
        if self._pconn.poll():
            self._exception = self._pconn.recv()
        return self._exception


# ============================================================
# Internal utility functions
# ============================================================

# A few special path mappings (relative to repos_root)
_STD_NONESTD_REL: Dict[str, str] = {
    "standalone/neo4j-_meta-deprecated.py":
        "neo4j---neo4j-python-driver/src/neo4j/_meta_deprecated_passk_validte.py",
    "standalone/neo4j-work-query-unit_of_work.py":
        "neo4j---neo4j-python-driver/src/neo4j/_work/query_unit_of_work_passk_validte.py",
    "standalone/krake-krake-controller-kubernetes-hooks-on.py":
        "rak-n-rok---Krake/krake/krake/controller/kubernetes/hooks_on_passk_validte.py",
}


def _build_id2file(repos_root: str, collection: Dict[str, dict]) -> Dict[str, str]:
    """
    Build the mapping _id -> absolute path of _passk_validte.py.
    repos_root must not end with /.
    """
    project_path = repos_root.rstrip("/") + "/"
    nonestd = {
        os.path.normpath(project_path + k): os.path.normpath(project_path + v)
        for k, v in _STD_NONESTD_REL.items()
    }

    id2file: Dict[str, str] = {}
    for key in collection:
        doc = collection[key]
        name = doc.get("name", "")
        file_path = doc.get("file_path", "")
        _id = str(doc["_id"])

        # standalone form
        standalone = os.path.normpath(
            project_path + "standalone/"
            + file_path.replace(".py", "").replace("/", "-")
            + "-" + name + ".py"
        )
        if standalone in nonestd:
            resolved = nonestd[standalone]
            if os.path.exists(resolved):
                id2file[_id] = resolved
            continue
        if os.path.exists(standalone):
            id2file[_id] = standalone
            continue

        # in-repo form
        project = doc.get("project", "")
        repo_dir = project.replace("/", "---")
        if project == "neo4j/neo4j-python-driver":
            candidate = os.path.normpath(
                os.path.join(project_path, repo_dir, "src", file_path)
            ).replace(".py", "_" + name + "_passk_validte.py")
        else:
            candidate = os.path.normpath(
                os.path.join(project_path, repo_dir, file_path)
            ).replace(".py", "_" + name + "_passk_validte.py")

        if candidate in nonestd:
            candidate = nonestd[candidate]
        if os.path.exists(candidate):
            id2file[_id] = candidate

    return id2file


def _eval_code_list(
    code_list: List[str],
    validate_path: str,
    target_name: str,
    repos_root: str,
    repo_root: str,
    project_root: str,
    timeout: int = 30,
) -> Tuple[int, List[dict]]:
    """
    Insert each function definition in code_list into _passk_validte.py in turn,
    run it, and count the number of passes.

    Returns (n_pass, per_code_results)
    per_code_results: [{"generate_code": ..., "is_pass": ..., "return_code": ...}, ...]
    """
    file_content = Path(validate_path).read_text(encoding="utf-8", errors="ignore")
    file_lines = file_content.split("\n")

    # Locate the target function via AST
    try:
        ast_tree = ast.parse(file_content)
    except SyntaxError:
        return 0, []

    start_indent = 0
    new_data = ""
    for node in ast.walk(ast_tree):
        if not isinstance(node, ast.FunctionDef):
            continue
        fn = node.name
        if fn != target_name and fn != ("_" + target_name):
            continue
        start_line = node.lineno
        end_line = getattr(node, "end_lineno", None)
        if end_line is None:
            break
        indent_s = file_lines[start_line - 1]
        start_indent = len(indent_s) - len(indent_s.lstrip(" "))

        head = "\n".join(file_lines[:start_line - 1])
        tail = "\n".join(file_lines[end_line:])
        new_data = head + "\n<insert generated code here>\n" + tail + "\n"
        break

    if not new_data:
        return 0, []

    results: List[dict] = []
    n_pass = 0
    for code_num, code in enumerate(code_list):
        # Indentation alignment (same logic as the original PythonExec.py)
        code_lines = code.split("\n")
        first_nonempty = next((ln for ln in code_lines if ln.strip()), "")
        code_indent = len(first_nonempty) - len(first_nonempty.lstrip(" "))

        if start_indent > code_indent:
            pad = " " * (start_indent - code_indent)
            new_code = "\n".join(pad + ln if ln.strip() else ln for ln in code_lines)
        else:
            new_code = code

        out_data = new_data.replace("<insert generated code here>", new_code)

        temp_dir = Path(tempfile.mkdtemp(prefix="codereval_pythonexec_"))
        tmp_path = temp_dir / Path(validate_path).name
        try:
            tmp_path.write_text(out_data, encoding="utf-8")
            passed, error = run_codereval_script(
                script_path=str(tmp_path),
                repos_root=repos_root,
                repo_root=repo_root,
                project_root=project_root,
                timeout_sec=timeout,
                logical_script_path=validate_path,
            )
        except Exception:
            passed = False
            error = "evaluator_exception"
        finally:
            try:
                tmp_path.unlink()
                temp_dir.rmdir()
            except OSError:
                pass

        if passed:
            n_pass += 1
        results.append({
            "generate_code": code,
            "is_pass": passed,
            "error": "" if passed else error,
        })

    return n_pass, results


# ============================================================
# Main module interface: batch evaluation
# ============================================================

def run_pythonexec(
    preds_jsonl: str,
    index_json: str,
    repos_root: str,
    n: int,
    output_jsonl: Optional[str] = None,
    timeout: int = 30,
) -> dict:
    """
    Read preds_jsonl (each line contains _id and generate_results),
    run the _passk_validte.py evaluation for each generation of each task,
    and return aggregated pass@k statistics.

    Parameters
    ----------
    preds_jsonl  : path to the prediction JSONL file (each line contains at least _id and generate_results)
    index_json   : path to CoderEval4Python.json (used to look up target function names and test stub files)
    repos_root   : root directory of repos (containing each repository and the standalone/ folder)
    n            : number of samples generated per task (denominator of pass@k)
    output_jsonl : output path for evaluation results; None means no file is written
    timeout      : timeout per execution (seconds)

    Returns
    -------
    {
      "total":    <int>,        # number of valid evaluated tasks
      "pass_k":   [p@1, ...],   # pass@1..n, range [0,1]
      "by_level": {"self": [p@1,...], ...}   # stratified by difficulty
    }
    """
    # 1) Load CoderEval4Python.json
    with open(index_json, "r", encoding="utf-8") as f:
        data = json.load(f)
    samples = data.get("RECORDS", [])
    collection: Dict[str, dict] = {str(s["_id"]): s for s in samples}

    # 2) Build the _id -> validate file mapping
    id2file = _build_id2file(repos_root, collection)

    # 3) Read the prediction file
    # Two formats are supported:
    #   - standard JSONL (one JSON object per line, append_jsonl pretty=False)
    #   - multi-line indented JSON (append_jsonl pretty=True, each record spans multiple lines, records separated by blank lines)
    preds_lines: List[dict] = []
    decoder = json.JSONDecoder()
    with open(preds_jsonl, "r", encoding="utf-8") as f:
        content = f.read()
    pos = 0
    while pos < len(content):
        # Skip whitespace characters (newlines, spaces, etc.)
        while pos < len(content) and content[pos] in " \t\n\r":
            pos += 1
        if pos >= len(content):
            break
        try:
            obj, end = decoder.raw_decode(content, pos)
            pos = end
        except json.JSONDecodeError:
            # Skip to the next line and try again
            nl = content.find("\n", pos)
            pos = nl + 1 if nl != -1 else len(content)
            continue
        # Skip records whose generation failed
        if obj.get("skipped_by_gen_error"):
            continue
        # Must have _id and generate_results
        _id = str(obj.get("_id", ""))
        gen_results = obj.get("generate_results")
        if not _id or not gen_results:
            continue
        preds_lines.append(obj)

    # 4) Initialize counters
    count_tot = len(preds_lines)
    tot_k = [0.0] * n
    dict_level_tot: Dict[str, int] = {}
    record_out: Dict[str, List[float]] = {}

    output_rows: List[dict] = []

    # 5) Evaluate each task
    for obj in preds_lines:
        _id = str(obj["_id"])
        code_list = obj["generate_results"][:n]   # Take at most n
        content_doc = collection.get(_id)
        if content_doc is None:
            continue

        validate_path = id2file.get(_id)
        if not validate_path or not os.path.exists(validate_path):
            continue

        target_name = content_doc.get("name", "")
        level = content_doc.get("level", "UNKNOWN")

        # Count
        dict_level_tot[level] = dict_level_tot.get(level, 0) + 1
        if level not in record_out:
            record_out[level] = [0.0] * n

        n_total = len(code_list)
        project = content_doc.get("project", "")
        repo_root = Path(repos_root) / project.replace("/", "---")
        if "standalone" in Path(validate_path).parts and not repo_root.exists():
            repo_root = Path(validate_path).parent
        c, per_results = _eval_code_list(
            code_list,
            validate_path,
            target_name,
            repos_root=str(Path(repos_root).resolve()),
            repo_root=str(repo_root.resolve()),
            project_root=str(Path(repos_root).resolve().parent),
            timeout=timeout,
        )

        # pass@k formula (standard unbiased estimator, same as the original OpenAI HumanEval)
        # Reference: 1 - C(n-c, k)/C(n, k)
        # Key point: range uses [n_fail+1, n+1], and the condition is n_fail < k
        for k_idx in range(n):
            k_val = k_idx + 1
            if n_total == 0:
                continue
            n_fail = n_total - c          # Number of failed samples
            if n_fail < k_val:            # Fewer than k failures → at least one must be correct
                delta = 1.0
            else:
                delta = float(1.0 - np.prod(
                    1.0 - k_val / np.arange(n_fail + 1, n_total + 1)
                ))
            tot_k[k_idx] += delta
            record_out[level][k_idx] += delta

        print(f"[PythonExec] {_id}  n={n_total}  c={c}  pass@1={tot_k[0]:.1f}/{count_tot}")

        if output_jsonl:
            row = {
                "_id": _id,
                "name": target_name,
                "level": level,
                "docstring": content_doc.get("docstring", ""),
                "generate_results": per_results,
            }
            if "project" in content_doc:
                row["project"] = content_doc["project"]
            output_rows.append(row)

    # 6) Aggregate output
    if output_jsonl and output_rows:
        os.makedirs(os.path.dirname(output_jsonl) or ".", exist_ok=True)
        with open(output_jsonl, "w", encoding="utf-8") as fw:
            for row in output_rows:
                fw.write(json.dumps(row, ensure_ascii=False) + "\n")

    pass_k_norm = [v / count_tot if count_tot else 0.0 for v in tot_k]
    by_level: Dict[str, List[float]] = {
        lvl: [v / dict_level_tot[lvl] if dict_level_tot[lvl] else 0.0
              for v in record_out[lvl]]
        for lvl in record_out
    }

    print("\n[PythonExec] ===== 聚合结果 =====")
    print(f"  总任务数: {count_tot}")
    for k_idx, pval in enumerate(pass_k_norm):
        print(f"  pass@{k_idx+1} = {pval:.4f}  ({pval*100:.2f}%)")
    print("  按难度分层:")
    for lvl, vals in by_level.items():
        n_lvl = dict_level_tot.get(lvl, 0)
        print(f"    {lvl}  ({n_lvl} tasks):", "  ".join(f"p@{i+1}={v*100:.1f}%" for i, v in enumerate(vals)))

    return {
        "total":    count_tot,
        "pass_k":   pass_k_norm,
        "by_level": by_level,
    }


# ============================================================
# Command-line entry point (compatible with the original PythonExec.py)
# ============================================================
if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="CoderEval batch evaluator")
    ap.add_argument("preds",  help="预测 JSONL 文件路径")
    ap.add_argument("n",      type=int, help="每题生成样本数（pass@k 的 n）")
    ap.add_argument("--repos-root", default="/home/travis/builds/repos",
                    help="repos 根目录（默认 /home/travis/builds/repos）")
    ap.add_argument("--index", default="src/datasets/CoderEval/CoderEval4Python.json",
                    help="CoderEval4Python.json 路径")
    ap.add_argument("--timeout", type=int, default=30, help="单次执行超时（秒）")
    ap.add_argument("--output", default=None, help="评测结果写出 JSONL 路径")
    args = ap.parse_args()

    result = run_pythonexec(
        preds_jsonl  = args.preds,
        index_json   = args.index,
        repos_root   = args.repos_root,
        n            = args.n,
        output_jsonl = args.output or args.preds + "_out.jsonl",
        timeout      = args.timeout,
    )
    sys.exit(0)
