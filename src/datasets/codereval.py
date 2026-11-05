# src/datasets/codereval.py
import json, os
from typing import Dict, List, Any, Optional

def build_codereval_index(codereval4python_json_path: str) -> Dict[str, dict]:
    with open(codereval4python_json_path, "r", encoding="utf-8") as f:
        obj = json.load(f)
    records = obj["RECORDS"] if isinstance(obj, dict) and "RECORDS" in obj else obj
    return {r.get("_id"): r for r in records if r.get("_id")}

def load_codereval_first_n(path: str, n: int = None) -> List[dict]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"CoderEval tasks not found: {path}")

    if path.endswith(".jsonl"):
        data = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                data.append(json.loads(line))
        return data[:n] if n is not None else data

    with open(path, "r", encoding="utf-8") as f:
        obj = json.load(f)

    if isinstance(obj, list):
        data = obj
    elif isinstance(obj, dict):
        if isinstance(obj.get("RECORDS"), list):
            data = obj["RECORDS"]
        elif isinstance(obj.get("tasks"), list):
            data = obj["tasks"]
        else:
            raise ValueError("Unsupported CoderEval json format.")
    else:
        raise ValueError("Unsupported CoderEval file type.")

    return data[:n] if n is not None else data

def project_to_repo_dir(project: str) -> str:
    # Key point: the repos directory uses names like neo4j---neo4j-python-driver
    return project.replace("/", "---")

def normalize_codereval_task(raw: dict, idx: int, repos_root: str, ce_index: dict) -> dict:
    """
    raw: from CoderEval-ALL.jsonl (the dataset with size=230)
    ce_index: RECORDS from CoderEval4Python.json (joined on _id)
    """
    # In the jsonl this is usually metadata.task_id: "CoderEval/<_id>"
    meta = raw.get("metadata", {})
    task_id_full = meta.get("task_id") or raw.get("task_id") or f"CoderEval/{idx}"
    qid = task_id_full.split("/")[-1]  # _id

    base = ce_index.get(qid)
    if base is None:
        # Do not raise; mark as skip directly to avoid crashing the whole run
        return {"task_id": task_id_full, "_id": qid, "skip": True, "reason": "id_not_in_index"}

    project = base.get("project")
    file_path = base.get("file_path")
    entry_point = base.get("name")
    start_line = int(base.get("lineno")) if base.get("lineno") else None
    end_line = int(base.get("end_lineno")) if base.get("end_lineno") else None

    if not project or not file_path or not entry_point or start_line is None or end_line is None:
        return {"task_id": task_id_full, "_id": qid, "skip": True, "reason": "missing_fields_in_index"}

    repo_dir = project_to_repo_dir(project)
    repo_path = os.path.join(repos_root, repo_dir)
    if not os.path.exists(repo_path):
        return {"task_id": task_id_full, "_id": qid, "skip": True, "reason": f"repo_not_found:{repo_path}"}

    # Prompt: prefer raw["prompt"] (CoderEval-ALL.jsonl itself contains the problem statement)
    prompt = (raw.get("prompt") or "").strip()
    if not prompt:
        # Fallback: human_label
        prompt = (base.get("human_label") or "").strip()

    docstring = (base.get("docstring") or raw.get("docstring") or "").strip()
    if docstring and docstring not in prompt:
        prompt += "\n\nDocstring:\n" + docstring

    return {
        "task_id": task_id_full,
        "_id": qid,
        "prompt": prompt,
        "repo": repo_dir,
        "repo_path": repo_path,
        "relpath": file_path,
        "entry_point": entry_point,
        "start_line": start_line,
        "end_line": end_line,
        "_raw": {"human": raw, "base": base},
    }


# ============================================================
# HumanEval-style integration of CoderEval
# Data source: CoderEval4Python.json (_id + file_content + docstring)
# Evaluation: PythonExec.py (_passk_validte.py file)
# ============================================================

# A few special path mappings (kept consistent with CoderEvalOriginalRunner._dict_std_nonestd)
_STD_NONESTD_REL: Dict[str, str] = {
    "standalone/neo4j-_meta-deprecated.py":
        "neo4j---neo4j-python-driver/src/neo4j/_meta_deprecated_passk_validte.py",
    "standalone/neo4j-work-query-unit_of_work.py":
        "neo4j---neo4j-python-driver/src/neo4j/_work/query_unit_of_work_passk_validte.py",
    "standalone/krake-krake-controller-kubernetes-hooks-on.py":
        "rak-n-rok---Krake/krake/krake/controller/kubernetes/hooks_on_passk_validte.py",
}

# Context character limit (file content before the function)
_MAX_CTX_CHARS = 3000


def build_validate_file_map(index_json_path: str, repos_root: str) -> Dict[str, str]:
    """
    Read CoderEval4Python.json and build the mapping _id -> absolute path of _passk_validte.py.
    Same logic as CoderEvalOriginalRunner.build_validate_file_map; can be called on its own.
    """
    with open(index_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    samples = data.get("RECORDS", [])

    repos_root = repos_root.rstrip("/") + "/"
    nonestd = {
        os.path.normpath(repos_root + k): os.path.normpath(repos_root + v)
        for k, v in _STD_NONESTD_REL.items()
    }

    id2file: Dict[str, str] = {}
    for doc in samples:
        _id = str(doc["_id"])
        name = doc.get("name", "")
        file_path = doc.get("file_path", "")

        # ---- standalone form (preferred) ----
        standalone = os.path.normpath(
            repos_root + "standalone/"
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

        # ---- in-repo form ----
        project = doc.get("project", "")
        repo_dir = project.replace("/", "---")
        if project == "neo4j/neo4j-python-driver":
            candidate = os.path.normpath(
                os.path.join(repos_root, repo_dir, "src", file_path)
            ).replace(".py", "_" + name + "_passk_validte.py")
        else:
            candidate = os.path.normpath(
                os.path.join(repos_root, repo_dir, file_path)
            ).replace(".py", "_" + name + "_passk_validte.py")

        if candidate in nonestd:
            candidate = nonestd[candidate]
        if os.path.exists(candidate):
            id2file[_id] = candidate

    return id2file


def load_codereval_simple(
    index_json_path: str,
    repos_root: Optional[str] = None,
    n: Optional[int] = None,
) -> List[dict]:
    """
    Load CoderEval tasks from CoderEval4Python.json in HumanEval style.

    Each task returns:
      _id           : original _id string
      task_id       : "CoderEval/{_id}"
      name          : function name
      level         : self / file / project / ... difficulty level
      project       : original project field (e.g., "neo4j/neo4j-python-driver")
      prompt        : HumanEval-style prompt
                      = file context (imports and surrounding code before the function)
                        + function signature line
                        + docstring
      canonical_code: code field (reference implementation, used for vs-canonical reproducibility metrics)
      validate_path : absolute path of _passk_validte.py (used for evaluation)
      metadata      : fields compatible with CoderEvalOriginalRunner.eval_one
                      {"task_id": ..., "function_name": ...}

    Tasks without a _passk_validte.py are skipped (same behavior as eval_original).

    Parameters:
      index_json_path : path to CoderEval4Python.json
      repos_root      : root directory of repos (containing the _passk_validte files);
                        if None, validate-file filtering is skipped
      n               : take only the first n records (None = all)
    """
    with open(index_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    records = data.get("RECORDS", [])

    # Build the validate-file mapping (filtering is applied only when repos_root is given)
    id2file: Dict[str, str] = {}
    if repos_root is not None:
        id2file = build_validate_file_map(index_json_path, repos_root)

    tasks: List[dict] = []
    for rec in records:
        _id = str(rec["_id"])
        name = rec.get("name", "")
        if not name:
            continue

        validate_path = id2file.get(_id)
        if repos_root is not None and not validate_path:
            continue   # No test stub; skip

        docstring = (rec.get("docstring") or "").strip()
        file_content = rec.get("file_content") or ""
        canonical_code = (rec.get("code") or "").strip()
        level = rec.get("level", "UNKNOWN")
        project = rec.get("project", "")

        # ---- Extract file context: code before the function's start line ----
        lineno = 0
        raw_lineno = rec.get("lineno")
        if raw_lineno:
            try:
                lineno = int(raw_lineno)
            except (ValueError, TypeError):
                lineno = 0

        if lineno > 1 and file_content:
            ctx_lines = file_content.split("\n")[:lineno - 1]
            ctx = "\n".join(ctx_lines)
            # Keep only the part closest to the function to avoid an overly long prompt
            if len(ctx) > _MAX_CTX_CHARS:
                ctx = ctx[-_MAX_CTX_CHARS:]
        else:
            ctx = ""

        # ---- Extract the function signature (first line "def ..." of canonical_code) ----
        sig_line = ""
        for line in canonical_code.split("\n"):
            if line.startswith("def "):
                sig_line = line.rstrip()
                break
        if not sig_line:
            sig_line = f"def {name}(...):"

        # ---- Assemble the HumanEval-style prompt ----
        parts: List[str] = []
        if ctx.strip():
            parts.append(ctx.strip())
            parts.append("")   # Blank line separator
        if docstring:
            parts.append(sig_line)
            parts.append(f'    """{docstring}"""')
        else:
            parts.append(sig_line)
        prompt = "\n".join(parts) + "\n"

        tasks.append({
            "_id": _id,
            "task_id": f"CoderEval/{_id}",
            "name": name,
            "level": level,
            "project": project,
            "prompt": prompt,
            "canonical_code": canonical_code,
            "validate_path": validate_path,
            # Metadata format required by CoderEvalOriginalRunner.eval_one
            "metadata": {
                "task_id": f"CoderEval/{_id}",
                "function_name": name,
            },
        })

        if n is not None and len(tasks) >= n:
            break

    return tasks