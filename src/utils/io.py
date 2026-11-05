# utils/io.py
# import os, json, time, platform, yaml
# from pathlib import Path
# from typing import Dict, Any, List
#
# def make_run_dir(out_dir: str, name: str) -> str:
#     """
#     Create the runs/<name>/<YYYY-mm-dd_HH-MM-SS>/ directory
#     """
#     ts = time.strftime("%Y-%m-%d_%H-%M-%S")
#     run_dir = os.path.join(out_dir, name, ts)
#     os.makedirs(run_dir, exist_ok=True)
#     return run_dir
#
# def dump_config_snapshot(run_dir: str, cfg: Dict[str, Any]) -> str:
#     """
#     Save a snapshot of this experiment's configuration to config.yaml
#     """
#     p = os.path.join(run_dir, "config.yaml")
#     with open(p, "w", encoding="utf-8") as f:
#         yaml.safe_dump(cfg, f, allow_unicode=True)
#     print(f"🧾 Config snapshot saved to: {p}")
#     return p
#
# def write_jsonl(path: str, rows: List[Dict[str, Any]]) -> None:
#     """
#     Save per-sample records to JSONL
#     """
#     Path(path).parent.mkdir(parents=True, exist_ok=True)
#     with open(path, "w", encoding="utf-8") as f:
#         for r in rows:
#             f.write(json.dumps(r, ensure_ascii=False) + "\n")
#     print(f"📝 Preds saved to: {path}")
#
# def save_summary(run_dir: str,
#                  cfg: Dict[str, Any],
#                  metrics: Dict[str, Any],
#                  extra: Dict[str, Any] | None = None) -> str:
#     """
#     Save summary metrics to summary.json
#     Example metrics:
#       {"exact_match": 0.595, "pass@1": 0.595, "pass@5": 0.731, "total": 800}
#     """
#     payload = {
#         "experiment": cfg.get("experiment", {}).get("name", "exp"),
#         "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
#         "dataset": {
#             "name": cfg.get("experiment", {}).get("dataset", "unknown"),
#             "split": cfg.get("experiment", {}).get("split", "test"),
#             "num_tasks": metrics.get("total", None),
#             "num_samples_per_task": cfg.get("experiment", {}).get("num_samples", 1),
#         },
#         "metrics": metrics,
#         "model": cfg.get("model", {}),
#         "prompt": cfg.get("prompt", {}),
#         "env": {
#             "python": platform.python_version(),
#             **({} if extra is None else extra),
#         },
#         "files": {
#             "config": os.path.join(run_dir, "config.yaml"),
#             "preds": os.path.join(run_dir, "preds.jsonl")
#         }
#     }
#     p = os.path.join(run_dir, "summary.json")
#     with open(p, "w", encoding="utf-8") as f:
#         json.dump(payload, f, ensure_ascii=False, indent=2)
#     print(f"📘 Summary saved to: {p}")
#     return p


# utils/io.py
import os, json, time, platform, yaml
from pathlib import Path
from typing import Dict, Any, List


# ============================================================
# Create the run directory
# ============================================================
def make_run_dir(out_dir: str, name: str) -> str:
    """
    Create the runs/<name>/<YYYY-mm-dd_HH-MM-SS>/ directory
    """
    ts = time.strftime("%Y-%m-%d_%H-%M-%S")
    run_dir = os.path.join(out_dir, name, ts)
    os.makedirs(run_dir, exist_ok=True)
    return run_dir


# ============================================================
# Save the configuration snapshot
# ============================================================
def dump_config_snapshot(run_dir: str, cfg: Dict[str, Any]) -> str:
    """
    Save a snapshot of this experiment's configuration to config.yaml
    """
    p = os.path.join(run_dir, "config.yaml")
    with open(p, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, allow_unicode=True)
    print(f"🧾 Config snapshot saved to: {p}")
    return p


# ============================================================
# Save per-sample records (JSONL or pretty)
# ============================================================
def write_jsonl(path: str, rows: List[Dict[str, Any]], pretty: bool = True) -> None:
    """
    Save per-sample records to JSONL (compact format by default)
    With pretty=True each sample is indented (more readable)
    """
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            if pretty:
                f.write(json.dumps(r, ensure_ascii=False, indent=4))
                f.write("\n\n")  # A blank line between records
            else:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"📝 Preds saved to: {path}")


def write_json_pretty(path: str, rows: List[Dict[str, Any]]) -> None:
    """
    Save the full records as pretty multi-line JSON (not JSONL)
    for presentation or manual inspection.
    """
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=4)
    print(f"🧮 Pretty JSON saved to: {path}")


# ============================================================
# Save summary metrics to summary.json
# ============================================================
def save_summary(run_dir: str,
                 cfg: Dict[str, Any],
                 metrics: Dict[str, Any],
                 extra: Dict[str, Any] | None = None) -> str:
    """
    Save summary metrics to summary.json
    Example metrics:
      {"exact_match": 0.595, "pass@1": 0.595, "pass@5": 0.731, "total": 800}
    """
    payload = {
        "experiment": cfg.get("experiment", {}).get("name", "exp"),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "dataset": {
            "name": cfg.get("experiment", {}).get("dataset", "unknown"),
            "split": cfg.get("experiment", {}).get("split", "test"),
            "num_tasks": metrics.get("total", None),
            "num_samples_per_task": cfg.get("experiment", {}).get("num_samples", 1),
        },
        "metrics": metrics,
        "model": cfg.get("model", {}),
        "prompt": cfg.get("prompt", {}),
        "env": {
            "python": platform.python_version(),
            **({} if extra is None else extra),
        },
        "files": {
            "config": os.path.join(run_dir, "config.yaml"),
            "preds": os.path.join(run_dir, "preds.jsonl")
        }
    }
    p = os.path.join(run_dir, "summary.json")
    with open(p, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(f"📘 Summary saved to: {p}")
    return p

def init_run_and_preds(out_dir: str, name: str, cfg: Dict[str, Any]) -> tuple[str, str]:
    """
    Generic: create run_dir for an experiment, save the config, and prepare an empty preds.jsonl file
    Returns: (run_dir, preds_path)
    """
    run_dir = make_run_dir(out_dir, name)
    dump_config_snapshot(run_dir, cfg)

    preds_path = os.path.join(run_dir, "preds.jsonl")
    Path(preds_path).parent.mkdir(parents=True, exist_ok=True)
    with open(preds_path, "w", encoding="utf-8"):
        pass

    return run_dir, preds_path

def append_jsonl(path: str, record: Dict[str, Any], pretty: bool = False) -> None:
    """
    Generic: append one record to a JSONL file (written to disk immediately)
    pretty=False  -> standard JSONL, one sample per line
    pretty=True   -> multi-line indented, for human reading only; no longer strict JSONL
    """
    Path(path).parent.mkdir(parents=True, exist_ok=True)

    if pretty:
        json_str = json.dumps(record, ensure_ascii=False, indent=4) + "\n\n"
    else:
        json_str = json.dumps(record, ensure_ascii=False) + "\n"

    with open(path, "a", encoding="utf-8") as f:
        f.write(json_str)
        f.flush()

# def append_jsonl(path: str, record: Dict[str, Any]) -> None:
#     """
#     Generic: append one record to a JSONL file (written to disk immediately)
#     """
#     Path(path).parent.mkdir(parents=True, exist_ok=True)
#     with open(path, "a", encoding="utf-8") as f:
#         f.write(json.dumps(record, ensure_ascii=False) + "\n")


