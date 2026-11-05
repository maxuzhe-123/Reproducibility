from __future__ import annotations

import copy
import json
import os
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List

import yaml


SECRET_KEYS = {
    "api_key",
    "apikey",
    "authorization",
    "openrouter_api_key",
    "token",
}


def load_config(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def sanitized_config(cfg: Dict[str, Any]) -> Dict[str, Any]:
    clean = copy.deepcopy(cfg)

    def scrub(value: Any) -> Any:
        if isinstance(value, dict):
            return {
                key: ("<redacted>" if key.lower() in SECRET_KEYS else scrub(item))
                for key, item in value.items()
            }
        if isinstance(value, list):
            return [scrub(item) for item in value]
        return value

    return scrub(clean)


def write_yaml(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(payload, f, allow_unicode=True, sort_keys=False)


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2, allow_nan=False)


def append_jsonl(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=False, allow_nan=False) + "\n")
        f.flush()


def load_json_objects(path: Path) -> List[Dict[str, Any]]:
    """Load strict JSONL and the repository's historical pretty-JSONL format."""
    if not path.exists():
        return []
    content = path.read_text(encoding="utf-8")
    decoder = json.JSONDecoder()
    records: List[Dict[str, Any]] = []
    pos = 0
    while pos < len(content):
        while pos < len(content) and content[pos].isspace():
            pos += 1
        if pos >= len(content):
            break
        try:
            obj, end = decoder.raw_decode(content, pos)
        except json.JSONDecodeError:
            next_line = content.find("\n", pos)
            if next_line == -1:
                break
            pos = next_line + 1
            continue
        records.append(obj)
        pos = end
    return records


def latest_by_sample(records: Iterable[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    latest: Dict[str, Dict[str, Any]] = {}
    for record in records:
        sample_id = record.get("sample_id")
        if sample_id:
            latest[str(sample_id)] = record
    return latest


def compact_jsonl(path: Path) -> int:
    """Rewrite a JSONL file keeping only the latest record for each sample_id, in order of first appearance.

    Generation uses append-mode writes, so rerunning the same sample_id leaves old "overwritten" lines (downstream readers
    take the latest via latest_by_sample, so the logic is already correct). This function physically removes those old lines
    from the file so that rerun results directly replace the original records. Returns the number of remaining records.
    """
    records = load_json_objects(path)
    if not records:
        return 0
    latest: Dict[str, Dict[str, Any]] = {}
    order: List[str] = []
    for record in records:
        sample_id = record.get("sample_id")
        if not sample_id:
            continue
        sample_id = str(sample_id)
        if sample_id not in latest:
            order.append(sample_id)
        latest[sample_id] = record

    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        for sample_id in order:
            f.write(
                json.dumps(latest[sample_id], ensure_ascii=False, allow_nan=False)
                + "\n"
            )
        f.flush()
    tmp.replace(path)
    return len(order)


def timestamp() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


def run_timestamp() -> str:
    return time.strftime("%Y-%m-%d_%H-%M-%S")


def resolve_generation_root(cfg: Dict[str, Any]) -> Path:
    configured = cfg.get("pipeline", {}).get("generation_root")
    if configured:
        return Path(configured)
    return Path("runs_v2") / "preds"


def resolve_evaluation_root(cfg: Dict[str, Any]) -> Path:
    configured = cfg.get("pipeline", {}).get("evaluation_root")
    if configured:
        return Path(configured)
    # The dataset is now provided by experiment_relpath (<dataset>/<model>/<sub>); only the root directory is given here.
    return Path("runs_v2") / "evaluations"


def model_snapshot(cfg: Dict[str, Any]) -> Dict[str, Any]:
    model = dict(cfg.get("model", {}))
    for key in list(model):
        if key.lower() in SECRET_KEYS:
            model.pop(key, None)
    return model


def config_dataset_path(cfg: Dict[str, Any], default: str) -> str:
    return str(
        cfg.get("experiment", {}).get(
            "dataset_path",
            cfg.get("dataset_path", default),
        )
    )


def ensure_project_root() -> Path:
    return Path(os.getcwd()).resolve()
