from __future__ import annotations

import hashlib
import os
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

from src.models.openrouter import OpenRouterLM, build_reasoning_payload
from src.pipeline_v2.common import (
    append_jsonl,
    model_snapshot,
    resolve_generation_root,
    timestamp,
)


SCHEMA_VERSION = 1
PreparedTask = Tuple[Dict[str, Any], str, Dict[str, Any]]


def build_model(cfg: Dict[str, Any]):
    model_cfg = cfg["model"]
    provider = str(model_cfg.get("provider", "openrouter")).lower()
    if provider == "openrouter":
        # Thinking control: translate the thinking switch in the config into the OpenRouter reasoning field.
        reasoning, thinking_budget = build_reasoning_payload(model_cfg.get("thinking"))
        return OpenRouterLM(
            model_id=model_cfg["model_id"],
            api_key=model_cfg.get("api_key") or os.getenv("OPENROUTER_API_KEY"),
            stop_strings=model_cfg.get("stop_strings", ["Stop."]),
            do_sample=model_cfg.get("do_sample", True),
            top_k=model_cfg.get("top_k"),
            top_p=model_cfg.get("top_p"),
            temperature=model_cfg.get("temperature", 1.0),
            max_retries=int(model_cfg.get("max_retries", 3)),
            reasoning=reasoning,
            thinking_budget=thinking_budget,
        )
    raise ValueError(f"Unsupported model provider: {provider}")


def _filename_value(value: Any) -> str:
    if value is None:
        return "none"
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value).lower()


def _filename_token(value: Any) -> str:
    token = re.sub(r"[^a-z0-9.]+", "_", _filename_value(value))
    return token.strip("_") or "unknown"


def dataset_token(cfg: Dict[str, Any]) -> str:
    return _filename_token(cfg["experiment"]["dataset"])


def model_token(cfg: Dict[str, Any]) -> str:
    return _filename_token(str(cfg["model"]["model_id"]).rstrip("/").split("/")[-1])


def experiment_subdir(cfg: Dict[str, Any]) -> Path:
    """Relative subdirectory of the experiment under <dataset>/<model>/:
    - regular experiment → <template>/
    - defense experiment → defense/<defense_name>/

    Rule: prompt.defense is a real defense, or experiment.name contains 'defense'.
    (Some configs have experiment.name=None, so the prompt.defense field takes precedence and the experiment name is secondary.)
    """
    exp_name = str(cfg.get("experiment", {}).get("name") or "")
    defense = cfg.get("prompt", {}).get("defense")
    defense_real = bool(defense) and str(defense).strip().lower() not in ("no", "none", "")
    if defense_real or ("defense" in exp_name.lower()):
        if defense_real:
            dname = defense
        else:
            # Defenses implemented via templates (e.g., instructional, defense=no): take the defense name from the experiment name
            m = re.search(r"defense[_\-](.+)$", exp_name.lower())
            dname = m.group(1) if m else (exp_name.lower() or "unknown")
        return Path("defense") / _filename_token(dname)
    template = _filename_token(cfg.get("prompt", {}).get("template") or "no_template")
    return Path(template)


def experiment_relpath(cfg: Dict[str, Any]) -> Path:
    """<dataset>/<model>/<template or defense/<name>>  — the same layout is shared by preds and eval."""
    return Path(dataset_token(cfg)) / model_token(cfg) / experiment_subdir(cfg)


def _use_thinking_naming(cfg: Dict[str, Any]) -> bool:
    """Whether to use the thinking file name (tem_<t>_thinking_<size>).

    Enabled only when the model supports thinking (Gemini / Claude) and the config sets the `thinking` switch,
    so existing experiments without a thinking block (humaneval1, etc.) keep the original
    top_p/top_k naming and do not change paths or break resume because of this change.
    """
    mid = str(cfg["model"].get("model_id", "")).lower()
    is_thinking_model = any(k in mid for k in ("gemini", "claude", "anthropic"))
    return is_thinking_model and bool(cfg["model"].get("thinking"))


def thinking_token(cfg: Dict[str, Any]) -> str:
    """Thinking size identifier used in file names:
    - disabled / budget 0      → "0"
    - mode=effort              → level name (e.g., high)
    - mode=budget + number     → that number (e.g., 4096)
    - dynamic thinking (no budget given) → "dynamic"
    """
    thinking = cfg["model"].get("thinking") or {}
    if not thinking.get("enabled", True):
        return "0"
    mode = str(thinking.get("mode") or "budget").lower()
    if mode == "effort":
        effort = thinking.get("effort")
        return _filename_token(effort) if effort else "dynamic"
    budget = thinking.get("budget", None)
    if budget == 0:
        return "0"
    if budget is None:
        return "dynamic"
    return _filename_token(budget)


def sampling_token(cfg: Dict[str, Any]) -> str:
    """Sampling parameter identifier (without extension).

    - Gemini (pro/flash): tem_<t>_thinking_<size>, without top_p/top_k.
    - Other models: tem_<t>_top_p_<p>_top_k_<k> (unchanged).
    """
    model_cfg = cfg["model"]
    if _use_thinking_naming(cfg):
        return (
            f"tem_{_filename_token(model_cfg.get('temperature', 1.0))}"
            f"_thinking_{thinking_token(cfg)}"
        )
    return (
        f"tem_{_filename_token(model_cfg.get('temperature', 1.0))}"
        f"_top_p_{_filename_token(model_cfg.get('top_p'))}"
        f"_top_k_{_filename_token(model_cfg.get('top_k'))}"
    )


def prediction_filename(cfg: Dict[str, Any]) -> str:
    # The file name consists only of sampling parameters; dataset/model/template (or defense) are encoded in the directory hierarchy.
    return sampling_token(cfg) + ".jsonl"


def create_run_dir(cfg: Dict[str, Any], requested: str | None) -> Path:
    if requested:
        run_dir = Path(requested)
    else:
        run_dir = resolve_generation_root(cfg) / experiment_relpath(cfg)
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def prediction_path(cfg: Dict[str, Any], run_dir: Path) -> Path:
    return run_dir / prediction_filename(cfg)


def generate_prepared_tasks(
    cfg: Dict[str, Any],
    model,
    run_dir: Path,
    prepared_tasks: List[PreparedTask],
    existing: Dict[str, Dict[str, Any]],
    resume: bool = True,
    retry_failed: bool = True,
) -> Path:
    generations_path = prediction_path(cfg, run_dir)
    num_samples = int(cfg["experiment"].get("num_samples", 1))
    max_new_tokens = int(cfg["model"].get("max_new_tokens", 512))
    # Forced regeneration: ignore resume's "skip successful samples" logic and regenerate the tasks included in this run.
    # Combined with experiment.task_ids, this allows "rerun a single task and replace" existing results (including truncated successes).
    force_regenerate = bool(cfg["experiment"].get("force_regenerate", False))
    success = failed = skipped = 0

    for enum_index, (task, prompt, task_metadata) in enumerate(prepared_tasks):
        task_id = str(task.get("task_id") or task["metadata"]["task_id"])
        # Use the fixed index within the dataset as task_index so that single-task/subset reruns match the full run
        # and are not misaligned because only a few tasks are run (falls back to the enumeration index if missing).
        task_index = task.get("idx", enum_index)
        if task_metadata.get("blocked_by_defense"):
            for sample_index in range(num_samples):
                sample_id = f"{task_id}::{sample_index}"
                current = existing.get(sample_id)
                if (
                    resume
                    and current
                    and current.get("status") == "blocked"
                    and not force_regenerate
                ):
                    skipped += 1
                    continue
                record = {
                    "schema_version": SCHEMA_VERSION,
                    "sample_id": sample_id,
                    "task_id": task_id,
                    "task_index": task_index,
                    "sample_index": sample_index,
                    "status": "blocked",
                    "prompt": prompt,
                    "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
                    "raw": None,
                    "error": None,
                    "model": model_snapshot(cfg),
                    "response_metadata": {},
                    "task_metadata": task_metadata,
                    "created_at": timestamp(),
                }
                append_jsonl(generations_path, record)
                existing[sample_id] = record
            continue

        for sample_index in range(num_samples):
            sample_id = f"{task_id}::{sample_index}"
            current = existing.get(sample_id)
            if resume and current and not force_regenerate:
                if current.get("status") == "success":
                    skipped += 1
                    continue
                if not retry_failed:
                    skipped += 1
                    continue

            started = time.monotonic()
            if hasattr(model, "last_finish_reason"):
                model.last_finish_reason = None
            try:
                raw = model.generate(
                    system=None,
                    user=prompt,
                    max_new_tokens=max_new_tokens,
                )
                status = "success"
                error = None
                success += 1
            except Exception as exc:
                raw = None
                # Distinguish blank/degenerate outputs (the model emitting whitespace in a repetition loop) from genuine API/network errors,
                # recording them as empty_generation so that retry_failed can rerun them.
                if "OPENROUTER_EMPTY_CONTENT" in str(exc):
                    status = "empty_generation"
                else:
                    status = "generation_error"
                error = {"type": type(exc).__name__, "message": str(exc)}
                failed += 1

            record = {
                "schema_version": SCHEMA_VERSION,
                "sample_id": sample_id,
                "task_id": task_id,
                "task_index": task_index,
                "sample_index": sample_index,
                "status": status,
                "prompt": prompt,
                "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
                "raw": raw,
                "error": error,
                "model": model_snapshot(cfg),
                "response_metadata": {
                    "latency_sec": round(time.monotonic() - started, 6),
                    "adapter": type(model).__name__,
                    "finish_reason": getattr(model, "last_finish_reason", None),
                },
                "task_metadata": task_metadata,
                "created_at": timestamp(),
            }
            append_jsonl(generations_path, record)
            existing[sample_id] = record
            print(
                f"[{task_index + 1}/{len(prepared_tasks)}] "
                f"{sample_id} -> {status}"
            )

    print(f"Predictions: {generations_path}")
    return run_dir
