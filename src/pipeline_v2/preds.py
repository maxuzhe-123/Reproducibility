from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Callable, Dict

from src.pipeline_v2.codejudge_preds import codejudge_gen
from src.pipeline_v2.codereval_preds import codereval_gen
from src.pipeline_v2.common import (
    compact_jsonl,
    latest_by_sample,
    load_config,
    load_json_objects,
)
from src.pipeline_v2.generation_core import (
    build_model,
    create_run_dir,
    prediction_path,
)
from src.pipeline_v2.humaneval_preds import humaneval_gen


GenerationHandler = Callable[..., Path]
GENERATION_HANDLERS: Dict[str, GenerationHandler] = {
    "humaneval": humaneval_gen,
    "codereval": codereval_gen,
    "codejudgebench": codejudge_gen,
}


def run_generation(
    config_path: str,
    run_dir_path: str | None = None,
    resume: bool = True,
    retry_failed: bool = True,
) -> Path:
    cfg: Dict[str, Any] = load_config(config_path)
    dataset = str(cfg["experiment"]["dataset"]).lower()
    handler = GENERATION_HANDLERS.get(dataset)
    if handler is None:
        raise ValueError(
            f"Unsupported dataset: {dataset}. "
            f"Available: {sorted(GENERATION_HANDLERS)}"
        )

    run_dir = create_run_dir(cfg, run_dir_path)
    generations_path = prediction_path(cfg, run_dir)
    model = build_model(cfg)
    existing = latest_by_sample(load_json_objects(generations_path))
    result = handler(
        cfg=cfg,
        model=model,
        run_dir=run_dir,
        existing=existing,
        resume=resume,
        retry_failed=retry_failed,
    )
    # Compact by default: remove old lines appended by reruns so that regenerated results directly replace the original records in the file.
    if cfg.get("experiment", {}).get("compact", True):
        kept = compact_jsonl(generations_path)
        print(f"Compacted predictions: {generations_path} ({kept} records)")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate predictions only.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--run-dir")
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--no-retry-failed", action="store_true")
    args = parser.parse_args()
    run_generation(
        config_path=args.config,
        run_dir_path=args.run_dir,
        resume=not args.no_resume,
        retry_failed=not args.no_retry_failed,
    )


if __name__ == "__main__":
    main()
