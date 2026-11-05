from __future__ import annotations

import argparse
from pathlib import Path
from typing import Callable, Dict

from src.pipeline_v2.codejudge_eval import codejudge_eval
from src.pipeline_v2.codereval_eval import codereval_eval
from src.pipeline_v2.common import load_config
from src.pipeline_v2.evaluation_core import load_generation_config
from src.pipeline_v2.humaneval_eval import humaneval_eval


EvaluationHandler = Callable[..., Path]
EVALUATION_HANDLERS: Dict[str, EvaluationHandler] = {
    "humaneval": humaneval_eval,
    "codereval": codereval_eval,
    "codejudgebench": codejudge_eval,
}


def run_evaluation(
    config_path: str | None = None,
    generation_run_path: str | None = None,
    output_dir_path: str | None = None,
) -> Path:
    # cfg source: prefer an explicit --config (gen no longer writes config.yaml to the preds directory, keeping only jsonl);
    # otherwise fall back to generation_run/config.yaml (compatible with older outputs).
    if config_path:
        cfg = load_config(config_path)
    elif generation_run_path:
        cfg = load_generation_config(Path(generation_run_path).resolve())
    else:
        raise ValueError("请提供 --config（推荐）或 --generation-run 其一")

    # gen directory: use it if given explicitly; otherwise derive <root>/<dataset>/<model>/<sub> from cfg using the unified layout.
    if generation_run_path:
        generation_run = Path(generation_run_path).resolve()
    else:
        from src.pipeline_v2.common import resolve_generation_root
        from src.pipeline_v2.generation_core import experiment_relpath

        generation_run = (
            resolve_generation_root(cfg) / experiment_relpath(cfg)
        ).resolve()
    dataset = str(cfg["experiment"]["dataset"]).lower()
    handler = EVALUATION_HANDLERS.get(dataset)
    if handler is None:
        raise ValueError(
            f"Unsupported dataset: {dataset}. "
            f"Available: {sorted(EVALUATION_HANDLERS)}"
        )
    return handler(
        generation_run=generation_run,
        cfg=cfg,
        output_dir_path=output_dir_path,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate fixed generations.")
    parser.add_argument(
        "--config",
        help="实验 config（推荐）；据此自动定位 preds 目录与文件名",
    )
    parser.add_argument(
        "--generation-run",
        help="预测目录；不传则按 --config 自动推导 <root>/<dataset>/<model>/<sub>",
    )
    parser.add_argument("--output-dir")
    args = parser.parse_args()
    run_evaluation(args.config, args.generation_run, args.output_dir)


if __name__ == "__main__":
    main()
