from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

from src.defenses.humaneval_defenses import HumanEvalDefense
from src.pipeline_v2.common import config_dataset_path
from src.pipeline_v2.generation_core import PreparedTask, generate_prepared_tasks
from src.prompts.coding import build_user_prompt


def load_humaneval_tasks(cfg: Dict[str, Any]) -> List[Dict[str, Any]]:
    path = config_dataset_path(cfg, "src/datasets/HumanEval.jsonl")
    # task_ids (highest priority): load only these tasks, for single-task/subset reruns; num_tasks is ignored.
    raw_ids = cfg["experiment"].get("task_ids")
    task_ids = {str(t) for t in raw_ids} if raw_ids else None
    limit = int(cfg["experiment"]["num_tasks"])
    tasks = []
    with open(path, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            if not line.strip():
                continue
            item = json.loads(line)
            task_id = item.get("task_id", f"HumanEval/{idx}")
            if task_ids is not None:
                if task_id not in task_ids:
                    continue
            elif len(tasks) >= limit:
                break
            tasks.append(
                {
                    "idx": idx,
                    "task_id": task_id,
                    "prompt": item["prompt"],
                    "entry_point": item.get("entry_point"),
                }
            )
    if task_ids is not None:
        found = {t["task_id"] for t in tasks}
        missing = task_ids - found
        if missing:
            print(f"[WARN] task_ids not found in dataset: {sorted(missing)}")
        print(f"[HumanEval] task_ids filter -> {len(tasks)} task(s): {[t['task_id'] for t in tasks]}")
    return tasks


def prepare_humaneval_tasks(
    cfg: Dict[str, Any],
    model,
    existing: Dict[str, Dict[str, Any]] | None = None,
) -> List[PreparedTask]:
    base_template = cfg.get("prompt", {}).get("template", "humaneval_original")
    defense_name = cfg.get("prompt", {}).get("defense", "no")
    defense = HumanEvalDefense(name=defense_name, lm=model, cfg=cfg)
    template = defense.get_template(base_template)
    prepared = []
    for task in load_humaneval_tasks(cfg):
        previous = next(
            (
                record
                for record in (existing or {}).values()
                if record.get("task_id") == task["task_id"]
            ),
            None,
        )
        if previous is not None:
            prepared.append(
                (
                    task,
                    previous.get("prompt", ""),
                    dict(previous.get("task_metadata", {})),
                )
            )
            continue
        blocked = defense.prehand_detect(task["prompt"])
        effective_prompt = "" if blocked else defense.preprocess(task["prompt"])
        user_prompt = (
            ""
            if blocked
            else defense.postprocess_prompt(
                build_user_prompt(template, effective_prompt)
            )
        )
        metadata = {
            "idx": task["idx"],
            "entry_point": task.get("entry_point"),
            "defense": defense_name,
            "effective_prompt": effective_prompt,
            "blocked_by_defense": blocked,
        }
        prepared.append((task, user_prompt, metadata))
    return prepared


def humaneval_gen(
    cfg: Dict[str, Any],
    model,
    run_dir: Path,
    existing: Dict[str, Dict[str, Any]],
    resume: bool = True,
    retry_failed: bool = True,
) -> Path:
    prepared_tasks = prepare_humaneval_tasks(cfg, model, existing=existing)
    return generate_prepared_tasks(
        cfg,
        model,
        run_dir,
        prepared_tasks,
        existing,
        resume,
        retry_failed,
    )
