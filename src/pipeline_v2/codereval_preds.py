from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Tuple

from src.datasets.codereval import load_codereval_simple
from src.pipeline_v2.generation_core import PreparedTask, generate_prepared_tasks
from src.prompts.coding import build_user_prompt
from src.utils.repocoder.codereval_adapter import CoderEvalRepoCoderAdapter


def load_codereval_prompt_sources(index_path: str) -> Tuple[dict, dict]:
    base = Path(index_path).parent
    outputs = []
    for name in ("CEPythonHumanLabel.jsonl", "CEPythonRaw.jsonl"):
        mapping = {}
        path = base / name
        if path.exists():
            with path.open(encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        item = json.loads(line)
                        mapping[str(item["question_id"])] = item
        outputs.append(mapping)
    return outputs[0], outputs[1]


def trim_lsp_context(context: str, max_chars: int = 3500) -> str:
    if len(context) <= max_chars:
        return context
    if "<PLACEHOLDER>" not in context:
        return context[:max_chars]
    index = context.index("<PLACEHOLDER>")
    start = max(0, index - max_chars // 2)
    end = min(len(context), start + max_chars)
    start = max(0, end - max_chars)
    if start:
        newline = context.rfind("\n", 0, start)
        if newline != -1:
            start = newline + 1
    return context[start:end]


def prepare_codereval_tasks(cfg: Dict[str, Any]) -> List[PreparedTask]:
    ce_cfg = cfg.get("codereval", {})
    rag_cfg = cfg.get("rag", {})
    mode = rag_cfg.get("mode", "no_rag")
    limit = int(cfg["experiment"]["num_tasks"])
    index_path = ce_cfg.get(
        "index_path", "src/datasets/CoderEval/CoderEval4Python.json"
    )
    repos_root = ce_cfg.get("repos_root", "repos")
    runner = CoderEvalRepoCoderAdapter(cfg)

    prepared = []
    if mode == "no_rag":
        human, raw = load_codereval_prompt_sources(index_path)
        source = ce_cfg.get("docstring_source", "human_label")
        tasks = load_codereval_simple(
            index_json_path=os.path.abspath(index_path),
            repos_root=os.path.abspath(repos_root),
            n=limit,
        )
        for task in tasks:
            primary = raw if source == "raw" else human
            fallback = human if source == "raw" else raw
            row = primary.get(task["_id"]) or fallback.get(task["_id"])
            prompt = row["input"] if row else task["prompt"]
            metadata = {
                "_id": task["_id"],
                "level": task.get("level"),
                "project": task.get("project"),
                "rag": {"mode": mode, "retrieval_used": False},
            }
            prepared.append((task, prompt, metadata))
        return prepared

    tasks = runner.load_tasks(num_tasks=limit)
    runner.prepare_rg1_prompts()
    index_records = json.loads(Path(index_path).read_text(encoding="utf-8"))[
        "RECORDS"
    ]
    levels = {str(item["_id"]): item.get("level", "UNKNOWN") for item in index_records}
    template = cfg.get("prompt", {}).get("template", "codereval")
    for task in tasks:
        task_id = task["metadata"]["task_id"]
        sample_id = task_id.split("/")[-1]
        repo_name = task_id.split("/")[0]
        effective = runner.get_effective_prompt(task, rag_mode=mode)
        context = trim_lsp_context(task.get("metadata", {}).get("lsp-context", ""))
        prompt_data = {
            "prompt": (
                "File context (imports/types in this file):\n"
                f"{context}\n\n"
                "Target function (replace this function only):\n"
                f"{effective}\n"
            ),
            "repo": repo_name,
        }
        prompt = build_user_prompt(template, prompt_data)
        metadata = {
            "_id": sample_id,
            "level": levels.get(sample_id, "UNKNOWN"),
            "project": repo_name,
            "rag": {
                "mode": mode,
                "retrieval_used": True,
                "top_k": int(rag_cfg.get("top_k", 3)),
            },
        }
        prepared.append((task, prompt, metadata))
    return prepared


def codereval_gen(
    cfg: Dict[str, Any],
    model,
    run_dir: Path,
    existing: Dict[str, Dict[str, Any]],
    resume: bool = True,
    retry_failed: bool = True,
) -> Path:
    prepared_tasks = prepare_codereval_tasks(cfg)
    return generate_prepared_tasks(
        cfg,
        model,
        run_dir,
        prepared_tasks,
        existing,
        resume,
        retry_failed,
    )
