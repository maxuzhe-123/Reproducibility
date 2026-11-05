"""Merge the 15 parallel-run shards of Gemini 2.5 Flash on CodeJudgeBench into
the canonical preds path for eval. Each part covers a disjoint dataset_idx
range (see scripts/split_codejudgebench_gemini.py), so task_id (and thus
sample_id) is globally unique; latest_by_sample dedupes defensively.

Only tem_0.6_thinking_1024.jsonl is merged: it's the file present in every
part and matching the current configs/codejudgebench_part_*.yaml (temperature
0.6). part_01/tem_0_thinking_1024.jsonl (4 lines) and
part_03/tem_0.3_thinking_1024.jsonl (3000 lines) are leftovers from earlier
temperature settings before the configs were updated to 0.6, and are left
untouched in preds_parallel.
"""
from pathlib import Path
import json

ROOT = Path(__file__).resolve().parents[1]
FNAME = "tem_0.6_thinking_1024.jsonl"
PARTS = sorted(
    (ROOT / "runs_v2/preds_parallel").glob(
        f"part_*/codejudgebench/gemini_2.5_flash/codejudgebench/{FNAME}"
    )
)
OUT = ROOT / f"runs_v2/preds/codejudgebench/gemini_2.5_flash/codejudgebench/{FNAME}"

latest = {}
order = []
for src in PARTS:
    with src.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            sid = d.get("sample_id")
            if not sid:
                continue
            if sid not in latest:
                order.append(sid)
            latest[sid] = d

OUT.parent.mkdir(parents=True, exist_ok=True)
with OUT.open("w", encoding="utf-8") as out:
    for sid in order:
        out.write(json.dumps(latest[sid], ensure_ascii=False) + "\n")
print(f"merged {len(order)} samples from {len(PARTS)} part files -> {OUT}")
