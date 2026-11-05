"""Merge all part prediction files (+ the original) into the canonical
preds path for eval. latest_by_sample dedupes; task_id is globally unique."""
from pathlib import Path
import json

ROOT = Path(__file__).resolve().parents[1]
ORIG = ROOT / "runs_v2/preds/codejudgebench/qwen3_coder_next/codejudgebench/tem_0_top_p_0.95_top_k_20.jsonl"
PARTS = sorted((ROOT / "runs_v2/preds_parallel").glob("part_*/codejudgebench/qwen3_coder_next/codejudgebench/tem_0_top_p_0.95_top_k_20.jsonl"))

latest = {}
order = []
for src in [ORIG, *PARTS]:
    if not src.exists():
        continue
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

with ORIG.open("w", encoding="utf-8") as out:
    for sid in order:
        out.write(json.dumps(latest[sid], ensure_ascii=False) + "\n")
print(f"merged {len(order)} samples from {1 + len(PARTS)} files -> {ORIG}")
