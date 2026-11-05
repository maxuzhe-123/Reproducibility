import json, re
from pathlib import Path
from collections import Counter

prompt_path = Path("prompts/rg-one-gram-ws-20-ss-2.jsonl")
assert prompt_path.exists(), f"missing {prompt_path}"

cnt = Counter()
examples = []

def has_gt_field(d):
    md = d.get("metadata", {})
    return "ground_truth" in md and md["ground_truth"].strip() != ""

def contains_env(prompt: str):
    p = prompt.lower()
    return ("site-packages" in p) or ("/venv/" in p) or ("\\venv\\" in p) or ("/.venv/" in p)

def contains_tests(prompt: str):
    return "\n# tests/" in prompt or "\n# test/" in prompt

def target_impl_exposed(d):
    # Check whether "def <function_name>" appears in top_k_context
    fn = (d.get("metadata") or {}).get("function_name")
    if not fn:
        return False
    blocks = (d.get("top_k_context") or [])
    text = "\n".join(b.get("context","") for b in blocks)
    return re.search(rf"\bdef\s+{re.escape(fn)}\s*\(", text) is not None

with prompt_path.open("r", encoding="utf-8") as f:
    for i, line in enumerate(f, 1):
        d = json.loads(line)
        prompt = d.get("prompt","")
        if has_gt_field(d):
            cnt["has_metadata_ground_truth"] += 1
            if len(examples) < 3:
                examples.append(("ground_truth_field", i, d.get("metadata",{}).get("task_id")))
        if contains_env(prompt):
            cnt["contains_env_paths"] += 1
            if len(examples) < 6:
                examples.append(("env_path", i, d.get("metadata",{}).get("task_id")))
        if contains_tests(prompt):
            cnt["contains_tests"] += 1
            if len(examples) < 9:
                examples.append(("tests", i, d.get("metadata",{}).get("task_id")))
        if target_impl_exposed(d):
            cnt["target_impl_in_topk"] += 1

        cnt["total"] += 1

print("=== Leakage Audit ===")
for k in ["total","has_metadata_ground_truth","contains_env_paths","contains_tests","target_impl_in_topk"]:
    print(f"{k}: {cnt[k]} ({cnt[k]/max(cnt['total'],1):.1%})")
print("\nExamples:")
for t, line_no, tid in examples:
    print(f"- {t} at line {line_no} task={tid}")