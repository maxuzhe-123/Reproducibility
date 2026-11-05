# Same Settings, Different Outcomes

Research code for **Same Settings, Different Outcomes: Understanding Reproducibility of LLM-based Code Generation and Evaluation**.

This project studies **system-level output reproducibility**: the similarity of final observable outputs when a task and its controllable execution conditions are held fixed. It distinguishes the underlying LLM's raw responses from extracted code or parsed verdicts, and measures consistency separately from test performance.

## Study overview

| Research question | Factors | Benchmark |
| --- | --- | --- |
| RQ1 | Baseline, few-shot, few-shot CoT, and prompt-injection defenses | HumanEval |
| RQ2 | Enabling thinking, thinking budget, and model capability | HumanEval |
| RQ3 | Raw repository context versus retrieval-augmented context | CoderEval |
| RQ4 | Forward versus reversed candidate order for LLM judges | CodeJudgeBench |

The study uses seven model variants across its experiments: Qwen3-Coder-Next, Claude Sonnet 4.6, Gemini 2.5 Flash-Lite, Gemini 2.5 Flash, Gemini 2.5 Pro, DeepSeek V3.2, and GPT-5.1 Codex. Not every model is used in every RQ. The intended protocol is ten repetitions per task or judge instance under each condition.

The paper reports 164 HumanEval tasks, 223 analyzed CoderEval tasks, and 694 CodeJudgeBench instances. These are **analysis counts**, not guarantees about the contents of every supplied template: CoderEval configurations request 230 tasks, and the judge configuration loads all records in its local file and expands each into the requested candidate orderings. Exact paper reproduction requires the original task manifest, filters, exclusions, and run configurations.

## What is included

- Generation and evaluation pipelines for the three benchmarks.
- Editable experiment configuration templates and prompt defenses.
- Self-CodeBLEU, candidate similarity metrics, and judge evaluation code.
- Paired statistical analysis and plotting utilities.
- HumanEval task data, a BPE vocabulary, and a prepared CoderEval retrieval-prompt file.

This checkout does **not** include the historical `runs_v2/` results, CoderEval repository environments, or the CodeJudgeBench data file. There is no verified paper or results-archive URL in this release. Historical results cannot be reproduced from this checkout alone until those artifacts and the exact analysis manifests are supplied.

## Repository structure

```text
configs/                  Editable condition templates (see the configuration map below)
src/
  pipeline_v2/            Generation, evaluation, comparison, and plotting
  models/                 Model clients and reasoning/thinking configuration
  datasets/               Benchmark loaders and included HumanEval.jsonl
  defenses/               Prompt transformations and BPE utilities
  prompts/                Code-generation prompt templates
  utils/                  Metrics, extraction, execution, and RepoCoder utilities
scripts/                  Statistical tests, table rendering, splitting/merging
prompts/                  Prepared CoderEval RG1 retrieval prompts
data/                     BPE vocabulary for retokenization
evaluate_reproducibility.py  Additional task-level similarity/test aggregation
requirements.txt          Main Python dependencies
requirements-codereval-eval.txt  CoderEval environment dependency list
.env.example              Environment-variable template
LICENSE                   MIT license for this repository's code
```

Use `src.pipeline_v2` for the two-stage workflow below; it is the only experiment entry point in this repository.

## Environment setup

Use Python 3.11 and run commands from the repository root.

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install --no-deps codebleu==0.7.0
cp .env.example .env
```

Edit `.env` to set your own `OPENROUTER_API_KEY`, then export it:

```bash
set -a
source .env
set +a
```

The pipeline reads environment variables; copying `.env.example` does not load them automatically. Generation uses OpenRouter and may incur API costs. Never commit credentials.

`requirements.txt` records pinned dependencies, including the tree-sitter parsers. CodeBLEU 0.7.0 is installed with `--no-deps` because its declared `tree-sitter<0.23` pin conflicts with the tree-sitter 0.25.2 used in the experiments. Installation and parser compatibility must be checked on the target platform; this README does not claim a freshly validated installation. CoderEval additionally requires its repository-specific environments and dependencies.

### Model and run settings

Before starting a run, inspect:

- `model.model_id`, provider, temperature, `top_p`, and `top_k`.
- `model.thinking` or `model.reasoning`, including enabled state, budget/effort, and exclusion of reasoning text.
- `model.max_new_tokens`, stop strings, task limits, sample count, and execution timeout.
- Dataset paths, prompt template, defense parameters, retrieval settings, and judge order.

The supplied YAML files are **templates, not a complete frozen run matrix**. Changing only the model identifier is insufficient to guarantee a matched comparison: supported reasoning controls and parameter constraints differ between model families. In particular, the provided Gemini Flash and Pro templates currently select different prompt templates; align them before a fixed-prompt capability comparison. Output limits also differ across some prompting and retrieval templates.

For historical reproduction, use archived per-run settings rather than assuming today's template equals the settings used in the paper. Record provider model identifiers, execution dates, configuration files, dependency versions, and sample IDs. A model's marketed name does not ensure an unchanged backend.

## Benchmark preparation

### HumanEval

The included file is `src/datasets/HumanEval.jsonl`. Keep its task IDs, signatures, and tests intact. Consult the [HumanEval project](https://github.com/openai/human-eval) for benchmark provenance and licensing.

### CodeJudgeBench

Obtain the code-generation data from [mattymchen/codejudgebench](https://huggingface.co/datasets/mattymchen/codejudgebench) and prepare:

```text
src/datasets/CodeJudgeBench/codegen.jsonl
```

The loader expects fields including `question_content`, `pos_response`, `neg_response`, `question_id`, `response_model`, and `difficulty`. Preserve problem/responder identifiers. In the current pipeline, `position_mode: both` creates forward and reversed instances for each input record, with ten judgments per ordering when `num_samples: 10`.

Downloading the full dataset does not automatically recreate the paper's 694-instance analysis set. Match the original selection and specify whether counts refer to input pairs or order-expanded instances.

### CoderEval and retrieval

Obtain the Python benchmark, corresponding repositories, and validation harness from [CoderEval](https://github.com/CoderEval/CoderEval). The templates/loaders refer to:

```text
src/datasets/CoderEval/CoderEval4Python.json
src/datasets/CoderEval/CEPythonRaw.jsonl
src/datasets/CoderEval/CEPythonHumanLabel.jsonl
src/datasets/CoderEval-ALL.jsonl
repos/
prompts/rg-one-gram-ws-20-ss-2.jsonl
```

The exact files needed depend on `docstring_source` and `rag.mode`. Prepared RG1 prompts are included, but the remaining benchmark files and repositories are not. Ensure task IDs agree across task records, repository indices, and retrieved prompts.

`src/utils/repocoder/` contains retrieval preparation utilities. The RAG loader consumes prepared prompts; selecting `rag.mode: rg1` is not itself an instruction to rebuild the retrieval index. When using the supplied prompt file, record it as part of the experimental input.

Before using `tools_codereval_eval_setup.sh`, replace its `xxxx/xxxx` Conda and pip paths with your actual environment paths. Inspect `requirements-codereval-eval.txt` and repository dependencies, and configure:

```bash
export CODEREVAL_PYTHON=/absolute/path/to/codereval_eval/bin/python
```

The repository layouts must include the validation files expected by the CoderEval harness, not just arbitrary current upstream checkouts. Run generated code only in an isolated evaluation environment without credentials or unrelated writable files.

## Running experiments

### Quick start: generation followed by evaluation

After configuring the API key and reviewing the selected YAML:

```bash
python -m src.pipeline_v2.preds --config configs/humaneval1.yaml
python -m src.pipeline_v2.evaluator --config configs/humaneval1.yaml
```

Generation resumes existing successful samples by default and retries failed requests unless disabled. Re-running generation can compact or replace records, and evaluation rebuilds result files in its output directory. Preserve historical artifacts and use separate run/output locations for new experiments.

### Configuration map

| Experiment | Configuration |
| --- | --- |
| HumanEval baseline | `configs/humaneval1.yaml` |
| Few-shot: three demonstrations | `configs/humaneval_three_shot.yaml` |
| Few-shot CoT | `configs/humaneval_cot.yaml` |
| Delimiters | `configs/humaneval_defense_delimiters.yaml` |
| Instructional defense | `configs/humaneval_defense_instructional.yaml` |
| Paraphrasing | `configs/humaneval_defense_paraphrasing.yaml` |
| Sandwich defense | `configs/humaneval_defense_template.yaml` |
| Retokenization | `configs/humaneval_defense_retokenization.yaml` |
| Thinking-budget experiment | `configs/humaneval_gemini_flash.yaml` |
| Model-capability experiment | `configs/humaneval_gemini_pro.yaml` |
| CoderEval raw context | `configs/codereval_raw.yaml` |
| CoderEval RAG | `configs/codereval_with_rag.yaml` |
| Judge order reversal | `configs/codejudgebench.yaml` |

Run each chosen configuration with both commands above, substituting its filename.

For RQ2, create separate configurations for Flash budgets 0, 1024, 2048, 4096, and 8192. For the capability comparison, use Flash-Lite, Flash, and Pro with the same prompt and 1024-token budget. Retain all other applicable controls. Reasoning output exclusion does not mean reasoning is disabled.

For RQ3, prepare the CoderEval environment before running either configuration. Record and reconcile different output limits and timeouts when interpreting comparisons.

For RQ4, the judge prompt is defined in `src/pipeline_v2/codejudge_preds.py`. Keep `codejudge.position_mode: both` to evaluate both candidate orders. Judge evaluation parses verdicts rather than executing candidate code.

### Output layout

Default outputs follow:

```text
runs_v2/preds/<dataset>/<model>/<condition>/<sampling-token>.jsonl
runs_v2/evaluations/<dataset>/<model>/<condition>/<sampling-token>/
```

Defense conditions may introduce extra directory levels. Generation JSONL stores raw responses, task/sample IDs, status, and metadata. Evaluation outputs include `evaluations.jsonl`, `task_results.jsonl`, and `summary.json`; generation evaluators also produce representation metrics and cleaned-code files where supported.

## Metrics and aggregation

### Code-generation reproducibility

Self-CodeBLEU compares generated code **with other outputs from the same task and condition**, not with a reference solution. Ten available samples give 45 unordered pairs. Each pair is symmetrized, its four components receive equal weights, and task scores are macro-averaged across tasks.

Report the overall score together with n-gram, weighted n-gram, syntax, and data-flow scores. These measure representation similarity, not functional correctness or complete semantic equivalence.

Files prefixed `codebleu_no_docstring` use code with **docstrings and comments removed**, with a fallback for preprocessing failures. Do not combine those scores with unstripped `task_results.jsonl` scores without identifying the difference.

### Test performance versus pass@10

- **Test pass rate:** passing evaluated generations divided by evaluated generations for each task, followed by a task-level macro-average. Inspect `task_results.jsonl` → `preds[].pass`.
- **Stored pass@10:** for a complete ten-sample task, whether at least one sample passes, averaged across tasks. This is not the fraction of the ten samples passing.
- **Stored pass@1:** some pipeline summaries use the first evaluated sample's pass flag. Inspect the implementation before treating this as an average over all generated samples.

When tasks have missing samples, report the actual counts and denominator. The stored at-least-one flag on fewer than ten outputs must not be presented as a fully observed ten-sample result. Generation/API failures, extraction failures, runtime errors, and failed assertions should be distinguished. Passing finite tests does not prove correctness for every input.

### Judge reproducibility

For each instance and candidate ordering, with `N` evaluated judgments:

- `Repro = max(count_A, count_B) / N`: concentration on the majority A/B verdict.
- `ReproCorrect = count_correct / N`: proportion of correct judgments.

Invalid or empty verdicts remain in the denominator and contribute to neither A/B count nor correctness. This is a majority-concentration measure, not pairwise CodeBLEU or pairwise verdict agreement. Inspect `task_results.jsonl` and the evaluator's summary for order-specific and difficulty-specific aggregation.

### Candidate metrics

`src/utils/text_metrics.py` implements standalone weighted-unigram BLEU. It is distinct from CodeBLEU's weighted n-gram component. `src/utils/struct_entropy_metrics.py` implements structural-feature-distribution JSD similarity and SCE. Keep preprocessing and implementation versions aligned when comparing them.

## Reproducing analysis from saved outputs

There are two different workflows:

1. **Reanalyze saved outputs:** recompute scores, statistical comparisons, and plots from archived generations or task results. This avoids additional model calls.
2. **Generate new outputs:** call current model endpoints using the configurations above. Sampling and backend changes mean new outputs need not reproduce historical values exactly.

Historical result files are not included in this checkout. The following commands are examples for use after those files have been obtained.

### Recompute cleaned CodeBLEU

The input must contain one task per record with `preds[].code`; use aggregated `task_results.jsonl`, not the per-sample generation JSONL:

```bash
python -m src.pipeline_v2.codebleu_no_docstring_from_preds \
  --preds /path/to/archived/task_results.jsonl \
  --output-dir runs_v2/reanalysis/no_docstring
```

This recalculates similarity without re-running functional tests.

### Compare conditions

Replace the example paths with matching per-task result files:

```bash
python scripts/wilcoxon_compare.py \
  --input baseline=/path/to/baseline/codebleu_no_docstring.jsonl \
  --input few_shot=/path/to/few_shot/codebleu_no_docstring.jsonl \
  --input cot=/path/to/cot/codebleu_no_docstring.jsonl \
  --pair few_shot:baseline \
  --pair cot:few_shot \
  --out runs_v2/reanalysis/prompt_comparisons
```

The comparison is `NEW:BASE`, so positive differences indicate an increase in the first-named condition. Tests pair task IDs; inspect unmatched IDs, effective sample counts, ties, and the multiple-comparison policy before interpreting significance.

### Figures and tables

```bash
python -m src.pipeline_v2.compare_reproducibility_groups \
  --baseline /path/to/baseline/codebleu_no_docstring.jsonl \
  --few-shot /path/to/few_shot/codebleu_no_docstring.jsonl \
  --cot /path/to/cot/codebleu_no_docstring.jsonl \
  --out runs_v2/reanalysis/prompt_figures
```

This produces component-change categories, not every figure in the paper. Other comparison/plotting modules are under `src/pipeline_v2/`; table rendering is in `scripts/render_wilcoxon_table.py`. Several scripts retain local paths or `xxxx/xxxx` placeholders. Inspect their arguments and update paths before running. There is currently no verified one-command reproduction of all paper tables and figures.

## Known limitations and release checklist

- Release the exact historical generations, test records, analysis manifests, and per-run configurations before claiming complete paper reproducibility.
- Reconcile the paper's analyzed benchmark counts with the template task limits and order expansion.
- Keep raw versus cleaned CodeBLEU, test pass rate versus pass@10, and model-level versus system-level outputs distinct.
- Historical comparisons may contain decoding/reasoning differences or incomplete samples; verify these from run records rather than assuming all settings match.
- Extraction can select an explanatory code block instead of the final implementation. Similarity and testing then describe the extracted artifact; preserve raw responses for diagnosis.
- Check current model availability, API support, parser compatibility, repository dependencies, and execution timeouts.
- Preserve third-party notices and verify redistribution permission for datasets, repository code, and prepared retrieval prompts.

## Third-party resources and licensing

This repository's code is licensed under the [MIT License](LICENSE). That license does not automatically relicense benchmark data, retrieved repository code, or third-party implementations.

The retokenization utilities derive from [Open Prompt Injection](https://github.com/liu00222/Open-Prompt-Injection). Judge data/protocols and the CoderEval execution workflow build on their respective benchmark projects. Follow the upstream licenses and retain attribution when redistributing derived artifacts.

## Citation and contact

The paper and archive links will be added when available. The provisional citation below does not assert acceptance or publication:

```bibtex
@misc{ma_samesettings,
  title  = {Same Settings, Different Outcomes: Understanding Reproducibility of LLM-based Code Generation and Evaluation},
  author = {Ma, Xuzhe and Feng, Ruitao and Shen, Haifeng},
  note   = {Research manuscript}
}
```

For questions, use the repository's issue tracker once the public repository is available, and include the relevant configuration, task IDs, dependency versions, and sanitized error output. Do not include API keys.
