# Running SWE-Exp On KOCO-bench

This adapter keeps the original SWE-bench workflow intact and adds a KOCO-specific
entrypoint:

```bash
python workflow_koco.py infer \
  --framework verl \
  --test-example ARES \
  --instance-ids compute_score \
  --provider aidp \
  --model ali-deepseek-v3.2 \
  --aidp-ak "$AIDP_AK" \
  --max-tokens 8192 \
  --timeout 90
```

The command writes KOCO-compatible output records to:

```text
tmp/koco/outputs/<framework>/swe_exp/<model-label>/algorithm_methods_data_<example>_output.jsonl
```

Each output record preserves the KOCO input schema and adds:

- `completions`: the extracted implementation of the target function
- `status`: `success`, `no_completion`, `invalid_patch_scope`, or `error`
- `model_patch`: the SWE-Exp patch for debugging
- `source_tree_path`: the persisted SWE-Exp trajectory

Run KOCO execution evaluation after inference:

```bash
# On the Docker-capable evaluation machine, copy the generated output into
# KOCO-bench's expected scripts/data layout first:
mkdir -p ../KOCO-bench/KOCO-bench-en/domain_code_generation/scripts/data/verl/swe_exp/ali-deepseek-v3.2_51c3eaf2
cp tmp/koco/outputs/verl/swe_exp/ali-deepseek-v3.2_51c3eaf2/algorithm_methods_data_ARES_output.jsonl \
  ../KOCO-bench/KOCO-bench-en/domain_code_generation/scripts/data/verl/swe_exp/ali-deepseek-v3.2_51c3eaf2/

python workflow_koco.py eval \
  --framework verl \
  --test-example ARES \
  --model ali-deepseek-v3.2
```

Or run both phases:

```bash
python workflow_koco.py run \
  --framework verl \
  --test-example ARES \
  --instance-ids compute_score \
  --provider aidp \
  --model ali-deepseek-v3.2 \
  --aidp-ak "$AIDP_AK" \
  --max-tokens 8192 \
  --timeout 90
```

## Notes

- The KOCO workspace is isolated under `tmp/koco/workspaces/`.
- The target function body is stubbed with `raise NotImplementedError` before
  SWE-Exp sees the workspace.
- The adapter rejects patches that touch files other than the KOCO target file.
- KOCO uses a local source index (`KocoCodeIndex`) compatible with SWE-Exp
  search actions instead of SWE-bench's prebuilt index store.
- Inference writes to `tmp/koco/outputs` by default because some development
  sandboxes expose KOCO's `scripts/data` path as read-only. Copy outputs into
  KOCO's `scripts/data/<framework>/swe_exp/<model-label>/` on the eval machine.
- SWE-Exp dependencies must be installed first, including `litellm`,
  `json_repair`, `faiss-cpu`, `llama-index`, and related packages listed in
  `requirements.txt`.

## Full SWE-Exp KOCO loop with usage logging

First-pass inference across all KOCO frameworks:

```bash
.venv311/bin/python workflow_koco.py infer \
  --framework all \
  --provider aidp \
  --model ali-deepseek-v3.2 \
  --aidp-ak 9Z11UXMn1ClCgk5ngzgLqDS6PzMZ82iq \
  --output-label ali-deepseek-v3.2_51c3eaf2_all_full1_iter100_c5 \
  --output-root tmp/koco/full1 \
  --workspace-dir tmp/koco/full1/workspaces \
  --max-iterations 100 \
  --max-depth 100 \
  --concurrency 5 \
  --max-tokens 8192 \
  --timeout 90 \
  --force-parse \
  --usage-stage infer1 \
  --usage-log tmp/koco/usage/infer1.jsonl
```

Run KOCO Docker evaluation on a Docker-capable evaluation machine after copying
the first-pass outputs into KOCO's expected `scripts/data/<framework>/swe_exp/`
layout. Docker evaluation itself does not call the LLM, so it has no usage log.

Build LLM-summarized SWE-Exp experience from the evaluation results:

```bash
.venv311/bin/python workflow_koco.py build-experience \
  --framework all \
  --provider aidp \
  --model ali-deepseek-v3.2 \
  --aidp-ak 9Z11UXMn1ClCgk5ngzgLqDS6PzMZ82iq \
  --output-label ali-deepseek-v3.2_51c3eaf2_all_full1_iter100_c5 \
  --output-root tmp/koco/full1 \
  --max-tokens 2000 \
  --timeout 90 \
  --usage-stage build-experience \
  --usage-log tmp/koco/usage/build_experience.jsonl
```

Second-pass inference using the generated experience bank:

```bash
.venv311/bin/python workflow_koco.py infer \
  --framework all \
  --provider aidp \
  --model ali-deepseek-v3.2 \
  --aidp-ak 9Z11UXMn1ClCgk5ngzgLqDS6PzMZ82iq \
  --output-label ali-deepseek-v3.2_51c3eaf2_all_full2_iter100_c5 \
  --output-root tmp/koco/full2 \
  --experience-path tmp/koco/full1/experience/koco_experience_bank.json \
  --workspace-dir tmp/koco/full2/workspaces \
  --max-iterations 100 \
  --max-depth 100 \
  --concurrency 5 \
  --max-tokens 8192 \
  --timeout 90 \
  --force-parse \
  --usage-stage infer2 \
  --usage-log tmp/koco/usage/infer2.jsonl
```

Summarize usage for the three LLM stages:

```bash
.venv311/bin/python workflow_koco.py usage-summary \
  tmp/koco/usage/infer1.jsonl \
  tmp/koco/usage/build_experience.jsonl \
  tmp/koco/usage/infer2.jsonl
```
