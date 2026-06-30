# KOCO SWE-Exp Full1 Eval Handoff

This directory contains first-pass SWE-Exp KOCO outputs for:

```text
ali-deepseek-v3.2_51c3eaf2_all_full1_iter100_c5
```

Run on a Docker-capable evaluation machine from the `SWE-Exp` directory:

```bash
bash tmp/koco/eval_handoff_full1/run_eval_full1.sh
```

The script copies `outputs/<framework>/swe_exp/<label>/*_output.jsonl` into
KOCO's expected `scripts/data/<framework>/swe_exp/<label>/` layout, then runs:

```bash
.venv311/bin/python workflow_koco.py eval --framework <framework> --output-label <label>
```

After eval, copy back the generated files:

```text
../KOCO-bench/KOCO-bench-en/domain_code_generation/scripts/data/<framework>/swe_exp/<label>/*_result.jsonl
../KOCO-bench/KOCO-bench-en/domain_code_generation/scripts/data/<framework>/swe_exp/<label>/*_result.metrics.json
```
