# KOCO native SWE-Exp infer1 — partial eval handoff

This is a **partial** handoff from the native-flow full infer1 run
(`--framework all`, label `ali-deepseek-v3.2_native_full1_iter100_c5`).
Infer is still running on the dev box; only the frameworks that were finished
at snapshot time are shipped here so eval can start in parallel.

## Included (frozen)

| framework   | instances | notes |
|-------------|-----------|-------|
| raganything | 8 / 8     | complete |
| open-r1     | 23 / 25   | see "known gaps" |

## Not included yet

- `smolagents` — in-flight at snapshot time
- `tensorrt_model_optimizer` — not started
- `verl` — not started

These will ship in a later handoff once infer1 finishes.

## Known gaps (open-r1, 2 instances)

These two crashed early during infer and were not re-attempted in this run, so
they are absent from the JSONL (not present as records at all):

- `open-r1 / AlphaDrive / Qwen2VLGRPOTrainer.compute_loss`
- `open-r1 / CANOE / format_reward`

Eval will therefore score open-r1 over the 23 present instances.

## How to run on the eval machine (docker required)

```bash
cd <repo>/SWE-Exp/tmp/koco/eval_handoff_native_full1
PYTHON_BIN=python bash run_eval_native_full1.sh
```

The script copies these outputs into
`KOCO-bench/.../scripts/data/<framework>/swe_exp/<LABEL>/`, runs the KOCO docker
eval per framework, and prints the resulting `*_result.metrics.json` paths.

After eval, add the produced `*_result.jsonl` and `*_result.metrics.json` under
`scripts/data/.../swe_exp/<LABEL>/` (use `git add -f`, that tree is gitignored),
commit, and push so the dev box can pull them back.
