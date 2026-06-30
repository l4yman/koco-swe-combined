from dotenv import load_dotenv

load_dotenv(".env")

import argparse
import json
import logging
import os
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

from moatless.benchmark.koco import (
    KocoCodeIndex,
    KocoExperienceSelector,
    append_koco_experience,
    build_koco_experience_from_eval,
    create_koco_repository,
    extract_completion_from_file_context,
    get_koco_frameworks,
    iter_koco_instances,
    model_label_from_model,
    run_koco_eval,
    safe_name,
    upsert_output_record,
)
from moatless.completion.aidp import AIDPCompletionModel
from moatless.completion.completion import CompletionModel, LLMResponseFormat
from moatless.file_context import ContextSpan


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def build_completion_model(
    model: str,
    temperature: float,
    max_tokens: int,
    base_url: str | None,
    api_key: str | None,
    provider: str,
    aidp_ak: str | None,
    usage_stage: str | None = None,
    usage_log: str | None = None,
) -> CompletionModel:
    if provider == "aidp":
        completion_model = AIDPCompletionModel(
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            model_base_url=base_url,
            model_api_key=api_key,
            aidp_ak=aidp_ak,
            usage_stage=usage_stage or "",
            usage_log=usage_log,
        )
    else:
        completion_model = CompletionModel(
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            model_base_url=base_url,
            model_api_key=api_key,
        )
        if usage_log:
            logger.warning("Usage logging is only implemented for provider=aidp; ignoring --usage-log")
    completion_model.response_format = LLMResponseFormat.REACT
    return completion_model


def create_code_index(instance: dict, repository):
    logger.info("Building local KOCO code index for %s", instance["instance_id"])
    return KocoCodeIndex(repository=repository)


def build_actions(completion_model, repository, code_index):
    from moatless.actions.find_class import FindClass
    from moatless.actions.find_code_snippet import FindCodeSnippet
    from moatless.actions.find_function import FindFunction
    from moatless.actions.finish import Finish
    from moatless.actions.semantic_search import SemanticSearch
    from moatless.actions.string_replace import StringReplace
    from moatless.actions.view_code import ViewCode

    actions = []
    if code_index is not None:
        actions.extend(
            [
                FindClass(completion_model=completion_model, code_index=code_index, repository=repository),
                FindFunction(completion_model=completion_model, code_index=code_index, repository=repository),
                FindCodeSnippet(completion_model=completion_model, code_index=code_index, repository=repository),
                SemanticSearch(completion_model=completion_model, code_index=code_index, repository=repository),
            ]
        )

    actions.extend(
        [
            ViewCode(completion_model=completion_model, repository=repository),
            StringReplace(repository=repository, code_index=code_index),
            Finish(enforce_patch=True),
        ]
    )
    return actions


def validate_patch_scope(instance: dict, patch: str) -> tuple[bool, str]:
    if not patch.strip():
        return False, "empty patch"

    target_path = instance["target_path"]
    touched_files = []
    for line in patch.splitlines():
        if line.startswith("diff --git "):
            parts = line.split()
            if len(parts) >= 4:
                path = parts[2]
                if path.startswith("a/"):
                    path = path[2:]
                touched_files.append(path)

    bad_files = sorted({path for path in touched_files if path != target_path})
    if bad_files:
        return False, f"patch touched non-target files: {', '.join(bad_files)}"
    for line in patch.splitlines():
        if line.startswith("@@ "):
            match = re.search(r"-(\d+)(?:,(\d+))?", line)
            if not match:
                continue
            old_start = int(match.group(1))
            old_len = int(match.group(2) or "1")
            old_end = old_start + max(old_len - 1, 0)
            target_start = max(1, int(instance["target_start_line"]) - 5)
            target_end = int(instance["target_end_line"]) + 20
            if old_start < target_start or old_end > target_end:
                return False, (
                    f"patch hunk {old_start}-{old_end} is outside target function "
                    f"window {target_start}-{target_end}"
                )
    return True, ""


def _node_rank(node) -> tuple:
    reward = 0.0
    try:
        if node.reward and node.reward.value is not None:
            reward = float(node.reward.value)
    except (TypeError, ValueError):
        reward = 0.0
    observation = getattr(node, "observation", None)
    expect_correction = bool(getattr(observation, "expect_correction", False))
    has_patch = False
    try:
        has_patch = bool(_get_node_patch(node))
    except Exception:
        has_patch = False
    return (
        bool(node.is_finished()),
        has_patch,
        not expect_correction,
        bool(node.reward),
        reward,
        node.get_depth(),
        node.node_id,
    )


def _get_node_patch(node) -> str:
    if not getattr(node, "file_context", None):
        return ""
    patch = ""
    try:
        patch = node.file_context.generate_git_patch()
    except Exception:
        patch = ""
    if patch:
        return patch

    # Persisted trajectories may contain only per-file patches without the
    # repository content needed to regenerate a diff.
    files = getattr(node.file_context, "_files", {}) or {}
    return "\n".join(
        file.patch for file in files.values() if getattr(file, "patch", None)
    )


def extract_valid_koco_node(search_tree, instance: dict, preferred_node=None) -> tuple[object | None, str, str, str]:
    candidates = []
    if preferred_node is not None:
        candidates.append(preferred_node)
    candidates.extend(search_tree.get_finished_nodes())
    candidates.extend(search_tree.get_leaf_nodes())
    candidates.extend(search_tree.root.get_all_nodes())

    unique = {}
    for node in candidates:
        if node is not None:
            unique[node.node_id] = node

    last_error = ""
    for node in sorted(unique.values(), key=_node_rank, reverse=True):
        if not getattr(node, "file_context", None):
            continue
        patch = _get_node_patch(node)
        ok, scope_error = validate_patch_scope(instance, patch)
        if not ok:
            last_error = scope_error
            continue
        try:
            completion = extract_completion_from_file_context(node.file_context, instance)
        except Exception as exc:
            last_error = f"failed to extract completion from node {node.node_id}: {exc}"
            continue
        if completion:
            return node, patch, completion, ""
        last_error = f"node {node.node_id} has valid patch but no extractable completion"

    return None, "", "", last_error or "no valid patched node found"


def run_single_koco_instance(args, instance: dict) -> dict:
    from moatless.agent.agent import ActionAgent
    from moatless.discriminator import AgentDiscriminator
    from moatless.experience.instructor import Instructor
    from moatless.experience.prompts.agent_prompts import (
        ASSISTANT_GUIDELINES,
        ASSISTANT_ROLE,
        INSTRUCTION_GUIDELINES,
        INSTRUCTOR_FORMAT,
        INSTRUCTOR_ROLE,
    )
    from moatless.feedback.feedback_agent import FeedbackAgent
    from moatless.file_context import FileContext
    from moatless.search_tree import SearchTree
    from moatless.value_function.base import ValueFunction

    logger.info("Running KOCO instance %s", instance["instance_id"])

    completion_model = build_completion_model(
        model=args.model,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        base_url=args.base_url,
        api_key=args.api_key,
        provider=args.provider,
        aidp_ak=args.aidp_ak,
        usage_stage=args.usage_stage,
        usage_log=args.usage_log,
    )
    completion_model.timeout = args.timeout
    discriminator_model = build_completion_model(
        model=args.model,
        temperature=args.discriminator_temperature,
        max_tokens=args.max_tokens,
        base_url=args.base_url,
        api_key=args.api_key,
        provider=args.provider,
        aidp_ak=args.aidp_ak,
        usage_stage=args.usage_stage,
        usage_log=args.usage_log,
    )
    discriminator_model.timeout = args.timeout
    value_model = build_completion_model(
        model=args.model,
        temperature=args.value_temperature,
        max_tokens=args.max_tokens,
        base_url=args.base_url,
        api_key=args.api_key,
        provider=args.provider,
        aidp_ak=args.aidp_ak,
        usage_stage=args.usage_stage,
        usage_log=args.usage_log,
    )
    value_model.timeout = args.timeout

    repository = create_koco_repository(
        instance,
        workspace_base_dir=args.workspace_dir,
        reset=not args.reuse_workspace,
    )

    code_index = create_code_index(instance=instance, repository=repository)

    file_context = FileContext(repo=repository)
    file_context.add_file(instance["target_path"], show_all_spans=False, add_extra=False)
    target_context = file_context.get_context_file(instance["target_path"])
    if target_context:
        target_context.was_viewed = True
        target_context.spans.append(
            ContextSpan(
                span_id=f"target_{instance['target_start_line']}_{instance['target_end_line']}",
                start_line=max(1, instance["target_start_line"] - 5),
                end_line=instance["target_end_line"] + 20,
            )
        )

    actions = build_actions(completion_model, repository, code_index)
    agent = ActionAgent(
        system_prompt=ASSISTANT_ROLE + ASSISTANT_GUIDELINES,
        actions=actions,
        completion=completion_model,
        use_few_shots=False,
    )

    discriminator = AgentDiscriminator(
        completion=discriminator_model,
        n_agents=args.discriminator_agents,
        n_rounds=args.discriminator_rounds,
    )
    value_function = ValueFunction(completion_model=value_model)

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    instance_dir = Path(args.output_root) / "trajectory" / safe_name(instance["instance_id"], 140)
    instance_dir.mkdir(parents=True, exist_ok=True)
    persist_path = instance_dir / f"{run_id}_trajectory.json"

    max_depth = min(args.max_iterations, args.max_depth)
    feedback_generator = FeedbackAgent(
        completion_model=completion_model,
        instance_dir=str(instance_dir),
        include_tree=True,
        include_node_suggestion=True,
    )
    experience_path = Path(args.experience_path) if args.experience_path else Path(args.output_root) / "experience" / "koco_experience_bank.json"
    experience_state_path = Path(args.output_root) / "experience" / safe_name(instance["instance_id"], 140) / f"{run_id}_experience.json"
    select_agent = KocoExperienceSelector(
        completion=completion_model,
        instance=instance,
        experience_path=experience_path,
        persist_dir=experience_state_path,
        top_k=args.experience_top_k,
    )
    old_experiences = select_agent.select_workflow(n=args.experience_top_k)
    select_agent.persist(
        {
            "old_experiences": old_experiences,
            "HET": {},
            "trajectory": str(persist_path),
        }
    )
    instructor = Instructor(
        completion=completion_model,
        system_prompt=INSTRUCTOR_ROLE + INSTRUCTION_GUIDELINES,
        output_format=INSTRUCTOR_FORMAT,
        task=instance["problem_statement"],
        taken_actions=max_depth,
    )

    search_tree = SearchTree.create(
        message=instance["problem_statement"],
        assistant=agent,
        instructor=instructor,
        file_context=file_context,
        value_function=value_function,
        discriminator=discriminator,
        feedback_generator=feedback_generator,
        metadata={"koco_stop_on_patch": True},
        max_finished_nodes=args.max_finished_nodes,
        max_iterations=args.max_iterations,
        max_expansions=args.max_expansions,
        max_depth=max_depth,
        persist_path=str(persist_path),
    )

    status = "no_finished_node"
    patch = ""
    completion = ""
    error = ""

    try:
        finished_node = search_tree.run_search(select_agent, old_experiences)
        search_tree.persist(str(persist_path))
        valid_node, patch, completion, selection_error = extract_valid_koco_node(
            search_tree, instance, finished_node
        )
        if valid_node:
            status = "success"
            logger.info(
                "Selected KOCO node %s for %s",
                valid_node.node_id,
                instance["instance_id"],
            )
        else:
            status = "no_completion"
            error = selection_error
            logger.warning("%s: %s", instance["instance_id"], selection_error)
    except Exception as exc:
        logger.exception("KOCO instance failed: %s", instance["instance_id"])
        try:
            search_tree.persist(str(persist_path))
        except Exception:
            logger.exception("Failed to persist crashed KOCO search tree: %s", instance["instance_id"])
        valid_node, patch, completion, selection_error = extract_valid_koco_node(search_tree, instance)
        if valid_node:
            status = "success"
            error = f"search raised after valid patch: {exc}"
            logger.info(
                "Recovered valid KOCO node %s for %s after exception: %s",
                valid_node.node_id,
                instance["instance_id"],
                exc,
            )
        else:
            status = "error"
            error = f"{exc}; {selection_error}"

    model_label = args.output_label or model_label_from_model(args.model)
    output_file = upsert_output_record(
        instance=instance,
        model_label=model_label,
        completion=completion,
        status=status,
        model_patch=patch,
        source_tree_path=str(persist_path),
    )

    result = {
        "instance_id": instance["instance_id"],
        "framework": instance["framework"],
        "example": instance["example"],
        "function_name": instance["function_name"],
        "status": status,
        "output_file": str(output_file),
        "trajectory": str(persist_path),
        "error": error,
    }
    append_koco_experience(instance, result, experience_path)
    logger.info("KOCO result: %s", json.dumps(result, ensure_ascii=False))
    return result


def cmd_infer(args) -> int:
    function_names = set(args.instance_ids or []) if args.instance_ids else None
    frameworks = get_koco_frameworks() if args.framework == "all" else [args.framework]
    instances = []
    for framework in frameworks:
        instances.extend(
            list(
                iter_koco_instances(
                    framework=framework,
                    test_example=args.test_example,
                    function_names=function_names,
                    force_parse=args.force_parse,
                )
            )
        )
    if args.limit:
        instances = instances[: args.limit]

    if not instances:
        logger.error("No KOCO instances selected.")
        return 1

    logger.info("Selected %d KOCO instance(s)", len(instances))
    failures = 0
    workers = max(1, args.concurrency)
    if workers == 1:
        for instance in instances:
            result = run_single_koco_instance(args, instance)
            if result["status"] != "success":
                failures += 1
    else:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(run_single_koco_instance, args, instance): instance for instance in instances}
            for future in as_completed(futures):
                instance = futures[future]
                try:
                    result = future.result()
                except Exception as exc:
                    failures += 1
                    logger.exception("KOCO instance crashed: %s", instance["instance_id"])
                    continue
                if result["status"] != "success":
                    failures += 1

    return 1 if failures else 0


def cmd_eval(args) -> int:
    frameworks = get_koco_frameworks() if args.framework == "all" else [args.framework]
    failures = 0
    for framework in frameworks:
        failures += run_koco_eval(
            framework=framework,
            model_label=getattr(args, "output_label", None) or model_label_from_model(args.model),
            test_example=args.test_example,
        )
    return 1 if failures else 0


def cmd_build_experience(args) -> int:
    model_label = args.output_label or model_label_from_model(args.model)
    frameworks = get_koco_frameworks() if args.framework == "all" else [args.framework]
    experience_path = Path(args.output_root) / "experience" / "koco_experience_bank.json"

    completion_model = None
    if not args.no_llm_experience:
        completion_model = build_completion_model(
            model=args.model,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            base_url=args.base_url,
            api_key=args.api_key,
            provider=args.provider,
            aidp_ak=args.aidp_ak,
            usage_stage=args.usage_stage,
            usage_log=args.usage_log,
        )
        completion_model.timeout = args.timeout

    for framework in frameworks:
        examples = [args.test_example] if args.test_example else None
        if not examples:
            from moatless.benchmark.koco import get_koco_examples

            examples = get_koco_examples(framework)
        for example in examples:
            built_path = build_koco_experience_from_eval(
                framework=framework,
                example=example,
                model_label=model_label,
                experience_path=experience_path,
                completion=completion_model,
            )
            logger.info("Updated KOCO experience bank: %s", built_path)
    return 0


def cmd_usage_summary(args) -> int:
    summary: dict[str, dict[str, float | int]] = {}
    by_model: dict[tuple[str, str], dict[str, float | int]] = {}

    def empty() -> dict[str, float | int]:
        return {
            "calls": 0,
            "errors": 0,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "chars": 0,
            "elapsed_seconds": 0.0,
        }

    for log_path in args.usage_logs:
        path = Path(log_path)
        if not path.exists():
            logger.warning("Usage log not found: %s", path)
            continue
        with path.open("r", encoding="utf-8") as fh:
            for line_no, line in enumerate(fh, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    logger.warning("Skipping invalid usage JSONL record: %s:%d", path, line_no)
                    continue
                stage = str(record.get("stage") or "unknown")
                model = str(record.get("model") or "unknown")
                stage_summary = summary.setdefault(stage, empty())
                model_summary = by_model.setdefault((stage, model), empty())
                for bucket in (stage_summary, model_summary):
                    bucket["calls"] += 1
                    bucket["errors"] += 1 if record.get("error") else 0
                    bucket["prompt_tokens"] += int(record.get("prompt_tokens") or 0)
                    bucket["completion_tokens"] += int(record.get("completion_tokens") or 0)
                    bucket["total_tokens"] += int(record.get("total_tokens") or 0)
                    bucket["chars"] += int(record.get("chars") or 0)
                    bucket["elapsed_seconds"] += float(record.get("elapsed_seconds") or 0.0)

    result = {
        "by_stage": summary,
        "by_stage_model": {
            f"{stage}:{model}": values
            for (stage, model), values in sorted(by_model.items())
        },
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


def cmd_run(args) -> int:
    infer_rc = cmd_infer(args)
    if infer_rc != 0 and not args.eval_on_infer_failure:
        return infer_rc
    eval_rc = cmd_eval(args)
    return 1 if infer_rc or eval_rc else 0


def add_common_args(parser):
    parser.add_argument("--framework", required=True, help="KOCO framework name, e.g. verl, or all")
    parser.add_argument("--model", default="ali-deepseek-v3.2", help="Model name")
    parser.add_argument("--test-example", default=None, help="Single KOCO test example")
    parser.add_argument("--instance-ids", nargs="+", help="Function names to run")
    parser.add_argument("--limit", type=int, default=0, help="Limit number of selected instances")
    parser.add_argument("--output-label", default="", help="Output label under swe_exp/ (default: derived from model)")


def add_infer_args(parser):
    parser.add_argument("--provider", choices=["aidp", "litellm"], default="aidp", help="LLM provider transport")
    parser.add_argument("--aidp-ak", default=os.getenv("AIDP_AK"), help="AIDP access key; defaults to AIDP_AK")
    parser.add_argument("--api-key", default=os.getenv("CUSTOM_LLM_API_KEY") or os.getenv("DEEPSEEK_API_KEY"), help="Model API key for non-AIDP providers")
    parser.add_argument("--base-url", default=os.getenv("CUSTOM_LLM_BASE_URL") or os.getenv("CUSTOM_LLM_API_BASE"), help="Optional custom model base URL")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--discriminator-temperature", type=float, default=1.0)
    parser.add_argument("--value-temperature", type=float, default=0.2)
    parser.add_argument("--max-tokens", type=int, default=2000)
    parser.add_argument("--timeout", type=float, default=90.0)
    parser.add_argument("--max-iterations", type=int, default=20)
    parser.add_argument("--max-depth", type=int, default=20)
    parser.add_argument("--max-finished-nodes", type=int, default=2)
    parser.add_argument("--max-expansions", type=int, default=3)
    parser.add_argument("--discriminator-agents", type=int, default=5)
    parser.add_argument("--discriminator-rounds", type=int, default=3)
    parser.add_argument("--experience-top-k", type=int, default=3)
    parser.add_argument("--workspace-dir", default="tmp/koco/workspaces")
    parser.add_argument("--output-root", default="tmp/koco")
    parser.add_argument("--experience-path", default="", help="Experience bank JSON to read/write (default: <output-root>/experience/koco_experience_bank.json)")
    parser.add_argument("--reuse-workspace", action="store_true", help="Do not delete existing per-instance workspace before run")
    parser.add_argument("--force-parse", action="store_true", help="Regenerate KOCO input JSONL before running")
    parser.add_argument("--concurrency", type=int, default=1, help="Concurrent KOCO instances for infer")
    parser.add_argument("--usage-stage", default="infer", help="Stage label written to AIDP usage JSONL")
    parser.add_argument("--usage-log", default=os.getenv("KOCO_USAGE_LOG"), help="Path to AIDP per-call usage JSONL")


def add_build_experience_args(parser):
    parser.add_argument("--provider", choices=["aidp", "litellm"], default="aidp", help="LLM provider transport")
    parser.add_argument("--aidp-ak", default=os.getenv("AIDP_AK"), help="AIDP access key; defaults to AIDP_AK")
    parser.add_argument("--api-key", default=os.getenv("CUSTOM_LLM_API_KEY") or os.getenv("DEEPSEEK_API_KEY"), help="Model API key for non-AIDP providers")
    parser.add_argument("--base-url", default=os.getenv("CUSTOM_LLM_BASE_URL") or os.getenv("CUSTOM_LLM_API_BASE"), help="Optional custom model base URL")
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--max-tokens", type=int, default=2000)
    parser.add_argument("--timeout", type=float, default=90.0)
    parser.add_argument("--usage-stage", default="build-experience", help="Stage label written to AIDP usage JSONL")
    parser.add_argument("--usage-log", default=os.getenv("KOCO_USAGE_LOG"), help="Path to AIDP per-call usage JSONL")
    parser.add_argument("--no-llm-experience", action="store_true", help="Build mechanical eval summaries without LLM experience summarization")


def main():
    parser = argparse.ArgumentParser(description="Run SWE-Exp search on KOCO-bench function tasks")
    subparsers = parser.add_subparsers(dest="command", required=True)

    infer = subparsers.add_parser("infer", help="Run inference and write KOCO output JSONL")
    add_common_args(infer)
    add_infer_args(infer)

    eval_parser = subparsers.add_parser("eval", help="Run KOCO Docker evaluation on SWE-Exp outputs")
    add_common_args(eval_parser)

    build_exp = subparsers.add_parser("build-experience", help="Build KOCO experience bank from evaluation results")
    add_common_args(build_exp)
    build_exp.add_argument("--output-root", default="tmp/koco")
    add_build_experience_args(build_exp)

    usage_summary = subparsers.add_parser("usage-summary", help="Aggregate AIDP usage JSONL files")
    usage_summary.add_argument("usage_logs", nargs="+", help="One or more AIDP usage JSONL files")

    run = subparsers.add_parser("run", help="Run inference then KOCO Docker evaluation")
    add_common_args(run)
    add_infer_args(run)
    run.add_argument("--eval-on-infer-failure", action="store_true", help="Run eval even if some inference instances fail")

    args = parser.parse_args()
    if args.command == "infer":
        return cmd_infer(args)
    if args.command == "eval":
        return cmd_eval(args)
    if args.command == "build-experience":
        return cmd_build_experience(args)
    if args.command == "usage-summary":
        return cmd_usage_summary(args)
    if args.command == "run":
        return cmd_run(args)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
