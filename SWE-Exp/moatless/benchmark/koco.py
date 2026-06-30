import ast
import hashlib
import importlib
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import threading
from fnmatch import fnmatch
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

SWE_EXP_ROOT = Path(__file__).resolve().parents[2]
COMBINED_ROOT = SWE_EXP_ROOT.parent
KOCO_PROJECT_ROOT = Path(
    os.environ.get(
        "KOCO_DOMAIN_CODE_GENERATION_ROOT",
        COMBINED_ROOT / "KOCO-bench" / "KOCO-bench-en" / "domain_code_generation",
    )
).resolve()
KOCO_SCRIPTS_DIR = KOCO_PROJECT_ROOT / "scripts"
_OUTPUT_LOCKS: dict[str, threading.Lock] = {}
_OUTPUT_LOCKS_GUARD = threading.Lock()


def _ensure_koco_scripts_on_path() -> None:
    scripts = str(KOCO_SCRIPTS_DIR)
    if scripts not in sys.path:
        sys.path.insert(0, scripts)


def safe_name(value: str, max_len: int = 80) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("._")
    if not cleaned:
        cleaned = "item"
    digest = hashlib.sha1(value.encode("utf-8")).hexdigest()[:8]
    return f"{cleaned[:max_len]}_{digest}"


def model_label_from_model(model: str) -> str:
    return safe_name(model.split("/")[-1] or model)


def load_jsonl(path: Path) -> list[dict]:
    records = []
    if not path.exists():
        return records
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def save_jsonl(records: Iterable[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for record in records:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")


def get_koco_examples(framework: str) -> list[str]:
    examples_dir = KOCO_PROJECT_ROOT / framework / "test_examples"
    if not examples_dir.is_dir():
        raise FileNotFoundError(f"KOCO test_examples directory not found: {examples_dir}")
    return sorted(path.name for path in examples_dir.iterdir() if path.is_dir())


def get_koco_frameworks() -> list[str]:
    candidates = []
    for path in KOCO_PROJECT_ROOT.iterdir():
        if not path.is_dir():
            continue
        if (path / "test_examples").is_dir() and (path / "knowledge_corpus").is_dir():
            candidates.append(path.name)
    return sorted(candidates)


def koco_data_file(framework: str, example: str) -> Path:
    local_root = Path(
        os.environ.get(
            "KOCO_SWE_EXP_INPUT_ROOT",
            SWE_EXP_ROOT / "tmp" / "koco" / "inputs",
        )
    )
    local_file = local_root / framework / f"algorithm_methods_data_{example}.jsonl"
    if local_file.exists():
        return local_file

    legacy_file = KOCO_SCRIPTS_DIR / "data" / framework / f"algorithm_methods_data_{example}.jsonl"
    if legacy_file.exists():
        return legacy_file
    return local_file


def koco_local_data_file(framework: str, example: str) -> Path:
    local_root = Path(
        os.environ.get(
            "KOCO_SWE_EXP_INPUT_ROOT",
            SWE_EXP_ROOT / "tmp" / "koco" / "inputs",
        )
    )
    return local_root / framework / f"algorithm_methods_data_{example}.jsonl"


def koco_output_dir(framework: str, model_label: str) -> Path:
    return Path(
        os.environ.get(
            "KOCO_SWE_EXP_OUTPUT_ROOT",
            SWE_EXP_ROOT / "tmp" / "koco" / "outputs",
        )
    ) / framework / "swe_exp" / model_label


def koco_output_file(framework: str, example: str, model_label: str) -> Path:
    return koco_output_dir(framework, model_label) / f"algorithm_methods_data_{example}_output.jsonl"


def koco_result_file(framework: str, example: str, model_label: str) -> Path:
    return koco_output_dir(framework, model_label) / f"algorithm_methods_data_{example}_result.jsonl"


def koco_metrics_file(framework: str, example: str, model_label: str) -> Path:
    return koco_output_dir(framework, model_label) / f"algorithm_methods_data_{example}_result.metrics.json"


def ensure_koco_input_data(framework: str, example: str, force: bool = False) -> Path:
    """Generate KOCO JSONL input for one example if it does not exist."""
    output_file = koco_local_data_file(framework, example) if force else koco_data_file(framework, example)
    if output_file.exists() and not force:
        return output_file

    _ensure_koco_scripts_on_path()
    from parse_algorithm_methods import parse_markdown_file, process_functions
    from prompts_construction import add_prompts_to_data

    input_md = (
        KOCO_PROJECT_ROOT
        / framework
        / "test_examples"
        / example
        / "requirements"
        / "03_algorithm_and_core_methods.md"
    )
    code_base = KOCO_PROJECT_ROOT / framework / "test_examples" / example / "code"
    test_base = code_base / "tests"
    metadata_path = KOCO_PROJECT_ROOT / framework / "knowledge_corpus" / "metadata.json"

    if not input_md.exists():
        raise FileNotFoundError(f"KOCO requirements file not found: {input_md}")
    if not code_base.is_dir():
        raise FileNotFoundError(f"KOCO code directory not found: {code_base}")

    output_file.parent.mkdir(parents=True, exist_ok=True)
    functions_data = parse_markdown_file(str(input_md))
    processed = process_functions(functions_data, str(code_base), str(test_base))
    save_jsonl(processed, output_file)
    add_prompts_to_data(str(output_file), str(output_file), str(metadata_path))
    return output_file


def parse_impl_location(impl_loc: str) -> tuple[str | None, int, int]:
    parts = impl_loc.strip().strip("`").replace("\\", "/").split(":line ")
    if len(parts) != 2:
        return None, 0, 0
    rel_path = parts[0]
    if rel_path.startswith("code/"):
        rel_path = rel_path[5:]
    try:
        start_s, end_s = parts[1].split("-")
        return rel_path, int(start_s), int(end_s)
    except ValueError:
        return None, 0, 0


def make_koco_instance_id(framework: str, example: str, function_name: str) -> str:
    raw = f"{framework}/{example}/{function_name}"
    return f"koco__{safe_name(framework, 32)}__{safe_name(example, 48)}__{safe_name(function_name, 80)}__{hashlib.sha1(raw.encode('utf-8')).hexdigest()[:8]}"


def build_problem_statement(record: dict, framework: str, example: str) -> str:
    function_name = record["function_name"]
    rel_path, start, end = parse_impl_location(record.get("implementation_location", ""))
    target_file = f"code/{rel_path}" if rel_path else "the target source file"

    system_context = ""
    user_task = ""
    for message in record.get("prompt") or []:
        if message.get("role") == "system":
            system_context = message.get("content", "")
        elif message.get("role") == "user":
            user_task = message.get("content", "")

    location_hint = ""
    if rel_path and start:
        location_hint = (
            f"\nTarget location: `{target_file}`, original lines {start}-{end}. "
            "The target function body has been replaced with `raise NotImplementedError` in this workspace."
        )

    return f"""You are solving a KOCO-bench function implementation task.

Framework: {framework}
Example: {example}
Target function: `{function_name}`{location_hint}

{system_context}

{user_task}

Workspace layout:
- `code/` contains the development repository for this KOCO example.
- `knowledge_corpus/` contains framework documentation and reference material.

Rules:
- Only edit `{target_file}`.
- Only implement the target function `{function_name}`.
- Do not edit tests, requirements, generated data, or knowledge corpus files.
- Do not create helper scripts.
- Finish when the implementation has been written.
"""


def build_koco_instance(record: dict, framework: str, example: str) -> dict:
    rel_path, start, end = parse_impl_location(record.get("implementation_location", ""))
    if not rel_path:
        raise ValueError(f"Invalid implementation_location for {record.get('function_name')}")

    function_name = record["function_name"]
    return {
        "instance_id": make_koco_instance_id(framework, example, function_name),
        "framework": framework,
        "example": example,
        "function_name": function_name,
        "problem_statement": build_problem_statement(record, framework, example),
        "record": record,
        "source_root": str(KOCO_PROJECT_ROOT / framework / "test_examples" / example / "code"),
        "knowledge_corpus_root": str(KOCO_PROJECT_ROOT / framework / "knowledge_corpus"),
        "target_rel_path": rel_path,
        "target_path": f"code/{rel_path}",
        "target_start_line": start,
        "target_end_line": end,
    }


def load_koco_instances(
    framework: str,
    example: str,
    function_names: set[str] | None = None,
    force_parse: bool = False,
) -> list[dict]:
    data_file = ensure_koco_input_data(framework, example, force=force_parse)
    records = load_jsonl(data_file)
    instances = []
    for record in records:
        if function_names and record.get("function_name") not in function_names:
            continue
        instances.append(build_koco_instance(record, framework, example))
    return instances


def iter_koco_instances(
    framework: str,
    test_example: str | None = None,
    function_names: set[str] | None = None,
    force_parse: bool = False,
) -> Iterable[dict]:
    examples = [test_example] if test_example else get_koco_examples(framework)
    for example in examples:
        yield from load_koco_instances(
            framework=framework,
            example=example,
            function_names=function_names,
            force_parse=force_parse,
        )


def _stub_one_function_regex(lines: list[str], start: int, end: int) -> list[str]:
    def_idx = None
    for i in range(start - 1, min(end, len(lines))):
        if re.match(r"\s*(async\s+)?def\s+", lines[i]):
            def_idx = i
            break
    if def_idx is None:
        return lines

    body_start = def_idx + 1
    while body_start < end and body_start < len(lines):
        stripped = lines[body_start].strip()
        if stripped and not stripped.startswith("#") and not stripped.startswith("@"):
            break
        body_start += 1

    body_indent = len(lines[def_idx]) - len(lines[def_idx].lstrip()) + 4
    stub_lines = [
        f"{' ' * body_indent}# TODO: implement this function\n",
        f"{' ' * body_indent}raise NotImplementedError\n",
    ]
    return lines[:body_start] + stub_lines + lines[end:]


def _stub_one_function(lines: list[str], start: int, end: int) -> list[str]:
    source = "".join(lines)
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return _stub_one_function_regex(lines, start, end)

    target = None
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and start <= node.lineno <= end:
            target = node
            if node.lineno == start:
                break
    if target is None or not target.body:
        return _stub_one_function_regex(lines, start, end)

    body = target.body
    has_docstring = (
        isinstance(body[0], ast.Expr)
        and isinstance(body[0].value, ast.Constant)
        and isinstance(body[0].value.value, str)
    )
    if has_docstring:
        stub_start = body[0].end_lineno
        indent = body[1].col_offset if len(body) > 1 else body[0].col_offset
        prefix = lines[:stub_start]
    else:
        stub_start = body[0].lineno
        indent = body[0].col_offset
        prefix = lines[: stub_start - 1]

    stub_lines = [
        f"{' ' * indent}# TODO: implement this function\n",
        f"{' ' * indent}raise NotImplementedError\n",
    ]
    return prefix + stub_lines + lines[end:]


def _stub_target_function(file_path: Path, start: int, end: int) -> None:
    lines = file_path.read_text(encoding="utf-8").splitlines(keepends=True)
    file_path.write_text("".join(_stub_one_function(lines, start, end)), encoding="utf-8")


def create_koco_repository(
    instance: dict,
    workspace_base_dir: str | Path | None = None,
    reset: bool = True,
):
    """Create an isolated KOCO workspace with code/ and knowledge_corpus/."""
    if workspace_base_dir is None:
        workspace_base_dir = SWE_EXP_ROOT / "tmp" / "koco" / "workspaces"
    workspace_base_dir = Path(workspace_base_dir)
    workspace_root = workspace_base_dir / safe_name(instance["instance_id"], 140)

    if reset and workspace_root.exists():
        shutil.rmtree(workspace_root)
    workspace_root.mkdir(parents=True, exist_ok=True)

    def ignore(_dir: str, names: list[str]) -> set[str]:
        ignored = {"__pycache__", ".pytest_cache", ".mypy_cache", "test_code"}
        return ignored.intersection(names)

    code_dst = workspace_root / "code"
    knowledge_dst = workspace_root / "knowledge_corpus"
    if not code_dst.exists():
        shutil.copytree(instance["source_root"], code_dst, symlinks=True, ignore=ignore)
    if not knowledge_dst.exists():
        shutil.copytree(instance["knowledge_corpus_root"], knowledge_dst, symlinks=True, ignore=ignore)

    target_file = code_dst / instance["target_rel_path"]
    if not target_file.exists():
        raise FileNotFoundError(f"KOCO target file not found in workspace: {target_file}")
    _stub_target_function(target_file, instance["target_start_line"], instance["target_end_line"])

    from moatless.repository.file import FileRepository

    return FileRepository(repo_path=str(workspace_root))


def _is_not_implemented_stub(node: ast.AST) -> bool:
    if not isinstance(node, ast.Raise) or node.exc is None:
        return False
    if isinstance(node.exc, ast.Name):
        return node.exc.id == "NotImplementedError"
    if isinstance(node.exc, ast.Call) and isinstance(node.exc.func, ast.Name):
        return node.exc.func.id == "NotImplementedError"
    return False


def extract_function_source(source: str, function_name: str) -> str:
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return ""

    lines = source.splitlines(keepends=True)
    parts = function_name.split(".")
    class_name, method_name = (parts[0], parts[1]) if len(parts) == 2 else (None, function_name)

    target = None
    for node in ast.walk(tree):
        if class_name and isinstance(node, ast.ClassDef) and node.name == class_name:
            for item in node.body:
                if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)) and item.name == method_name:
                    target = item
                    break
        elif not class_name and isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == method_name:
            target = node
            break
    if target is None and class_name:
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == method_name:
                target = node
                break
    if target is None or not target.body:
        return ""

    body = target.body
    real_body = body[1:] if (
        isinstance(body[0], ast.Expr)
        and isinstance(body[0].value, ast.Constant)
        and isinstance(body[0].value.value, str)
    ) else body
    if len(real_body) == 1 and _is_not_implemented_stub(real_body[0]):
        return ""

    start = target.decorator_list[0].lineno if target.decorator_list else target.lineno
    end = target.end_lineno
    return "".join(lines[start - 1 : end]).strip()


def extract_completion_from_file_context(file_context, instance: dict) -> str:
    context_file = file_context.get_file(instance["target_path"])
    if not context_file:
        return ""
    return extract_function_source(context_file.content, instance["function_name"])


def upsert_output_record(
    instance: dict,
    model_label: str,
    completion: str,
    status: str,
    model_patch: str = "",
    source_tree_path: str = "",
) -> Path:
    output_file = koco_output_file(instance["framework"], instance["example"], model_label)
    lock_key = str(output_file.resolve())
    with _OUTPUT_LOCKS_GUARD:
        lock = _OUTPUT_LOCKS.setdefault(lock_key, threading.Lock())
    with lock:
        existing = {record.get("function_name"): record for record in load_jsonl(output_file)}
        record = dict(instance["record"])
        record["completions"] = [completion]
        record["status"] = status
        if model_patch:
            record["model_patch"] = model_patch
        if source_tree_path:
            record["source_tree_path"] = source_tree_path
        existing[instance["function_name"]] = record
        save_jsonl(existing.values(), output_file)
    return output_file


def run_koco_eval(framework: str, model_label: str, test_example: str | None = None) -> int:
    _ensure_koco_scripts_on_path()
    from config import get_docker_image

    image = get_docker_image(framework)
    examples = [test_example] if test_example else get_koco_examples(framework)
    data_subdir = f"scripts/data/{framework}/swe_exp/{model_label}"
    local_output_dir = koco_output_dir(framework, model_label)
    koco_data_dir = KOCO_SCRIPTS_DIR / "data" / framework / "swe_exp" / model_label
    if local_output_dir.resolve() != koco_data_dir.resolve():
        print(
            "ERROR: KOCO eval expects outputs under KOCO scripts/data. "
            f"Copy {local_output_dir} to {koco_data_dir} on the eval machine before running eval."
        )
        return 1
    container_mnt = "/workspace/project"
    host_root = str(KOCO_PROJECT_ROOT)

    try:
        subprocess.run(["docker", "info"], capture_output=True, check=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("ERROR: Docker is not running or not installed.")
        return 1

    if not subprocess.run(["docker", "images", "-q", image], capture_output=True, text=True).stdout.strip():
        print(f"ERROR: Docker image '{image}' not found.")
        return 1

    failures = 0
    for example in examples:
        input_name = f"algorithm_methods_data_{example}_output.jsonl"
        output_name = f"algorithm_methods_data_{example}_result.jsonl"
        host_input = KOCO_PROJECT_ROOT / data_subdir / input_name
        if not host_input.exists():
            print(f"{example}: SKIP (input file not found: {host_input})")
            continue

        docker_cmd = [
            "docker",
            "run",
            "--rm",
            "--gpus",
            "all",
            "-v",
            f"{host_root}:{container_mnt}",
            image,
            "python3",
            f"{container_mnt}/scripts/execution_evaluation_pure.py",
            "--source_dir",
            f"{container_mnt}/{framework}/test_examples/{example}/code",
            "--input_file",
            f"{container_mnt}/{data_subdir}/{input_name}",
            "--output_file",
            f"{container_mnt}/{data_subdir}/{output_name}",
        ]
        if platform.system() == "Linux":
            uid, gid = os.getuid(), os.getgid()
            docker_cmd[3:3] = ["--user", f"{uid}:{gid}", "-e", "HOME=/tmp", "-e", "USER=benchuser"]

        print(f"\n--- Evaluating {example} with {image} ---")
        rc = subprocess.run(docker_cmd).returncode
        failures += 1 if rc else 0

    model_dir = koco_output_dir(framework, model_label)
    saved_argv = sys.argv
    try:
        aggregate_metrics = importlib.import_module("aggregate_metrics")
        sys.argv = ["aggregate_metrics.py", "--model_dir", str(model_dir), "--framework", framework]
        agg_rc = aggregate_metrics.main()
        failures += 1 if agg_rc else 0
    finally:
        sys.argv = saved_argv

    return 1 if failures else 0


@dataclass
class LocalSpanHit:
    span_id: str
    rank: int = 0
    tokens: int = 0
    start_line: int | None = None
    end_line: int | None = None


@dataclass
class LocalSearchCodeHit:
    file_path: str
    spans: list[LocalSpanHit] = field(default_factory=list)

    @property
    def span_ids(self):
        return [span.span_id for span in self.spans]


@dataclass
class LocalSearchCodeResponse:
    hits: list[LocalSearchCodeHit] = field(default_factory=list)
    message: str | None = None

    def sum_tokens(self):
        return sum(sum(span.tokens for span in hit.spans) for hit in self.hits)


class KocoCodeIndex:
    """Local KOCO code index compatible with SWE-Exp search actions.

    This avoids SWE-bench's prebuilt index assumption while preserving the
    FindClass/FindFunction/FindCodeSnippet/SemanticSearch action surface.
    It indexes Python classes/functions with AST and uses lexical scoring for
    semantic-style searches. The workspace still exposes `knowledge_corpus/` so
    agents can view docs explicitly when needed.
    """

    def __init__(self, repository, max_results: int = 25):
        self._repository = repository
        self.max_results = max_results
        self._functions: dict[str, list[tuple[str, str, int, int]]] = {}
        self._classes: dict[str, list[tuple[str, str, int, int]]] = {}
        self._file_text: dict[str, str] = {}
        self._build()

    def _iter_python_files(self) -> Iterable[str]:
        root = Path(self._repository.repo_path)
        for path in root.rglob("*.py"):
            rel = path.relative_to(root).as_posix()
            if any(part in {"__pycache__", ".pytest_cache"} for part in rel.split("/")):
                continue
            yield rel

    def _build(self) -> None:
        for rel_path in self._iter_python_files():
            full_path = Path(self._repository.repo_path) / rel_path
            try:
                source = full_path.read_text(encoding="utf-8")
            except UnicodeDecodeError:
                source = full_path.read_text(encoding="utf-8", errors="ignore")
            self._file_text[rel_path] = source
            try:
                tree = ast.parse(source)
            except SyntaxError:
                continue

            parents: dict[ast.AST, ast.AST] = {}
            for parent in ast.walk(tree):
                for child in ast.iter_child_nodes(parent):
                    parents[child] = parent

            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef):
                    self._classes.setdefault(node.name, []).append(
                        (rel_path, node.name, node.lineno, getattr(node, "end_lineno", node.lineno))
                    )
                elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    parent = parents.get(node)
                    if isinstance(parent, ast.ClassDef):
                        span_id = f"{parent.name}.{node.name}"
                    else:
                        span_id = node.name
                    self._functions.setdefault(node.name, []).append(
                        (rel_path, span_id, node.lineno, getattr(node, "end_lineno", node.lineno))
                    )

    def _match_pattern(self, file_path: str, file_pattern: str | None) -> bool:
        if not file_pattern:
            return True
        pattern = file_pattern.lstrip("/")
        return fnmatch(file_path, pattern) or fnmatch(file_path, f"**/{pattern}")

    def _response(self, entries: list[tuple[str, str, int, int]]) :
        by_file: dict[str, LocalSearchCodeHit] = {}
        for rank, (file_path, span_id, start_line, end_line) in enumerate(entries[: self.max_results]):
            hit = by_file.setdefault(file_path, LocalSearchCodeHit(file_path=file_path))
            hit.spans.append(
                LocalSpanHit(
                    span_id=span_id,
                    rank=rank,
                    tokens=0,
                    start_line=start_line,
                    end_line=end_line,
                )
            )
        return LocalSearchCodeResponse(hits=list(by_file.values()))

    def find_function(self, function_name: str, class_name: str | None = None, file_pattern: str | None = None):
        entries = []
        for file_path, span_id, start_line, end_line in self._functions.get(function_name, []):
            if class_name and span_id != f"{class_name}.{function_name}":
                continue
            if self._match_pattern(file_path, file_pattern):
                entries.append((file_path, span_id, start_line, end_line))
        return self._response(entries)

    def find_class(self, class_name: str, file_pattern: str | None = None):
        entries = [
            (file_path, span_id, start_line, end_line)
            for file_path, span_id, start_line, end_line in self._classes.get(class_name, [])
            if self._match_pattern(file_path, file_pattern)
        ]
        return self._response(entries)

    def semantic_search(
        self,
        query: str | None = None,
        code_snippet: str | None = None,
        file_pattern: str | None = None,
        category: str | None = None,
        max_results: int = 100,
        **_kwargs,
    ):
        terms = [
            term.lower()
            for term in re.findall(r"[A-Za-z_][A-Za-z0-9_]+", query or code_snippet or "")
            if len(term) >= 3
        ]
        scored: list[tuple[int, str, str]] = []
        for file_path, source in self._file_text.items():
            if category == "test" and "test" not in file_path.lower():
                continue
            if category == "implementation" and "test" in file_path.lower():
                continue
            if not self._match_pattern(file_path, file_pattern):
                continue
            lower = source.lower()
            score = sum(lower.count(term) for term in terms)
            if score <= 0:
                continue
            spans = []
            try:
                tree = ast.parse(source)
                for node in ast.walk(tree):
                    if isinstance(node, ast.ClassDef):
                        spans.append((node.name, getattr(node, "lineno", 0), getattr(node, "end_lineno", getattr(node, "lineno", 0))))
                    elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        spans.append((node.name, getattr(node, "lineno", 0), getattr(node, "end_lineno", getattr(node, "lineno", 0))))
            except SyntaxError:
                spans = []
            if spans:
                for span_id, start_line, end_line in spans[:2]:
                    scored.append((score, file_path, span_id, start_line, end_line))

        scored.sort(reverse=True, key=lambda item: item[0])
        return self._response(
            [
                (file_path, span_id, start_line, end_line)
                for _score, file_path, span_id, start_line, end_line in scored[:max_results]
            ]
        )

    def find_test_files(self, file_path: str, query: str | None = None, max_results: int = 2, max_spans: int = 2):
        from moatless.schema import FileWithSpans

        base_terms = set(Path(file_path).stem.lower().split("_"))
        results = []
        for path in self._file_text:
            lower = path.lower()
            if "test" not in lower:
                continue
            score = sum(1 for term in base_terms if term and term in lower)
            if query:
                score += sum(1 for term in query.lower().split("/") if term and term in lower)
            if score:
                results.append((score, path))
        results.sort(reverse=True)
        return [FileWithSpans(file_path=path, span_ids=[]) for _score, path in results[:max_results]]


class KocoExperienceSelector:
    """KOCO implementation of SWE-Exp's experiencer interface.

    ActionAgent only depends on a small protocol from SelectAgent:
    select_workflow, generalize_workflow, polish_workflow, get_json and persist.
    This class provides the same protocol with KOCO-native files and prompts.
    """

    def __init__(
        self,
        completion,
        instance: dict,
        experience_path: str | Path,
        persist_dir: str | Path,
        top_k: int = 3,
    ):
        self._completion = completion
        self.instance = instance
        self.instance_id = instance["instance_id"]
        self.experience_path = Path(experience_path)
        self.persist_dir = str(persist_dir)
        self.top_k = top_k

    def get_json(self, path):
        path = Path(path)
        if not path.exists():
            return {"old_experiences": {}, "HET": {}, "trajectory": None}
        with path.open("r", encoding="utf-8") as fh:
            return json.load(fh)

    def persist(self, experiences: dict) -> None:
        path = Path(self.persist_dir)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as fh:
            json.dump(experiences, fh, ensure_ascii=False, indent=2)

    def _load_bank(self) -> dict:
        if not self.experience_path.exists():
            return {}
        with self.experience_path.open("r", encoding="utf-8") as fh:
            return json.load(fh)

    def _lexical_score(self, text_a: str, text_b: str) -> int:
        terms_a = set(re.findall(r"[A-Za-z_][A-Za-z0-9_]+", text_a.lower()))
        terms_b = set(re.findall(r"[A-Za-z_][A-Za-z0-9_]+", text_b.lower()))
        return len(terms_a & terms_b)

    def select_workflow(self, n: int = 1) -> dict:
        bank = self._load_bank()
        if not bank:
            return {}
        current = self.instance.get("problem_statement", "")
        scored = []
        for key, exp in bank.items():
            if key == self.instance_id:
                continue
            issue = exp.get("issue", "")
            scored.append((self._lexical_score(current, issue), key, exp))
        scored.sort(reverse=True, key=lambda item: item[0])
        selected = {
            key: exp
            for score, key, exp in scored[: max(n, self.top_k)]
            if score > 0 or len(scored) <= max(n, self.top_k)
        }
        return selected

    def _chat_json(self, system_prompt: str, user_prompt: str) -> dict:
        response = self._completion._litellm_base_completion(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ]
        )
        content = response.choices[0].message.content
        try:
            import json_repair

            return json_repair.loads(content)
        except Exception:
            try:
                return json.loads(content)
            except Exception:
                return {}

    def generalize_workflow(self, old_experiences: dict, type: str, history, cur_code, instruction) -> str:
        if not old_experiences:
            return ""
        system_prompt = (
            "You adapt past software-engineering experiences to a current KOCO-bench "
            "function implementation task. Return JSON: {\"experiences\": [\"...\"]}."
        )
        user_prompt = json.dumps(
            {
                "current_task": self.instance.get("problem_statement", ""),
                "experience_type": type,
                "history": history,
                "instruction": instruction,
                "past_experiences": old_experiences,
            },
            ensure_ascii=False,
        )
        data = self._chat_json(system_prompt, user_prompt)
        experiences = data.get("experiences") or []
        if isinstance(experiences, str):
            experiences = [experiences]
        return "".join(f"***Experience {idx + 1}***: {exp}\n" for idx, exp in enumerate(experiences))

    def polish_workflow(self, old_experiences: dict, type: str, history, instruction) -> str:
        if not old_experiences:
            return instruction
        system_prompt = (
            "You improve the next instruction for a KOCO-bench code agent using "
            "past experience. Preserve the original intent. Return JSON: "
            "{\"enhanced_instruction\": \"...\"}."
        )
        user_prompt = json.dumps(
            {
                "current_task": self.instance.get("problem_statement", ""),
                "experience_type": type,
                "history": history,
                "instruction": instruction,
                "past_experiences": old_experiences,
            },
            ensure_ascii=False,
        )
        data = self._chat_json(system_prompt, user_prompt)
        return data.get("enhanced_instruction") or instruction


def append_koco_experience(instance: dict, result: dict, experience_path: str | Path) -> None:
    path = Path(experience_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        with path.open("r", encoding="utf-8") as fh:
            bank = json.load(fh)
    else:
        bank = {}

    status = result.get("status", "")
    if status not in {"success", "no_completion", "invalid_patch_scope", "error"}:
        return
    bank[instance["instance_id"]] = {
        "issue": instance.get("problem_statement", ""),
        "framework": instance.get("framework"),
        "example": instance.get("example"),
        "function_name": instance.get("function_name"),
        "flag": "success" if status == "success" else "failed",
        "perspective": [
            f"Target function: {instance.get('function_name')}",
            f"Status: {status}",
            f"Error: {result.get('error', '')}",
        ],
        "trajectory": result.get("trajectory"),
        "output_file": result.get("output_file"),
    }
    with path.open("w", encoding="utf-8") as fh:
        json.dump(bank, fh, ensure_ascii=False, indent=2)


def _truncate_for_prompt(value: str, max_chars: int = 6000) -> str:
    if len(value) <= max_chars:
        return value
    return value[:max_chars] + "\n...[truncated]"


def _parse_json_object(content: str) -> dict:
    try:
        import json_repair

        data = json_repair.loads(content)
    except Exception:
        data = json.loads(content)
    return data if isinstance(data, dict) else {}


def _summarize_koco_experience_with_llm(
    completion,
    *,
    framework: str,
    example: str,
    record: dict,
    passed: bool,
    avg_pass_ratio: float,
) -> dict:
    system_prompt = (
        "You summarize KOCO-bench code-generation evaluation results into reusable "
        "SWE-Exp experience. Return only JSON with keys: issue_type, description, "
        "perspective. perspective must be a list of concise implementation lessons. "
        "Use ASCII text only."
    )
    user_prompt = json.dumps(
        {
            "framework": framework,
            "example": example,
            "function_name": record.get("function_name"),
            "implementation_location": record.get("implementation_location"),
            "overview": record.get("overview"),
            "function_signature": record.get("function_signature"),
            "detailed_description": record.get("detailed_description"),
            "evaluation_passed": passed,
            "average_pass_ratio": avg_pass_ratio,
            "results": record.get("results"),
            "pass_ratios": record.get("pass_ratios"),
            "status": record.get("status"),
            "generated_completion": _truncate_for_prompt("\n\n".join(record.get("completions") or []), 8000),
        },
        ensure_ascii=False,
    )
    response = completion._litellm_base_completion(
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
    )
    content = response.choices[0].message.content or "{}"
    data = _parse_json_object(content)
    perspective = data.get("perspective") or []
    if isinstance(perspective, str):
        perspective = [perspective]
    return {
        "issue_type": str(data.get("issue_type") or "koco_function_implementation"),
        "description": str(
            data.get("description")
            or (
                f"KOCO {framework}/{example}/{record.get('function_name')}: "
                f"passed={passed}, avg_pass_ratio={avg_pass_ratio:.4f}"
            )
        ),
        "perspective": [str(item) for item in perspective if str(item).strip()],
    }


def build_koco_experience_from_eval(
    framework: str,
    example: str,
    model_label: str,
    experience_path: str | Path,
    result_file: str | Path | None = None,
    metrics_file: str | Path | None = None,
    completion=None,
) -> Path:
    """Build/update KOCO experience bank from KOCO Docker evaluation results."""
    result_path = Path(result_file) if result_file else koco_result_file(framework, example, model_label)
    metrics_path = Path(metrics_file) if metrics_file else koco_metrics_file(framework, example, model_label)
    if not result_path.exists():
        raise FileNotFoundError(f"KOCO eval result file not found: {result_path}")

    metrics = {}
    if metrics_path.exists():
        with metrics_path.open("r", encoding="utf-8") as fh:
            metrics = json.load(fh)

    path = Path(experience_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        with path.open("r", encoding="utf-8") as fh:
            bank = json.load(fh)
    else:
        bank = {}

    for record in load_jsonl(result_path):
        function_name = record.get("function_name")
        if not function_name:
            continue
        instance_id = make_koco_instance_id(framework, example, function_name)
        passed = bool(any(record.get("results") or []))
        pass_ratios = record.get("pass_ratios") or []
        avg_pass_ratio = sum(pass_ratios) / len(pass_ratios) if pass_ratios else 0.0
        issue = build_problem_statement(record, framework, example)
        summary = {}
        if completion is not None:
            summary = _summarize_koco_experience_with_llm(
                completion,
                framework=framework,
                example=example,
                record=record,
                passed=passed,
                avg_pass_ratio=avg_pass_ratio,
            )
        perspective = summary.get("perspective") or [
            f"Function: {function_name}",
            f"Evaluation passed: {passed}",
            f"Average pass ratio: {avg_pass_ratio:.4f}",
            f"Results: {record.get('results')}",
            f"Pass ratios: {pass_ratios}",
        ]
        bank[instance_id] = {
            "issue": issue,
            "framework": framework,
            "example": example,
            "function_name": function_name,
            "flag": "success" if passed else "failed",
            "issue_type": summary.get("issue_type") or "koco_function_implementation",
            "description": summary.get("description")
            or (
                f"KOCO {framework}/{example}/{function_name}: "
                f"passed={passed}, avg_pass_ratio={avg_pass_ratio:.4f}"
            ),
            "perspective": perspective,
            "metrics": metrics,
            "result_file": str(result_path),
            "metrics_file": str(metrics_path) if metrics_path.exists() else "",
        }

    with path.open("w", encoding="utf-8") as fh:
        json.dump(bank, fh, ensure_ascii=False, indent=2)
    return path
