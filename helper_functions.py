"""Helper functions for the local LLM evaluation notebooks.

The notebook tells the experiment story: choose models, generate candidate
programs, test those programs, and compare results. This module holds the
mechanical pieces so the notebook stays readable.

Read this file from top to bottom as a small pipeline:

1. Describe the shapes of the data we will pass around.
2. Generate Python files from an Ollama model.
3. Run each generated file in a Docker Python container.
4. Turn many test results into summaries we can display in the notebook.

The comments are intentionally more narrative than production code comments.
They are here to help you connect each helper to the larger evaluation idea.
"""

import os
import re
import selectors
import shutil
import subprocess
import time
import uuid
from pathlib import Path
from statistics import median
from typing import Literal, TypedDict

import ollama
from IPython.display import Markdown, display

# ---------------------------------------------------------------------------
# Data shapes
# ---------------------------------------------------------------------------
#
# Python dictionaries are flexible, but flexible data can become hard to
# reason about. TypedDict lets us keep the friendliness of dictionaries while
# documenting which keys each dictionary should contain.

TestCase = tuple[list[str], str]
"""A command-line argument list paired with the expected standard output."""

AnswerFormat = Literal["integer_only", "final_answer"]
"""The response convention used to locate an arithmetic answer."""


class TestResult(TypedDict):
    """The result of running one generated program against one test case.

    Attributes:
        args: Command-line arguments passed to the generated program.
        expected: The exact output we wanted from the program.
        actual: The exact output the program printed.
        stderr: Error text printed by Python or Docker.
        returncode: The process exit code, or None if the run timed out.
        passed: Whether this one test case succeeded.
    """

    args: list[str]
    expected: str
    actual: str
    stderr: str
    returncode: int | None
    passed: bool


class EvaluationRow(TypedDict):
    """The result of evaluating one generated script file.

    Attributes:
        model: The Ollama model that generated the script.
        script: The path to the generated Python file.
        passed: Whether the script passed every test case.
        tests: The individual test-case results for this script.
    """

    model: str
    script: str
    passed: bool
    tests: list[TestResult]


class Summary(TypedDict):
    """The compact per-model counts displayed in the final summary table.

    Attributes:
        attempts: Number of generated scripts evaluated.
        successes: Number of scripts that passed every test case.
        failures: Number of scripts that failed at least one test case.
        success_rate: Fraction of attempts that succeeded, from 0.0 to 1.0.
    """

    attempts: int
    successes: int
    failures: int
    success_rate: float


class TextGenerationRow(TypedDict):
    """One saved-in-memory response from an Ollama generation.

    ``generated_tokens`` is Ollama's aggregate ``eval_count``. For responses
    generated with thinking mode enabled, that count includes both the
    reasoning trace and final response; Ollama does not currently report
    separate token counts for those fields.
    """

    model: str
    condition: str
    sample_number: int
    prompt: str
    response: str
    thinking: str
    prompt_tokens: int
    generated_tokens: int
    total_duration_seconds: float
    generation_duration_seconds: float
    done_reason: str


class TextEvaluationRow(TextGenerationRow):
    """A text response graded as an integer arithmetic answer."""

    expected_answer: int
    parsed_answer: int | None
    math_correct: bool
    format_correct: bool


class TextSummary(TypedDict):
    """Aggregate correctness and runtime metrics for one text condition."""

    attempts: int
    math_successes: int
    format_successes: int
    math_success_rate: float
    format_success_rate: float
    length_stops: int
    empty_responses: int
    median_generated_tokens: float
    median_duration_seconds: float


def ollama_options(
    *,
    seed: int,
    max_generated_tokens: int,
    temperature: float,
    top_k: int | None = None,
    top_p: float | None = None,
) -> dict[str, int | float]:
    """Build a generation-options dictionary shared by notebook experiments."""

    options: dict[str, int | float] = {
        "seed": seed,
        "num_predict": max_generated_tokens,
        "temperature": temperature,
    }
    if top_k is not None:
        options["top_k"] = top_k
    if top_p is not None:
        options["top_p"] = top_p
    return options


# ---------------------------------------------------------------------------
# Generating model outputs
# ---------------------------------------------------------------------------
#
# Models often wrap requested code in Markdown so it displays nicely in chat.
# Before saving a response as Python, we remove one outer code fence when the
# entire response is a single Python code block. This deterministic cleanup
# removes presentation syntax, not program logic. Prose, incorrect code, and
# other substantive generation mistakes remain untouched for the evaluator.


_OUTER_PYTHON_CODE_FENCE = re.compile(
    r"\A[ \t\r\n]*```[ \t]*(?:python(?:3)?|py)?[ \t]*\r?\n"
    r"(?P<code>.*?)"
    r"\r?\n[ \t]*```[ \t\r\n]*\Z",
    flags=re.IGNORECASE | re.DOTALL,
)
_INNER_CODE_FENCE_LINE = re.compile(r"(?m)^[ \t]*```")


def strip_outer_python_code_fence(response: str) -> str:
    """Remove one Markdown wrapper from an otherwise complete Python response.

    The response is changed only when, aside from surrounding whitespace, it
    consists of one fenced block labeled ``python``, ``python3``, ``py``, or
    left unlabeled. A response containing prose outside the block, an incomplete
    fence, or another fence inside the block is returned unchanged. This keeps
    the step narrowly focused on transport formatting rather than code repair.

    Args:
        response: Raw text returned by the model.

    Returns:
        The code inside one full outer fence, or the original text when the
        conservative wrapper pattern does not match.
    """

    match = _OUTER_PYTHON_CODE_FENCE.fullmatch(response)
    if match is None:
        return response

    code = match.group("code")
    if _INNER_CODE_FENCE_LINE.search(code):
        return response
    return code


def model_directory_name(model_name: str) -> str:
    """Convert an Ollama model name into a directory-friendly string.

    Ollama model tags often contain a colon, such as ``qwen3.5:2b``. Colons are
    meaningful in some shells and filesystems, so we replace them before using
    the model name as a folder name.

    Args:
        model_name: The Ollama model tag, such as ``"qwen3.5:2b"``.

    Returns:
        A version of the model name that is safe to use as a directory name.
    """

    return model_name.replace(":", "-").replace("/", "-")


def output_path_for(output_dir: Path, model_name: str, sample_number: int) -> Path:
    """Build the file path for one generated program.

    This function also creates the model-specific output directory if it does
    not already exist. For example, attempt 1 from ``qwen3.5:2b`` becomes
    ``generated-programs/qwen3.5-2b/001.py``.

    Args:
        output_dir: Root directory where generated programs are stored.
        model_name: The Ollama model that produced the program.
        sample_number: The 1-based attempt number for this model.

    Returns:
        The path where the generated Python program should be written.
    """

    model_dir = output_dir / model_directory_name(model_name)
    model_dir.mkdir(parents=True, exist_ok=True)
    return model_dir / f"{sample_number:03}.py"


def reset_output_dir(output_dir: Path) -> None:
    """Clear the generated-programs directory before a fresh experiment run.

    Deleting files from code should always make us pause. This helper has
    guardrails so it only clears a simple relative directory inside the current
    project. That keeps the notebook convenient without making deletion casual.

    Args:
        output_dir: Relative directory containing generated programs.

    Raises:
        ValueError: If ``output_dir`` is absolute, points at the project root,
            points outside the project, or tries to move upward with ``..``.
    """

    project_dir = Path.cwd().resolve()
    resolved_output_dir = output_dir.resolve()

    if output_dir.is_absolute() or output_dir == Path(".") or ".." in output_dir.parts:
        raise ValueError("output_dir must be a simple relative path")
    if (
        resolved_output_dir == project_dir
        or project_dir not in resolved_output_dir.parents
    ):
        raise ValueError("output_dir must be inside this project directory")

    shutil.rmtree(resolved_output_dir, ignore_errors=True)
    resolved_output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Cleared {output_dir}/")


def generate_one_program(
    model_name: str,
    sample_number: int,
    output_dir: Path,
    prompt: str,
    system_prompt: str,
    think: bool,
    max_generated_tokens: int,
    temperature: float,
    top_k: int | None = None,
    top_p: float | None = None,
) -> Path:
    """Ask one Ollama model for one Python program and save dispatchable code.

    The attempt number is also used as the Ollama seed. This makes attempt 1,
    attempt 2, and so on reproducible when the model and prompt settings stay
    the same.

    Args:
        model_name: Ollama model tag used for generation.
        sample_number: 1-based attempt number, also used as the random seed.
        output_dir: Root directory where generated programs are stored.
        prompt: User prompt describing the program to write.
        system_prompt: System-level instruction sent to Ollama.
        think: Whether to request thinking mode from the model.
        max_generated_tokens: Ollama ``num_predict`` cap for generated tokens.
        temperature: Sampling temperature. Lower values are more predictable.
        top_k: Optional number of highest-scoring candidates kept for sampling.
        top_p: Optional cumulative-probability nucleus kept for sampling.

    Returns:
        Path to the Python file containing the conservatively normalized model
        response.
    """

    response = ollama.generate(
        model=model_name,
        prompt=prompt,
        system=system_prompt,
        think=think,
        options=ollama_options(
            seed=sample_number,
            max_generated_tokens=max_generated_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
        ),
    )
    print(
        f"{model_name} attempt {sample_number} - "
        f"prompt tokens={response.prompt_eval_count}, "
        f"eval tokens={response.eval_count}, "
        f"duration={(response.total_duration or 0.0) / 1_000_000_000:.1f}s"
    )

    code = strip_outer_python_code_fence(response.response or "")
    path = output_path_for(output_dir, model_name, sample_number)
    path.write_text(code)
    return path


def generate_programs(
    model_name: str,
    count: int,
    output_dir: Path,
    prompt: str,
    system_prompt: str,
    think: bool,
    max_generated_tokens: int,
    temperature: float = 0.2,
    top_k: int | None = 20,
    top_p: float | None = 0.8,
) -> list[Path]:
    """Generate many candidate programs from one model.

    This is a simple loop around ``generate_one_program``. The loop is written
    plainly because the notebook is meant to be easy to inspect during a
    workshop.

    Args:
        model_name: Ollama model tag used for generation.
        count: Number of attempts to generate.
        output_dir: Root directory where generated programs are stored.
        prompt: User prompt describing the program to write.
        system_prompt: System-level instruction sent to Ollama.
        think: Whether to request thinking mode from the model.
        max_generated_tokens: Ollama ``num_predict`` cap for generated tokens.
        temperature: Sampling temperature. Lower values are more predictable.
        top_k: Optional number of highest-scoring candidates kept for sampling.
        top_p: Optional cumulative-probability nucleus kept for sampling.

    Returns:
        Paths to the generated Python files.
    """

    paths: list[Path] = []
    for sample_number in range(1, count + 1):
        print(f"{model_name}: generating {sample_number}")
        path = generate_one_program(
            model_name=model_name,
            sample_number=sample_number,
            output_dir=output_dir,
            prompt=prompt,
            system_prompt=system_prompt,
            think=think,
            max_generated_tokens=max_generated_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
        )
        paths.append(path)
    return paths


def generate_text_samples(
    model_name: str,
    condition: str,
    count: int,
    prompt: str,
    system_prompt: str,
    think: bool,
    max_generated_tokens: int,
    temperature: float = 0.7,
    top_k: int = 20,
    top_p: float = 0.8,
) -> list[TextGenerationRow]:
    """Generate text responses while retaining thinking and runtime metrics.

    Unlike program generation, these responses stay in memory because the
    arithmetic and decoding experiments inspect text rather than execute it.
    The 1-based attempt number is also the seed, just as in Session 1.
    """

    rows: list[TextGenerationRow] = []

    for sample_number in range(1, count + 1):
        response = ollama.generate(
            model=model_name,
            prompt=prompt,
            system=system_prompt,
            think=think,
            options=ollama_options(
                seed=sample_number,
                max_generated_tokens=max_generated_tokens,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
            ),
        )

        row: TextGenerationRow = {
            "model": model_name,
            "condition": condition,
            "sample_number": sample_number,
            "prompt": prompt,
            "response": response.response or "",
            "thinking": response.thinking or "",
            "prompt_tokens": response.prompt_eval_count or 0,
            "generated_tokens": response.eval_count or 0,
            "total_duration_seconds": (response.total_duration or 0) / 1_000_000_000,
            "generation_duration_seconds": (response.eval_duration or 0)
            / 1_000_000_000,
            "done_reason": response.done_reason or "",
        }
        rows.append(row)
        print(
            f"{condition} attempt {sample_number} - "
            f"response={row['response']!r}, "
            f"generated tokens={row['generated_tokens']}, "
            f"duration={row['total_duration_seconds']:.1f}s"
        )

    return rows


def parse_arithmetic_answer(
    text: str, answer_format: AnswerFormat
) -> tuple[int | None, bool]:
    """Parse an integer answer using the response convention for a condition.

    Direct and thinking-mode final responses must contain only an integer. For
    a prompted scratchpad, an exact ``Final answer: N`` line satisfies the
    format check.
    The math check can also use a labeled final answer or the final displayed
    equation, which lets us distinguish a mathematically correct parsed answer
    from imperfect format compliance without accepting an arbitrary last
    number.
    """

    stripped = text.strip()
    integer_pattern = r"[-+]?(?:\d{1,3}(?:,\d{3})+|\d+)"

    if answer_format == "integer_only":
        match = re.fullmatch(f"({integer_pattern})", stripped)
        if match is not None:
            return int(match.group(1).replace(",", "")), True
    else:
        final_line = stripped.splitlines()[-1].strip() if stripped else ""
        exact_match = re.fullmatch(f"Final answer:\\s*({integer_pattern})", final_line)
        if exact_match is not None:
            return int(exact_match.group(1).replace(",", "")), True

    labeled_answers = re.findall(
        rf"(?i)\b(?:final\s+)?(?:answer|result)\s*(?:is|:)\s*"
        rf"<?({integer_pattern})>?",
        stripped,
    )
    if labeled_answers:
        return int(labeled_answers[-1].replace(",", "")), False

    displayed_results = re.findall(rf"=\s*({integer_pattern})(?![\d,])", stripped)
    if displayed_results:
        return int(displayed_results[-1].replace(",", "")), False

    return None, False


def evaluate_arithmetic_samples(
    samples: list[TextGenerationRow],
    expected_answer: int,
    answer_format: AnswerFormat,
) -> list[TextEvaluationRow]:
    """Grade text generations for mathematical and requested-format correctness."""

    rows: list[TextEvaluationRow] = []
    for sample in samples:
        parsed_answer, format_correct = parse_arithmetic_answer(
            sample["response"], answer_format
        )
        rows.append(
            {
                **sample,
                "expected_answer": expected_answer,
                "parsed_answer": parsed_answer,
                "math_correct": parsed_answer == expected_answer,
                "format_correct": format_correct,
            }
        )
    return rows


def summarize_text_results(rows: list[TextEvaluationRow]) -> TextSummary:
    """Summarize correctness, generated tokens, and end-to-end latency."""

    attempts = len(rows)
    math_successes = sum(row["math_correct"] for row in rows)
    format_successes = sum(row["format_correct"] for row in rows)

    return {
        "attempts": attempts,
        "math_successes": math_successes,
        "format_successes": format_successes,
        "math_success_rate": math_successes / attempts if attempts else 0,
        "format_success_rate": format_successes / attempts if attempts else 0,
        "length_stops": sum(row["done_reason"] == "length" for row in rows),
        "empty_responses": sum(not row["response"].strip() for row in rows),
        "median_generated_tokens": (
            median(row["generated_tokens"] for row in rows) if rows else 0
        ),
        "median_duration_seconds": (
            median(row["total_duration_seconds"] for row in rows) if rows else 0
        ),
    }


def show_text_samples(
    rows: list[TextGenerationRow], limit: int = 5, include_thinking: bool = False
) -> None:
    """Print representative responses and, optionally, reasoning traces."""

    for row in rows[:limit]:
        print(f"\n{row['condition']} attempt {row['sample_number']}")
        if include_thinking:
            print("REASONING TRACE:")
            print(row["thinking"] or "(none)")
        print("RESPONSE:")
        print(row["response"] or "(empty)")


# ---------------------------------------------------------------------------
# Running generated programs
# ---------------------------------------------------------------------------
#
# The generated files are untrusted code. Instead of running them directly in
# this notebook's Python process, we run each one in a disposable Docker
# container. The container has no network, can read only the generated script,
# drops Linux capabilities, and receives conservative resource limits. The host
# also caps captured output and force-removes the named container after each run.


def _run_with_bounded_output(
    command: list[str], timeout_seconds: int, max_output_bytes: int
) -> subprocess.CompletedProcess[str]:
    """Run a process while retaining at most ``max_output_bytes`` of output."""

    process = subprocess.Popen(
        command,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    assert process.stdout is not None
    assert process.stderr is not None

    selector = selectors.DefaultSelector()
    selector.register(process.stdout, selectors.EVENT_READ, "stdout")
    selector.register(process.stderr, selectors.EVENT_READ, "stderr")
    buffers = {"stdout": bytearray(), "stderr": bytearray()}
    deadline = time.monotonic() + timeout_seconds
    stop_reason: Literal["timeout", "output"] | None = None

    try:
        while selector.get_map():
            remaining_time = deadline - time.monotonic()
            if remaining_time <= 0:
                stop_reason = "timeout"
                break

            for key, _ in selector.select(timeout=min(0.1, remaining_time)):
                # key.fileobj may be an int file descriptor or a file-like object.
                fileobj = key.fileobj
                fd = fileobj if isinstance(fileobj, int) else fileobj.fileno()
                chunk = os.read(fd, 8192)
                if not chunk:
                    selector.unregister(key.fileobj)
                    continue

                captured = sum(len(buffer) for buffer in buffers.values())
                remaining_bytes = max_output_bytes - captured
                buffers[key.data].extend(chunk[:remaining_bytes])
                if len(chunk) > remaining_bytes:
                    stop_reason = "output"
                    break

            if stop_reason is not None:
                break

        if stop_reason is not None:
            process.kill()
        process.wait(timeout=2)
    finally:
        if process.poll() is None:
            process.kill()
            process.wait()
        selector.close()
        process.stdout.close()
        process.stderr.close()

    stdout = buffers["stdout"].decode("utf-8", errors="replace")
    stderr = buffers["stderr"].decode("utf-8", errors="replace")

    if stop_reason == "timeout":
        raise subprocess.TimeoutExpired(
            command, timeout_seconds, output=stdout, stderr=stderr
        )
    if stop_reason == "output":
        stderr += f"\nOutput exceeded the {max_output_bytes}-byte safety limit."
        return subprocess.CompletedProcess(command, -1, stdout, stderr)

    return subprocess.CompletedProcess(command, process.returncode or 0, stdout, stderr)


def run_in_python_container(
    script_path: str | Path,
    args: list[str],
    docker_image: str,
    timeout_seconds: int = 5,
    project_dir: Path | None = None,
    max_output_bytes: int = 64 * 1024,
) -> subprocess.CompletedProcess[str]:
    """Run one generated Python script inside a Docker Python container.

    Standard input is connected to ``subprocess.DEVNULL``. That means a program
    that calls ``input()`` receives EOF instead of blocking forever waiting for
    a person to type something.

    Args:
        script_path: Path to the generated Python script.
        args: Command-line arguments to pass to the script.
        docker_image: Python Docker image, such as ``"python:3.12"``.
        timeout_seconds: Maximum number of seconds before the run is stopped.
        project_dir: Project boundary used to validate ``script_path``.
            Defaults to the current working directory. The directory itself is
            not exposed to the container.
        max_output_bytes: Combined stdout and stderr retained from the process.

    Returns:
        The completed process, including stdout, stderr, and return code.

    Raises:
        ValueError: If ``script_path`` is not inside ``project_dir``.
        subprocess.TimeoutExpired: If Docker does not finish before the timeout.
    """

    project_dir = Path.cwd().resolve() if project_dir is None else project_dir.resolve()
    script_path = Path(script_path).resolve()
    script_path.relative_to(project_dir)
    container_name = f"ai-workshop-eval-{uuid.uuid4().hex}"

    command = [
        "docker",
        "run",
        "--name",
        container_name,
        "--rm",
        "--network",
        "none",
        "--read-only",
        "--cap-drop",
        "ALL",
        "--security-opt",
        "no-new-privileges",
        "--pids-limit",
        "64",
        "--memory",
        "256m",
        "--cpus",
        "1",
        "--tmpfs",
        "/tmp:rw,noexec,nosuid,size=16m",
        "--user",
        "65534:65534",
        "-v",
        f"{script_path}:/app/program.py:ro",
        "-w",
        "/app",
        docker_image,
        "python",
        "-B",
        "/app/program.py",
        *args,
    ]

    try:
        return _run_with_bounded_output(
            command,
            timeout_seconds=timeout_seconds,
            max_output_bytes=max_output_bytes,
        )
    finally:
        try:
            subprocess.run(
                ["docker", "rm", "-f", container_name],
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=5,
                check=False,
            )
        except (OSError, subprocess.TimeoutExpired):
            pass


def test_program(
    script_path: str | Path,
    test_cases: list[TestCase],
    docker_image: str,
    timeout_seconds: int = 5,
    project_dir: Path | None = None,
) -> list[TestResult]:
    """Run all test cases against one generated program.

    A program passes a test only when it exits successfully and prints exactly
    the expected text. Extra words, prompts, tracebacks, and missing output all
    count as failures.

    Args:
        script_path: Path to the generated Python script.
        test_cases: Pairs of command-line arguments and expected output.
        docker_image: Python Docker image used to run the script.
        timeout_seconds: Maximum seconds allowed for each test case.
        project_dir: Project boundary used to validate ``script_path``.

    Returns:
        One ``TestResult`` dictionary for each test case.
    """

    test_results: list[TestResult] = []

    for args, expected_stdout in test_cases:
        try:
            completed = run_in_python_container(
                script_path=script_path,
                args=args,
                docker_image=docker_image,
                timeout_seconds=timeout_seconds,
                project_dir=project_dir,
            )
            # Permit the one conventional newline added by ``print`` while
            # preserving every other leading or trailing character.
            actual_stdout = completed.stdout.removesuffix("\n")
            passed = completed.returncode == 0 and actual_stdout == expected_stdout

            test_results.append(
                {
                    "args": args,
                    "expected": expected_stdout,
                    "actual": actual_stdout,
                    "stderr": completed.stderr.strip(),
                    "returncode": completed.returncode,
                    "passed": passed,
                }
            )
        except subprocess.TimeoutExpired:
            test_results.append(
                {
                    "args": args,
                    "expected": expected_stdout,
                    "actual": "",
                    "stderr": "Timed out",
                    "returncode": None,
                    "passed": False,
                }
            )

    return test_results


# ---------------------------------------------------------------------------
# Evaluating and summarizing results
# ---------------------------------------------------------------------------
#
# Once each generated file has test results, we collapse those details into a
# per-model summary. The details are still kept around so you can inspect
# failures and learn what kinds of mistakes the models made.


def evaluate_model(
    model_name: str,
    output_dir: Path,
    test_cases: list[TestCase],
    docker_image: str,
    timeout_seconds: int = 5,
    project_dir: Path | None = None,
) -> list[EvaluationRow]:
    """Evaluate every generated script for one model.

    The model name determines which output directory to read. For example,
    ``qwen3.5:2b`` maps to ``generated-programs/qwen3.5-2b``.

    Args:
        model_name: Ollama model tag whose generated files should be evaluated.
        output_dir: Root directory containing generated programs.
        test_cases: Pairs of command-line arguments and expected output.
        docker_image: Python Docker image used to run each script.
        timeout_seconds: Maximum seconds allowed for each test case.
        project_dir: Project directory to mount into Docker. Defaults to the
            current working directory.

    Returns:
        One evaluation row for each generated script file.
    """

    model_dir = output_dir / model_directory_name(model_name)
    script_paths = sorted(model_dir.glob("*.py"))
    if not script_paths:
        raise FileNotFoundError(f"No generated programs found in {model_dir}")
    rows: list[EvaluationRow] = []

    for script_path in script_paths:
        test_results = test_program(
            script_path=script_path,
            test_cases=test_cases,
            docker_image=docker_image,
            timeout_seconds=timeout_seconds,
            project_dir=project_dir,
        )
        passed = all(result["passed"] for result in test_results)
        rows.append(
            {
                "model": model_name,
                "script": str(script_path),
                "passed": passed,
                "tests": test_results,
            }
        )

    return rows


def summarize_results(rows: list[EvaluationRow]) -> Summary:
    """Count successes and failures for one model.

    Args:
        rows: Per-script evaluation rows for a single model.

    Returns:
        A compact summary with attempts, successes, failures, and success rate.
    """

    successes = sum(row["passed"] for row in rows)
    failures = len(rows) - successes
    success_rate = successes / len(rows) if rows else 0

    return {
        "attempts": len(rows),
        "successes": successes,
        "failures": failures,
        "success_rate": success_rate,
    }


# ---------------------------------------------------------------------------
# Display helpers for the notebook
# ---------------------------------------------------------------------------
#
# These functions are small, but keeping them here means the notebook can focus
# on the evaluation conversation rather than table formatting details.


def show_summary_table(summary: dict[str, Summary]) -> None:
    """Display model summaries as a Markdown table in Jupyter.

    Args:
        summary: Mapping from model name to that model's summary counts.
    """

    lines = [
        "| Model | Attempts | Successes | Failures | Success rate |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]

    for model_name, result in summary.items():
        percent = result["success_rate"] * 100
        lines.append(
            f"| `{model_name}` | {result['attempts']} | {result['successes']} | "
            f"{result['failures']} | {percent:.1f}% |"
        )

    display(Markdown("\n".join(lines)))


def show_text_summary_table(summary: dict[str, TextSummary]) -> None:
    """Display arithmetic-response summaries as a Markdown table in Jupyter."""

    lines = [
        "| Condition | Attempts | Math correct | Format correct | "
        "Length stops | Empty | Median tokens | Median seconds |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]

    for condition, result in summary.items():
        math_percent = result["math_success_rate"] * 100
        format_percent = result["format_success_rate"] * 100
        lines.append(
            f"| {condition} | {result['attempts']} | "
            f"{result['math_successes']} ({math_percent:.1f}%) | "
            f"{result['format_successes']} ({format_percent:.1f}%) | "
            f"{result['length_stops']} | {result['empty_responses']} | "
            f"{result['median_generated_tokens']:.0f} | "
            f"{result['median_duration_seconds']:.2f} |"
        )

    display(Markdown("\n".join(lines)))


def first_failure(row: EvaluationRow) -> TestResult:
    """Return the first failing test result from one evaluation row.

    Args:
        row: Per-script evaluation row that is expected to contain a failure.

    Returns:
        The first failed test case in ``row``.

    Raises:
        ValueError: If every test in ``row`` passed.
    """

    for test in row["tests"]:
        if not test["passed"]:
            return test

    raise ValueError("row does not contain a failing test")
