"""Helper functions for the local LLM evaluation notebook.

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

from pathlib import Path
import shutil
import subprocess
from typing import TypedDict

from IPython.display import Markdown, display
import ollama


# ---------------------------------------------------------------------------
# Data shapes
# ---------------------------------------------------------------------------
#
# Python dictionaries are flexible, but flexible data can become hard to
# reason about. TypedDict lets us keep the friendliness of dictionaries while
# documenting which keys each dictionary should contain.

TestCase = tuple[list[str], str]
"""A command-line argument list paired with the expected standard output."""


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


# ---------------------------------------------------------------------------
# Generating model outputs
# ---------------------------------------------------------------------------
#
# The generation phase asks a model for code and saves that code exactly as the
# model wrote it. We do not clean up Markdown fences or prose, because those
# mistakes are part of what the evaluation is meant to measure.


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
) -> Path:
    """Ask one Ollama model for one Python program and save the response.

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

    Returns:
        Path to the Python file containing the model response.
    """

    response = ollama.generate(
        model=model_name,
        prompt=prompt,
        system=system_prompt,
        think=think,
        options={
            "seed": sample_number,
            "num_predict": max_generated_tokens,
            "temperature": temperature,
        },
    )
    print(
        f"{model_name} attempt {sample_number} - "
        f"prompt tokens={response.prompt_eval_count}, "
        f"eval tokens={response.eval_count}, "
        f"duration={(response.total_duration or 0.0) / 1_000_000_000:.1f}s"
    )

    code = response.response or ""
    path = output_path_for(output_dir, model_name, sample_number)
    path.write_text(code.strip() + "\n")
    return path


def generate_programs(
    model_name: str,
    count: int,
    output_dir: Path,
    prompt: str,
    system_prompt: str,
    think: bool,
    max_generated_tokens: int,
    temperature: float,
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
        )
        paths.append(path)
    return paths


# ---------------------------------------------------------------------------
# Running generated programs
# ---------------------------------------------------------------------------
#
# The generated files are untrusted code. Instead of running them directly in
# this notebook's Python process, we run each one in a disposable Docker
# container. The container still has access to the project directory, but it
# gives us a clean Python version and keeps each run separate.


def run_in_python_container(
    script_path: str | Path,
    args: list[str],
    docker_image: str,
    timeout_seconds: int = 5,
    project_dir: Path | None = None,
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
        project_dir: Project directory to mount into Docker. Defaults to the
            current working directory.

    Returns:
        The completed process, including stdout, stderr, and return code.

    Raises:
        ValueError: If ``script_path`` is not inside ``project_dir``.
        subprocess.TimeoutExpired: If Docker does not finish before the timeout.
    """

    project_dir = Path.cwd().resolve() if project_dir is None else project_dir.resolve()
    script_path = Path(script_path).resolve()
    script_inside_container = script_path.relative_to(project_dir)

    command = [
        "docker",
        "run",
        "--rm",
        "-v",
        f"{project_dir}:/app",
        "-w",
        "/app",
        docker_image,
        "python",
        str(script_inside_container),
        *args,
    ]

    return subprocess.run(
        command,
        stdin=subprocess.DEVNULL,
        text=True,
        capture_output=True,
        timeout=timeout_seconds,
    )


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
        project_dir: Project directory to mount into Docker. Defaults to the
            current working directory.

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
            actual_stdout = completed.stdout.strip()
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
