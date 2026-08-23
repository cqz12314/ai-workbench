import re
import subprocess
from pathlib import Path
from typing import Any

from app.services.workspace import WorkspaceError, workspace_root

ALLOWED_TESTS = {
    "pytest": ["pytest"],
    "ruff check": ["ruff", "check", "."],
    "npm run build": ["npm", "run", "build"],
    "npm run lint": ["npm", "run", "lint"],
}
MAX_OUTPUT = 50_000
TEST_TIMEOUT = 120
SECRET_PATTERN = re.compile(r"(?i)(token|secret|password|api[_-]?key)\s*[=:]\s*[^\s]+")


class TestRunnerError(RuntimeError):
    pass


def _sanitize(value: str) -> str:
    return SECRET_PATTERN.sub(r"\1=[REDACTED]", value)[:MAX_OUTPUT]


def run_test(test_name: str) -> dict[str, Any]:
    command = ALLOWED_TESTS.get(test_name)
    if command is None:
        raise TestRunnerError("Test command is not allowlisted")
    try:
        root: Path = workspace_root()
    except WorkspaceError as exc:
        raise TestRunnerError("A valid workspace is required") from exc
    try:
        result = subprocess.run(
            command,
            cwd=root,
            shell=False,
            check=False,
            capture_output=True,
            text=True,
            timeout=TEST_TIMEOUT,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise TestRunnerError("Test runner is unavailable or timed out") from exc
    return {
        "test": test_name,
        "command": command[0:3],
        "returncode": result.returncode,
        "passed": result.returncode == 0,
        "stdout": _sanitize(result.stdout),
        "stderr": _sanitize(result.stderr),
    }
