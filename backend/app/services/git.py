import re
import subprocess
from pathlib import Path
from typing import Any

from app.core.config import settings
from app.services.workspace import WorkspaceError, resolve_path, workspace_root

MAX_GIT_OUTPUT = 200_000
MAX_COMMIT_MESSAGE = 200
BRANCH_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._/-]{0,99}$")


class GitError(RuntimeError):
    """Base error for controlled Git operations."""


class GitDisabledError(GitError):
    pass


class GitValidationError(GitError):
    pass


class GitCommandError(GitError):
    pass


def _root() -> Path:
    if not settings.git_enabled:
        raise GitDisabledError("Git tools are disabled")
    if not settings.workspace_enabled:
        raise GitDisabledError("Workspace must be enabled for Git tools")
    try:
        root = workspace_root()
    except WorkspaceError as exc:
        raise GitDisabledError("A valid workspace is required for Git tools") from exc
    if not (root / ".git").is_dir() and not (root / ".git").is_file():
        raise GitValidationError("Workspace is not a Git repository")
    return root


def _run(args: list[str], root: Path) -> subprocess.CompletedProcess[str]:
    if not args or args[0] != "git":
        raise GitValidationError("Invalid Git command")
    try:
        result = subprocess.run(
            args,
            cwd=root,
            shell=False,
            check=False,
            capture_output=True,
            text=True,
            timeout=15,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise GitCommandError("Git operation is unavailable") from exc
    if result.returncode != 0:
        raise GitCommandError("Git operation failed")
    return result


def _trim(value: str) -> str:
    return value[:MAX_GIT_OUTPUT]


def _status_entries(output: str) -> list[dict[str, str]]:
    entries = []
    for line in output.splitlines():
        if len(line) < 4:
            continue
        entries.append({"index": line[0], "worktree": line[1], "path": line[3:]})
    return entries


def git_status() -> dict[str, Any]:
    root = _root()
    branch = _run(["git", "branch", "--show-current"], root).stdout.strip()
    result = _run(["git", "status", "--porcelain=v1", "--untracked-files=all"], root)
    entries = _status_entries(_trim(result.stdout))
    return {"branch": branch, "clean": not entries, "changes": entries}


def git_diff(staged: bool = False) -> dict[str, str | bool]:
    root = _root()
    args = ["git", "diff"]
    if staged:
        args.append("--cached")
    result = _run(args, root)
    diff = _trim(result.stdout)
    return {"staged": staged, "diff": diff, "truncated": len(result.stdout) > len(diff)}


def git_create_branch(branch_name: str) -> dict[str, str]:
    root = _root()
    name = branch_name.strip() if isinstance(branch_name, str) else ""
    if not BRANCH_PATTERN.fullmatch(name) or name in {"HEAD", ".", ".."}:
        raise GitValidationError("Invalid branch name")
    if ".." in name or name.endswith(("/", ".lock")) or "@{" in name:
        raise GitValidationError("Invalid branch name")
    result = _run(["git", "switch", "-c", name], root)
    return {"branch": name, "output": _trim(result.stdout)}


def _validate_commit_message(message: str) -> str:
    if not isinstance(message, str):
        raise GitValidationError("Commit message must be text")
    normalized = message.strip()
    if not normalized or len(normalized) > MAX_COMMIT_MESSAGE:
        raise GitValidationError("Commit message is invalid")
    if any(ord(char) < 32 and char not in "\n\t" for char in normalized):
        raise GitValidationError("Commit message contains control characters")
    return normalized


def git_commit(
    message: str,
    file_paths: list[str],
    confirmation: bool,
    proposal_id: str | None = None,
) -> dict[str, Any]:
    if confirmation is not True:
        raise GitValidationError("Explicit confirmation=true is required")
    normalized_message = _validate_commit_message(message)
    if not isinstance(file_paths, list) or not file_paths or len(file_paths) > 50:
        raise GitValidationError("Explicit file paths are required")
    root = _root()
    status = git_status()
    if any(change["index"] == "D" or change["worktree"] == "D" for change in status["changes"]):
        raise GitValidationError("Commits containing deleted files are not allowed")

    relative_paths: list[str] = []
    for file_path in file_paths:
        try:
            path = resolve_path(file_path)
        except WorkspaceError as exc:
            raise GitValidationError("Commit path is outside the workspace") from exc
        if not path.exists() or not path.is_file():
            raise GitValidationError("Commit paths must refer to existing files")
        relative_paths.append(str(path.relative_to(root)))

    _run(["git", "add", "--", *relative_paths], root)
    _run(["git", "commit", "-m", normalized_message], root)
    commit_hash = _run(["git", "rev-parse", "HEAD"], root).stdout.strip()
    return {
        "commit_hash": commit_hash,
        "file_paths": relative_paths,
        "proposal_id": proposal_id,
        "message": normalized_message,
        "status": "committed",
    }
