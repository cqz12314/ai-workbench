import re
from typing import Any

import httpx

from app.core.config import settings
from app.services.github import (
    GITHUB_TIMEOUT,
    GitHubTokenMissingError,
    _headers,
    _repo_path,
    _url,
)

SHA_PATTERN = re.compile(r"^[0-9a-fA-F]{40}$")
BRANCH_COMPONENT_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._/-]{0,99}$")
MAX_TITLE_LENGTH = 256
MAX_DESCRIPTION_LENGTH = 20_000


class GitHubWriteError(RuntimeError):
    """Base error for controlled GitHub write operations."""


class GitHubWriteDisabledError(GitHubWriteError):
    pass


class GitHubWriteValidationError(GitHubWriteError):
    pass


class GitHubWriteAPIError(GitHubWriteError):
    pass


def _write_headers() -> dict[str, str]:
    if not settings.github_enabled or not settings.github_write_enabled:
        raise GitHubWriteDisabledError("GitHub write tools are disabled")
    try:
        return _headers()
    except GitHubTokenMissingError as exc:
        raise GitHubWriteError("GitHub token is not configured") from exc


def _validate_repository(owner: str, repo: str) -> tuple[str, str]:
    if not isinstance(owner, str) or not isinstance(repo, str):
        raise GitHubWriteValidationError("Repository is invalid")
    owner = owner.strip()
    repo = repo.strip()
    if not re.fullmatch(r"[A-Za-z0-9_.-]{1,100}", owner) or not re.fullmatch(
        r"[A-Za-z0-9_.-]{1,100}", repo
    ):
        raise GitHubWriteValidationError("Repository is invalid")
    return owner, repo


def _validate_feature_branch(feature_name: str) -> str:
    if not isinstance(feature_name, str):
        raise GitHubWriteValidationError("Branch name is invalid")
    component = feature_name.strip()
    if component in {"main", "master"}:
        raise GitHubWriteValidationError("Protected branches cannot be written")
    branch = component if component.startswith("ai/") else f"ai/{component}"
    if (
        not BRANCH_COMPONENT_PATTERN.fullmatch(branch)
        or branch in {"main", "master"}
        or ".." in branch
        or branch.endswith(("/", ".lock"))
        or "@{" in branch
    ):
        raise GitHubWriteValidationError("Branch must be a valid ai/* branch")
    return branch


def _validate_source_branch(branch: str) -> str:
    if not isinstance(branch, str):
        raise GitHubWriteValidationError("Source branch is invalid")
    normalized = branch.strip()
    if not normalized.startswith("ai/") or not BRANCH_COMPONENT_PATTERN.fullmatch(normalized):
        raise GitHubWriteValidationError("Source branch must use the ai/* namespace")
    if normalized in {"main", "master"}:
        raise GitHubWriteValidationError("Protected branches cannot be written")
    return normalized


def _validate_sha(commit_sha: str) -> str:
    if not isinstance(commit_sha, str) or not SHA_PATTERN.fullmatch(commit_sha.strip()):
        raise GitHubWriteValidationError("Commit SHA is invalid")
    return commit_sha.strip()


async def _request(method: str, path: str, payload: dict[str, Any] | None = None) -> Any:
    headers = _write_headers()
    try:
        async with httpx.AsyncClient(timeout=GITHUB_TIMEOUT) as client:
            response = await client.request(
                method,
                _url(path),
                headers=headers,
                json=payload,
            )
    except (httpx.TimeoutException, httpx.TransportError) as exc:
        raise GitHubWriteAPIError("GitHub API is unavailable") from exc
    if not response.is_success:
        messages = {
            401: "GitHub authentication failed",
            403: "GitHub access was forbidden or rate limited",
            409: "GitHub resource conflict",
            422: "GitHub rejected the write request",
            429: "GitHub rate limit exceeded",
        }
        raise GitHubWriteAPIError(messages.get(response.status_code, "GitHub write request failed"))
    if not response.content:
        return {}
    try:
        return response.json()
    except ValueError as exc:
        raise GitHubWriteAPIError("GitHub returned invalid write response") from exc


async def _ensure_repository(owner: str, repo: str) -> dict[str, Any]:
    payload = await _request("GET", _repo_path(owner, repo))
    if not isinstance(payload, dict) or not payload.get("full_name"):
        raise GitHubWriteAPIError("GitHub repository could not be verified")
    return payload


async def _ensure_commit(owner: str, repo: str, commit_sha: str) -> None:
    payload = await _request("GET", _repo_path(owner, repo, f"/git/commits/{commit_sha}"))
    if not isinstance(payload, dict) or payload.get("sha", "").lower() != commit_sha.lower():
        raise GitHubWriteAPIError("Commit object does not exist on GitHub")


async def create_branch(
    owner: str, repo: str, feature_name: str, base_sha: str, confirmation: bool
) -> dict[str, Any]:
    if confirmation is not True:
        raise GitHubWriteValidationError("Explicit confirmation=true is required")
    owner, repo = _validate_repository(owner, repo)
    branch = _validate_feature_branch(feature_name)
    base_sha = _validate_sha(base_sha)
    await _ensure_repository(owner, repo)
    await _ensure_commit(owner, repo, base_sha)
    payload = await _request(
        "POST",
        _repo_path(owner, repo, "/git/refs"),
        {"ref": f"refs/heads/{branch}", "sha": base_sha},
    )
    return {
        "repository": f"{owner}/{repo}",
        "branch": branch,
        "base_sha": base_sha,
        "result": payload,
    }


async def push_branch(
    owner: str,
    repo: str,
    branch: str,
    commit_sha: str,
    confirmation: bool,
) -> dict[str, Any]:
    if confirmation is not True:
        raise GitHubWriteValidationError("Explicit confirmation=true is required")
    owner, repo = _validate_repository(owner, repo)
    branch = _validate_source_branch(branch)
    commit_sha = _validate_sha(commit_sha)
    await _ensure_repository(owner, repo)
    await _ensure_commit(owner, repo, commit_sha)
    payload = await _request(
        "PATCH",
        _repo_path(owner, repo, f"/git/refs/heads/{branch}"),
        {"sha": commit_sha, "force": False},
    )
    return {
        "repository": f"{owner}/{repo}",
        "branch": branch,
        "commit_sha": commit_sha,
        "force": False,
        "result": payload,
    }


async def create_pull_request(
    owner: str,
    repo: str,
    title: str,
    description: str,
    source_branch: str,
    target_branch: str = "main",
    confirmation: bool = False,
) -> dict[str, Any]:
    if confirmation is not True:
        raise GitHubWriteValidationError("Explicit confirmation=true is required")
    owner, repo = _validate_repository(owner, repo)
    if not isinstance(title, str) or not 1 <= len(title.strip()) <= MAX_TITLE_LENGTH:
        raise GitHubWriteValidationError("Pull request title is invalid")
    if not isinstance(description, str) or len(description) > MAX_DESCRIPTION_LENGTH:
        raise GitHubWriteValidationError("Pull request description is invalid")
    source_branch = _validate_source_branch(source_branch)
    target_branch = target_branch.strip() if isinstance(target_branch, str) else ""
    if not target_branch or ".." in target_branch or target_branch.startswith("-"):
        raise GitHubWriteValidationError("Target branch is invalid")
    await _ensure_repository(owner, repo)
    source_ref = await _request(
        "GET", _repo_path(owner, repo, f"/git/ref/heads/{source_branch}")
    )
    if not isinstance(source_ref, dict) or not source_ref.get("object"):
        raise GitHubWriteAPIError("Source branch does not exist on GitHub")
    payload = await _request(
        "POST",
        _repo_path(owner, repo, "/pulls"),
        {
            "title": title.strip(),
            "body": description,
            "head": source_branch,
            "base": target_branch,
        },
    )
    return {
        "repository": f"{owner}/{repo}",
        "source_branch": source_branch,
        "target_branch": target_branch,
        "pull_request": payload,
    }
