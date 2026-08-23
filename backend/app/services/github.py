import base64
from typing import Any
from urllib.parse import quote, urlencode

import httpx

from app.core.config import settings

MAX_FILE_BYTES = 1_048_576
MAX_RESULTS = 20
MAX_TREE_ENTRIES = 2_000
GITHUB_TIMEOUT = 15.0


class GitHubError(RuntimeError):
    """Base error for safe GitHub read-only operations."""


class GitHubDisabledError(GitHubError):
    pass


class GitHubTokenMissingError(GitHubError):
    pass


class GitHubAPIError(GitHubError):
    pass


class GitHubFileTooLargeError(GitHubError):
    pass


class GitHubBinaryFileError(GitHubError):
    pass


def _headers() -> dict[str, str]:
    if not settings.github_enabled:
        raise GitHubDisabledError("GitHub tools are disabled")
    if settings.github_token is None or not settings.github_token.get_secret_value():
        raise GitHubTokenMissingError("GitHub token is not configured")
    return {
        "Accept": "application/vnd.github+json",
        "Authorization": f"Bearer {settings.github_token.get_secret_value()}",
        "X-GitHub-Api-Version": "2022-11-28",
    }


def _url(path: str, params: dict[str, Any] | None = None) -> str:
    url = f"{settings.github_api_base.rstrip('/')}/{path.lstrip('/')}"
    if params:
        url = f"{url}?{urlencode(params)}"
    return url


def _raise_for_github_error(response: httpx.Response) -> None:
    if response.is_success:
        return
    if response.status_code == 401:
        raise GitHubAPIError("GitHub authentication failed")
    if response.status_code == 403:
        raise GitHubAPIError("GitHub access was forbidden or rate limited")
    if response.status_code == 429:
        raise GitHubAPIError("GitHub rate limit exceeded")
    raise GitHubAPIError("GitHub API request failed")


async def _get(path: str, params: dict[str, Any] | None = None) -> Any:
    headers = _headers()
    try:
        async with httpx.AsyncClient(timeout=GITHUB_TIMEOUT) as client:
            response = await client.get(_url(path, params), headers=headers)
    except (httpx.TimeoutException, httpx.TransportError) as exc:
        raise GitHubAPIError("GitHub API is unavailable") from exc
    _raise_for_github_error(response)
    try:
        return response.json()
    except ValueError as exc:
        raise GitHubAPIError("GitHub API returned invalid data") from exc


def _repo_path(owner: str, repo: str, suffix: str = "") -> str:
    safe_owner = quote(owner.strip(), safe="")
    safe_repo = quote(repo.strip(), safe="")
    return f"repos/{safe_owner}/{safe_repo}{suffix}"


async def list_repositories(per_page: int = 20, page: int = 1) -> dict[str, Any]:
    payload = await _get(
        "user/repos",
        {"per_page": per_page, "page": page, "sort": "updated", "direction": "desc"},
    )
    if not isinstance(payload, list):
        raise GitHubAPIError("GitHub returned an invalid repository list")
    return {
        "repositories": [
            {
                "name": item.get("name"),
                "full_name": item.get("full_name"),
                "private": item.get("private"),
                "default_branch": item.get("default_branch"),
                "html_url": item.get("html_url"),
                "description": item.get("description"),
            }
            for item in payload[:MAX_RESULTS]
            if isinstance(item, dict)
        ]
    }


async def get_file(owner: str, repo: str, path: str, ref: str | None = None) -> dict[str, Any]:
    params = {"ref": ref} if ref else None
    file_path = quote(path.strip(), safe="/")
    payload = await _get(_repo_path(owner, repo, f"/contents/{file_path}"), params)
    if not isinstance(payload, dict) or payload.get("type") != "file":
        raise GitHubAPIError("GitHub path is not a file")
    size = payload.get("size")
    if isinstance(size, int) and size > MAX_FILE_BYTES:
        raise GitHubFileTooLargeError("GitHub file exceeds the size limit")
    encoded = payload.get("content")
    if payload.get("encoding") != "base64" or not isinstance(encoded, str):
        raise GitHubAPIError("GitHub file content is unavailable")
    try:
        raw = base64.b64decode(encoded, validate=False)
        content = raw.decode("utf-8")
    except (ValueError, UnicodeDecodeError) as exc:
        raise GitHubBinaryFileError("Binary GitHub files are not supported") from exc
    if len(raw) > MAX_FILE_BYTES or len(content.encode("utf-8")) > MAX_FILE_BYTES:
        raise GitHubFileTooLargeError("GitHub file exceeds the size limit")
    if "\x00" in content:
        raise GitHubBinaryFileError("Binary GitHub files are not supported")
    return {
        "repository": f"{owner}/{repo}",
        "path": payload.get("path", path),
        "sha": payload.get("sha"),
        "size": len(raw),
        "content": content,
    }


async def get_repository_tree(
    owner: str, repo: str, ref: str | None = None
) -> dict[str, Any]:
    """Fetch a bounded, read-only repository tree from GitHub."""
    tree_ref = ref.strip() if ref else "HEAD"
    payload = await _get(
        _repo_path(owner, repo, f"/git/trees/{quote(tree_ref, safe='')}"),
        {"recursive": "1"},
    )
    if not isinstance(payload, dict) or not isinstance(payload.get("tree"), list):
        raise GitHubAPIError("GitHub returned invalid repository tree data")
    if payload.get("truncated") is True:
        raise GitHubAPIError("GitHub repository tree was truncated")
    entries = [
        {
            "path": item.get("path"),
            "type": item.get("type"),
            "size": item.get("size", 0),
            "sha": item.get("sha"),
        }
        for item in payload["tree"][:MAX_TREE_ENTRIES]
        if isinstance(item, dict) and isinstance(item.get("path"), str)
    ]
    if len(payload["tree"]) > MAX_TREE_ENTRIES:
        raise GitHubAPIError("GitHub repository tree exceeds the size limit")
    return {"sha": payload.get("sha"), "truncated": False, "tree": entries}


async def search_code(query: str, per_page: int = 20, page: int = 1) -> dict[str, Any]:
    payload = await _get(
        "search/code", {"q": query.strip(), "per_page": per_page, "page": page}
    )
    if not isinstance(payload, dict) or not isinstance(payload.get("items"), list):
        raise GitHubAPIError("GitHub returned invalid code search data")
    return {
        "total_count": payload.get("total_count", 0),
        "items": [
            {
                "name": item.get("name"),
                "path": item.get("path"),
                "repository": (item.get("repository") or {}).get("full_name"),
                "html_url": item.get("html_url"),
            }
            for item in payload["items"][:MAX_RESULTS]
            if isinstance(item, dict)
        ],
    }


async def list_issues(
    owner: str,
    repo: str,
    state: str = "open",
    per_page: int = 20,
    page: int = 1,
) -> dict[str, Any]:
    payload = await _get(
        _repo_path(owner, repo, "/issues"),
        {"state": state, "per_page": per_page, "page": page},
    )
    if not isinstance(payload, list):
        raise GitHubAPIError("GitHub returned invalid issue data")
    return {
        "issues": [
            {
                "number": item.get("number"),
                "title": item.get("title"),
                "state": item.get("state"),
                "user": (item.get("user") or {}).get("login"),
                "html_url": item.get("html_url"),
                "is_pull_request": "pull_request" in item,
            }
            for item in payload[:MAX_RESULTS]
            if isinstance(item, dict)
        ]
    }
