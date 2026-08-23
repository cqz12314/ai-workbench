import asyncio
from unittest.mock import AsyncMock

import pytest
from pydantic import SecretStr

from app.core.config import settings
from app.services import github
from app.services.tools import ToolArgumentsError, ToolExecutionError, execute_tool


def test_github_tools_are_registered() -> None:
    from app.services.tools import TOOL_REGISTRY, get_llm_tools

    names = {name for name in TOOL_REGISTRY if name.startswith("github_")}
    assert names == {
        "github_list_repositories",
        "github_get_file",
        "github_search_code",
        "github_list_issues",
        "github_create_branch",
        "github_push_branch",
        "github_create_pull_request",
    }
    assert {item["function"]["name"] for item in get_llm_tools()} >= names


def test_github_token_is_required(monkeypatch) -> None:
    monkeypatch.setattr(settings, "github_enabled", True)
    monkeypatch.setattr(settings, "github_token", None)

    with pytest.raises(ToolExecutionError, match="token is not configured"):
        asyncio.run(execute_tool("github_list_repositories", {}))


class FakeResponse:
    def __init__(self, status_code: int, payload=None):
        self.status_code = status_code
        self._payload = payload
        self.is_success = 200 <= status_code < 300

    def json(self):
        return self._payload


class FakeAsyncClient:
    response = FakeResponse(200, [])
    requested = None

    def __init__(self, **_kwargs):
        pass

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_args):
        return None

    async def get(self, url, headers):
        self.requested = (url, headers)
        return self.response


@pytest.mark.parametrize("status_code", [401, 403, 429])
def test_github_api_errors_are_normalized(monkeypatch, status_code: int) -> None:
    monkeypatch.setattr(settings, "github_enabled", True)
    monkeypatch.setattr(settings, "github_token", SecretStr("test-token"))
    FakeAsyncClient.response = FakeResponse(status_code, {"message": "secret details"})
    monkeypatch.setattr(github.httpx, "AsyncClient", FakeAsyncClient)

    with pytest.raises(github.GitHubAPIError) as error:
        asyncio.run(github.list_repositories())

    assert "secret details" not in str(error.value)
    assert "test-token" not in str(error.value)


def test_github_request_uses_token_without_returning_it(monkeypatch) -> None:
    monkeypatch.setattr(settings, "github_enabled", True)
    monkeypatch.setattr(settings, "github_token", SecretStr("test-token"))
    FakeAsyncClient.response = FakeResponse(
        200,
        [{"name": "demo", "full_name": "octo/demo", "private": False}],
    )
    client = FakeAsyncClient()
    monkeypatch.setattr(github.httpx, "AsyncClient", lambda **kwargs: client)

    result = asyncio.run(github.list_repositories())

    assert result["repositories"][0]["full_name"] == "octo/demo"
    assert "token" not in str(result).lower()


def test_github_file_size_and_binary_limits(monkeypatch) -> None:
    monkeypatch.setattr(settings, "github_enabled", True)
    monkeypatch.setattr(settings, "github_token", SecretStr("test-token"))
    monkeypatch.setattr(
        github,
        "_get",
        AsyncMock(
            return_value={
                "type": "file",
                "size": github.MAX_FILE_BYTES + 1,
                "encoding": "base64",
                "content": "",
            }
        ),
    )
    with pytest.raises(github.GitHubFileTooLargeError):
        asyncio.run(github.get_file("octo", "demo", "large.txt"))

    monkeypatch.setattr(
        github,
        "_get",
        AsyncMock(
            return_value={
                "type": "file",
                "size": 4,
                "encoding": "base64",
                "content": "AAECAw==",
            }
        ),
    )
    with pytest.raises(github.GitHubBinaryFileError):
        asyncio.run(github.get_file("octo", "demo", "binary.bin"))


def test_repository_tree_rejects_truncated_and_oversized_trees(monkeypatch) -> None:
    monkeypatch.setattr(github, "_get", AsyncMock(return_value={"truncated": True, "tree": []}))
    with pytest.raises(github.GitHubAPIError, match="truncated"):
        asyncio.run(github.get_repository_tree("octo", "demo"))

    oversized = {
        "truncated": False,
        "tree": [
            {"path": str(index), "type": "blob"}
            for index in range(github.MAX_TREE_ENTRIES + 1)
        ],
    }
    monkeypatch.setattr(github, "_get", AsyncMock(return_value=oversized))
    with pytest.raises(github.GitHubAPIError, match="size limit"):
        asyncio.run(github.get_repository_tree("octo", "demo"))


def test_github_tool_arguments_reject_write_like_fields() -> None:
    with pytest.raises(ToolArgumentsError):
        asyncio.run(
            execute_tool(
                "github_get_file",
                {"owner": "octo", "repo": "demo", "path": "a.txt", "push": True},
            )
        )
