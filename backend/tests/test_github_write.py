import asyncio
from unittest.mock import AsyncMock

import pytest
from pydantic import SecretStr

from app.core.config import settings
from app.services import github_write
from app.services.llm import ToolCall, ToolSelectionResult
from app.services.tools import ToolArgumentsError, ToolExecutionError, execute_tool

SHA = "a" * 40


@pytest.fixture(autouse=True)
def write_settings(monkeypatch):
    monkeypatch.setattr(settings, "github_enabled", True)
    monkeypatch.setattr(settings, "github_write_enabled", True)
    monkeypatch.setattr(settings, "github_token", SecretStr("secret-token"))


def request_responder(method: str, path: str, payload=None):
    if method == "GET" and path.endswith("/git/commits/" + SHA):
        return {"sha": SHA}
    if method == "GET" and path.endswith("/git/ref/heads/ai/feature"):
        return {"object": {"sha": SHA}}
    if method == "GET":
        return {"full_name": "octo/demo"}
    if method == "POST" and path.endswith("/git/refs"):
        return {"ref": "refs/heads/ai/feature"}
    if method == "POST" and path.endswith("/pulls"):
        return {"number": 7, "html_url": "https://github.test/pr/7"}
    if method == "PATCH":
        return {"object": {"sha": SHA}}
    raise AssertionError((method, path, payload))


def test_create_branch_uses_ai_namespace_and_confirmation(monkeypatch):
    request = AsyncMock(side_effect=request_responder)
    monkeypatch.setattr(github_write, "_request", request)

    result = asyncio.run(
        github_write.create_branch("octo", "demo", "feature", SHA, confirmation=True)
    )

    assert result["branch"] == "ai/feature"
    assert request.await_args_list[-1].args[:2] == ("POST", "repos/octo/demo/git/refs")
    with pytest.raises(github_write.GitHubWriteValidationError, match="confirmation"):
        asyncio.run(github_write.create_branch("octo", "demo", "feature", SHA, False))


def test_branch_and_main_protection(monkeypatch):
    with pytest.raises(github_write.GitHubWriteValidationError):
        asyncio.run(github_write.create_branch("octo", "demo", "main", SHA, True))
    with pytest.raises(github_write.GitHubWriteValidationError):
        asyncio.run(github_write.push_branch("octo", "demo", "main", SHA, True))
    with pytest.raises(github_write.GitHubWriteValidationError):
        asyncio.run(github_write.push_branch("octo", "demo", "feature", SHA, True))


def test_push_requires_existing_remote_commit_and_force_is_not_supported(monkeypatch):
    request = AsyncMock(side_effect=request_responder)
    monkeypatch.setattr(github_write, "_request", request)
    result = asyncio.run(github_write.push_branch("octo", "demo", "ai/feature", SHA, True))
    assert result["force"] is False
    with pytest.raises(ToolArgumentsError):
        asyncio.run(
            execute_tool(
                "github_push_branch",
                {
                    "owner": "octo",
                    "repo": "demo",
                    "branch": "ai/feature",
                    "commit_sha": SHA,
                    "confirmation": True,
                    "force": True,
                },
            )
        )

    monkeypatch.setattr(
        github_write,
        "_request",
        AsyncMock(side_effect=lambda method, path, payload=None: {"full_name": "octo/demo"}),
    )
    with pytest.raises(github_write.GitHubWriteAPIError, match="does not exist"):
        asyncio.run(github_write.push_branch("octo", "demo", "ai/feature", SHA, True))


def test_pull_request_checks_source_and_confirmation(monkeypatch):
    request = AsyncMock(side_effect=request_responder)
    monkeypatch.setattr(github_write, "_request", request)
    result = asyncio.run(
        github_write.create_pull_request(
            "octo", "demo", "Add feature", "Details", "ai/feature", confirmation=True
        )
    )
    assert result["target_branch"] == "main"
    assert result["pull_request"]["number"] == 7
    with pytest.raises(github_write.GitHubWriteValidationError, match="confirmation"):
        asyncio.run(
            github_write.create_pull_request(
                "octo", "demo", "Add feature", "Details", "ai/feature", confirmation=False
            )
        )


def test_token_missing_and_api_errors(monkeypatch):
    monkeypatch.setattr(settings, "github_token", None)
    with pytest.raises(ToolExecutionError, match="token"):
        asyncio.run(execute_tool("github_create_branch", {
            "owner": "octo", "repo": "demo", "feature_name": "feature",
            "base_sha": SHA, "confirmation": True,
        }))
    monkeypatch.setattr(settings, "github_token", SecretStr("secret-token"))
    monkeypatch.setattr(
        github_write,
        "_request",
        AsyncMock(side_effect=github_write.GitHubWriteAPIError("GitHub access was forbidden")),
    )
    with pytest.raises(ToolExecutionError, match="forbidden"):
        asyncio.run(execute_tool("github_create_branch", {
            "owner": "octo", "repo": "demo", "feature_name": "feature",
            "base_sha": SHA, "confirmation": True,
        }))


def test_agent_unconfirmed_write_is_rejected(monkeypatch):
    from app.services import agent

    monkeypatch.setattr(
        agent,
        "select_tools",
        AsyncMock(return_value=ToolSelectionResult(
            content=None,
            tool_calls=[ToolCall(
                id="write", name="github_create_branch",
                arguments=(
                    '{"owner":"octo","repo":"demo","feature_name":"x","base_sha":"'
                    + SHA
                    + '"}'
                ),
            )],
            model="model",
        )),
    )
    with pytest.raises(ToolExecutionError, match="confirmation"):
        asyncio.run(agent.create_agent_completion([{"role": "user", "content": "创建分支"}]))
