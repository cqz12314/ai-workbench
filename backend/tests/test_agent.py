import asyncio
import json
from unittest.mock import AsyncMock

import pytest

from app.services import agent
from app.services.llm import CompletionResult, ToolCall, ToolSelectionResult
from app.services.tools import ToolNotFoundError


def tool_selection(*calls: ToolCall) -> ToolSelectionResult:
    return ToolSelectionResult(content=None, tool_calls=list(calls), model="selector-model")


def test_agent_selects_and_calls_search_tool(monkeypatch) -> None:
    selection = AsyncMock(
        return_value=tool_selection(
            ToolCall(id="call-1", name="search_knowledge", arguments='{"query":"CAXA"}')
        )
    )
    execute = AsyncMock(return_value={"results": [{"content": "CAXA is CAD"}]})
    completion = AsyncMock(
        return_value=CompletionResult(content="CAXA 是 CAD 软件。", model="final-model")
    )
    monkeypatch.setattr(agent, "select_tools", selection)
    monkeypatch.setattr(agent, "execute_tool", execute)
    monkeypatch.setattr(agent, "create_completion", completion)

    result = asyncio.run(
        agent.create_agent_completion([{"role": "user", "content": "CAXA 是什么？"}])
    )

    assert result.content == "CAXA 是 CAD 软件。"
    execute.assert_called_once_with("search_knowledge", '{"query":"CAXA"}')
    final_messages = completion.await_args.args[0]
    assert final_messages[-1]["role"] == "tool"
    assert json.loads(final_messages[-1]["content"])["results"][0]["content"] == (
        "CAXA is CAD"
    )


def test_agent_calls_github_tool(monkeypatch) -> None:
    selection = AsyncMock(
        return_value=tool_selection(
            ToolCall(
                id="github-call",
                name="github_get_file",
                arguments='{"owner":"octo","repo":"demo","path":"README.md"}',
            )
        )
    )
    execute = AsyncMock(return_value={"path": "README.md", "content": "hello"})
    completion = AsyncMock(
        return_value=CompletionResult(content="GitHub 文件内容", model="final-model")
    )
    monkeypatch.setattr(agent, "select_tools", selection)
    monkeypatch.setattr(agent, "execute_tool", execute)
    monkeypatch.setattr(agent, "create_completion", completion)

    result = asyncio.run(
        agent.create_agent_completion([{"role": "user", "content": "读取 GitHub 文件"}])
    )

    assert result.content == "GitHub 文件内容"
    execute.assert_awaited_once_with(
        "github_get_file", '{"owner":"octo","repo":"demo","path":"README.md"}'
    )


def test_agent_calls_repository_analyzer(monkeypatch) -> None:
    selection = AsyncMock(
        return_value=tool_selection(
            ToolCall(
                id="analyze-call",
                name="analyze_repository",
                arguments='{"owner":"octo","repo":"demo"}',
            )
        )
    )
    execute = AsyncMock(
        return_value={"language": "Python", "framework": "FastAPI", "project_structure": {}}
    )
    completion = AsyncMock(
        return_value=CompletionResult(content="这是一个 FastAPI 项目。", model="final-model")
    )
    monkeypatch.setattr(agent, "select_tools", selection)
    monkeypatch.setattr(agent, "execute_tool", execute)
    monkeypatch.setattr(agent, "create_completion", completion)

    result = asyncio.run(
        agent.create_agent_completion([{"role": "user", "content": "分析我的 GitHub 项目"}])
    )

    assert result.content == "这是一个 FastAPI 项目。"
    execute.assert_awaited_once_with("analyze_repository", '{"owner":"octo","repo":"demo"}')


def test_agent_proposes_change_without_applying(monkeypatch) -> None:
    selection = AsyncMock(
        return_value=tool_selection(
            ToolCall(
                id="proposal-call",
                name="propose_change",
                arguments='{"file_path":"main.py","proposed_content":"print(2)\\n"}',
            )
        )
    )
    execute = AsyncMock(return_value={"proposal_id": "p1", "diff": "-print(1)\\n+print(2)"})
    completion = AsyncMock(
        return_value=CompletionResult(content="已生成修改方案，请确认后再执行。", model="model")
    )
    monkeypatch.setattr(agent, "select_tools", selection)
    monkeypatch.setattr(agent, "execute_tool", execute)
    monkeypatch.setattr(agent, "create_completion", completion)

    result = asyncio.run(
        agent.create_agent_completion([{"role": "user", "content": "修改项目"}])
    )

    assert "确认" in result.content
    assert execute.await_args.args[0] == "propose_change"
    assert all(call.args[0] != "apply_change" for call in execute.await_args_list)


def test_agent_checks_diff_before_commit(monkeypatch) -> None:
    selection = AsyncMock(
        return_value=tool_selection(
            ToolCall(id="diff-call", name="git_diff", arguments="{}")
        )
    )
    execute = AsyncMock(return_value={"diff": "--- a/main.py\n+++ b/main.py"})
    completion = AsyncMock(
        return_value=CompletionResult(content="这是当前修改差异，等待确认。", model="model")
    )
    monkeypatch.setattr(agent, "select_tools", selection)
    monkeypatch.setattr(agent, "execute_tool", execute)
    monkeypatch.setattr(agent, "create_completion", completion)

    result = asyncio.run(
        agent.create_agent_completion([{"role": "user", "content": "查看修改"}])
    )

    assert "等待确认" in result.content
    execute.assert_awaited_once_with("git_diff", "{}")


def test_agent_answers_without_tool(monkeypatch) -> None:
    selection = AsyncMock(
        return_value=ToolSelectionResult(
            content="2 + 2 = 4", tool_calls=[], model="selector-model"
        )
    )
    execute = AsyncMock()
    completion = AsyncMock()
    monkeypatch.setattr(agent, "select_tools", selection)
    monkeypatch.setattr(agent, "execute_tool", execute)
    monkeypatch.setattr(agent, "create_completion", completion)

    result = asyncio.run(
        agent.create_agent_completion([{"role": "user", "content": "2 + 2?"}])
    )

    assert result == CompletionResult(content="2 + 2 = 4", model="selector-model")
    execute.assert_not_called()
    completion.assert_not_awaited()


def test_agent_rejects_tool_outside_allowlist(monkeypatch) -> None:
    selection = AsyncMock(
        return_value=tool_selection(
            ToolCall(id="call-1", name="shell", arguments='{"command":"id"}')
        )
    )
    monkeypatch.setattr(agent, "select_tools", selection)

    with pytest.raises(ToolNotFoundError):
        asyncio.run(
            agent.create_agent_completion([{"role": "user", "content": "运行命令"}])
        )


def test_agent_can_call_multiple_read_only_tools(monkeypatch) -> None:
    selection = AsyncMock(
        return_value=tool_selection(
            ToolCall(id="call-1", name="list_documents", arguments="{}"),
            ToolCall(id="call-2", name="search_knowledge", arguments='{"query":"Agent"}'),
        )
    )
    execute = AsyncMock(side_effect=[{"documents": []}, {"results": []}])
    completion = AsyncMock(
        return_value=CompletionResult(content="没有匹配内容。", model="final-model")
    )
    monkeypatch.setattr(agent, "select_tools", selection)
    monkeypatch.setattr(agent, "execute_tool", execute)
    monkeypatch.setattr(agent, "create_completion", completion)

    asyncio.run(agent.create_agent_completion([{"role": "user", "content": "查找 Agent"}]))

    assert [call.args[0] for call in execute.call_args_list] == [
        "list_documents",
        "search_knowledge",
    ]
