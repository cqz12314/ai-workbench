import asyncio
import json
from unittest.mock import AsyncMock

import pytest

from app.services import agent
from app.services.llm import CompletionResult, ToolCall, ToolSelectionResult
from app.services.tools import ToolExecutionError


def tool_selection(*calls: ToolCall) -> ToolSelectionResult:
    return ToolSelectionResult(content=None, tool_calls=list(calls), model="selector-model")


def direct_answer(content: str) -> ToolSelectionResult:
    return ToolSelectionResult(content=content, tool_calls=[], model="final-model")


def test_agent_selects_and_calls_search_tool(monkeypatch) -> None:
    selection = AsyncMock(
        side_effect=[
            tool_selection(
                ToolCall(id="call-1", name="search_knowledge", arguments='{"query":"CAXA"}')
            ),
            direct_answer("CAXA 是 CAD 软件。"),
        ]
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
    completion.assert_not_awaited()
    final_messages = selection.await_args_list[1].args[0]
    assert final_messages[-1]["role"] == "tool"
    assert json.loads(final_messages[-1]["content"])["results"][0]["content"] == ("CAXA is CAD")


def test_agent_calls_github_tool(monkeypatch) -> None:
    selection = AsyncMock(
        side_effect=[
            tool_selection(
                ToolCall(
                    id="github-call",
                    name="github_get_file",
                    arguments='{"owner":"octo","repo":"demo","path":"README.md"}',
                )
            ),
            direct_answer("GitHub 文件内容"),
        ]
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
        side_effect=[
            tool_selection(
                ToolCall(
                    id="analyze-call",
                    name="analyze_repository",
                    arguments='{"owner":"octo","repo":"demo"}',
                )
            ),
            direct_answer("这是一个 FastAPI 项目。"),
        ]
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
        side_effect=[
            tool_selection(
                ToolCall(
                    id="proposal-call",
                    name="propose_change",
                    arguments='{"file_path":"main.py","proposed_content":"print(2)\\n"}',
                )
            ),
            direct_answer("已生成修改方案，请确认后再执行。"),
        ]
    )
    execute = AsyncMock(return_value={"proposal_id": "p1", "diff": "-print(1)\\n+print(2)"})
    completion = AsyncMock(
        return_value=CompletionResult(content="已生成修改方案，请确认后再执行。", model="model")
    )
    monkeypatch.setattr(agent, "select_tools", selection)
    monkeypatch.setattr(agent, "execute_tool", execute)
    monkeypatch.setattr(agent, "create_completion", completion)

    result = asyncio.run(agent.create_agent_completion([{"role": "user", "content": "修改项目"}]))

    assert "确认" in result.content
    assert execute.await_args.args[0] == "propose_change"
    assert all(call.args[0] != "apply_change" for call in execute.await_args_list)


def test_agent_checks_diff_before_commit(monkeypatch) -> None:
    selection = AsyncMock(
        side_effect=[
            tool_selection(ToolCall(id="diff-call", name="git_diff", arguments="{}")),
            direct_answer("这是当前修改差异，等待确认。"),
        ]
    )
    execute = AsyncMock(return_value={"diff": "--- a/main.py\n+++ b/main.py"})
    completion = AsyncMock(
        return_value=CompletionResult(content="这是当前修改差异，等待确认。", model="model")
    )
    monkeypatch.setattr(agent, "select_tools", selection)
    monkeypatch.setattr(agent, "execute_tool", execute)
    monkeypatch.setattr(agent, "create_completion", completion)

    result = asyncio.run(agent.create_agent_completion([{"role": "user", "content": "查看修改"}]))

    assert "等待确认" in result.content
    execute.assert_awaited_once_with("git_diff", "{}")


def test_agent_answers_without_tool(monkeypatch) -> None:
    selection = AsyncMock(
        return_value=ToolSelectionResult(content="2 + 2 = 4", tool_calls=[], model="selector-model")
    )
    execute = AsyncMock()
    completion = AsyncMock()
    monkeypatch.setattr(agent, "select_tools", selection)
    monkeypatch.setattr(agent, "execute_tool", execute)
    monkeypatch.setattr(agent, "create_completion", completion)

    result = asyncio.run(agent.create_agent_completion([{"role": "user", "content": "2 + 2?"}]))

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

    with pytest.raises(ToolExecutionError):
        asyncio.run(agent.create_agent_completion([{"role": "user", "content": "运行命令"}]))


def test_agent_can_call_multiple_read_only_tools(monkeypatch) -> None:
    selection = AsyncMock(
        side_effect=[
            tool_selection(
                ToolCall(id="call-1", name="list_documents", arguments="{}"),
                ToolCall(id="call-2", name="search_knowledge", arguments='{"query":"Agent"}'),
            ),
            direct_answer("没有匹配内容。"),
        ]
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


def test_agent_uses_retrieval_first_multi_round_chain(monkeypatch) -> None:
    selection = AsyncMock(
        side_effect=[
            tool_selection(ToolCall(id="index", name="index_codebase", arguments="{}")),
            tool_selection(
                ToolCall(
                    id="search",
                    name="search_codebase",
                    arguments='{"query":"login","limit":3}',
                )
            ),
            tool_selection(
                ToolCall(id="read", name="read_file", arguments='{"file_path":"auth.py"}')
            ),
            direct_answer("登录逻辑位于 auth.py。"),
        ]
    )
    execute = AsyncMock(
        side_effect=[
            {"indexed": 1},
            {"results": [{"relative_path": "auth.py", "content": "def login(): ..."}]},
            {"path": "auth.py", "content": "def login(): ..."},
        ]
    )
    monkeypatch.setattr(agent, "select_tools", selection)
    monkeypatch.setattr(agent, "execute_tool", execute)

    result = asyncio.run(
        agent.create_agent_completion([{"role": "user", "content": "登录功能在哪里？"}])
    )

    assert result.content == "登录逻辑位于 auth.py。"
    assert [call.args[0] for call in execute.await_args_list] == [
        "index_codebase",
        "search_codebase",
        "read_file",
    ]
    system_prompt = selection.await_args_list[0].args[0][0]["content"]
    assert "search_codebase first" in system_prompt
    assert "never request or construct a repository dump" in system_prompt
    exposed_tools = selection.await_args_list[0].args[1]
    exposed_names = {tool["function"]["name"] for tool in exposed_tools}
    assert "search_codebase" in exposed_names
    assert "apply_change" not in exposed_names
    assert "git_commit" not in exposed_names
    assert "task_plan" not in exposed_names
    assert "run_task" not in exposed_names
    assert "call task_plan before run_task" not in system_prompt
    assert "separate controlled entrypoint" in system_prompt


@pytest.mark.parametrize(
    "tool_name",
    [
        "apply_change",
        "git_commit",
        "git_create_branch",
        "github_create_branch",
        "github_push_branch",
        "github_create_pull_request",
    ],
)
def test_model_confirmation_cannot_trigger_automatic_write(monkeypatch, tool_name: str) -> None:
    selection = AsyncMock(
        return_value=tool_selection(
            ToolCall(id="write", name=tool_name, arguments='{"confirmation":true}')
        )
    )
    execute = AsyncMock()
    monkeypatch.setattr(agent, "select_tools", selection)
    monkeypatch.setattr(agent, "execute_tool", execute)

    with pytest.raises(ToolExecutionError, match="confirmation"):
        asyncio.run(agent.create_agent_completion([{"role": "user", "content": "执行写操作"}]))
    execute.assert_not_awaited()


def test_agent_enforces_maximum_tool_rounds(monkeypatch) -> None:
    selection = AsyncMock(
        return_value=tool_selection(
            ToolCall(id="search", name="search_codebase", arguments='{"query":"x"}')
        )
    )
    execute = AsyncMock(return_value={"results": []})
    completion = AsyncMock(
        return_value=CompletionResult(content="达到工具限制。", model="final-model")
    )
    monkeypatch.setattr(agent, "select_tools", selection)
    monkeypatch.setattr(agent, "execute_tool", execute)
    monkeypatch.setattr(agent, "create_completion", completion)

    result = asyncio.run(agent.create_agent_completion([{"role": "user", "content": "查找"}]))

    assert result.content == "达到工具限制。"
    assert selection.await_count == agent.MAX_AGENT_TOOL_ROUNDS
    assert execute.await_count == agent.MAX_AGENT_TOOL_ROUNDS
    assert "tool limit" in completion.await_args.args[0][-1]["content"]


def test_agent_enforces_cumulative_tool_output_budget(monkeypatch) -> None:
    selection = AsyncMock(
        return_value=tool_selection(
            ToolCall(id="search", name="search_codebase", arguments='{"query":"x"}')
        )
    )
    execute = AsyncMock(return_value={"content": "x" * 30_000})
    completion = AsyncMock(return_value=CompletionResult(content="bounded", model="final-model"))
    monkeypatch.setattr(agent, "select_tools", selection)
    monkeypatch.setattr(agent, "execute_tool", execute)
    monkeypatch.setattr(agent, "create_completion", completion)

    asyncio.run(agent.create_agent_completion([{"role": "user", "content": "查找"}]))

    assert execute.await_count == 1
    tool_message = completion.await_args.args[0][-2]
    assert len(tool_message["content"]) <= agent.MAX_AGENT_TOOL_OUTPUT_CHARS
    assert json.loads(tool_message["content"])["truncated"] is True


def test_tool_output_wrapper_overhead_is_inside_hard_budget() -> None:
    serialized = agent.bounded_tool_output(
        {"content": ('"\\复杂内容' * 10_000)},
        agent.MAX_AGENT_TOOL_OUTPUT_CHARS,
    )

    assert serialized is not None
    assert len(serialized) <= agent.MAX_AGENT_TOOL_OUTPUT_CHARS
    parsed = json.loads(serialized)
    assert parsed["truncated"] is True
    assert isinstance(parsed["content"], str)


def test_same_round_stops_executing_tools_when_output_budget_is_exhausted(
    monkeypatch,
) -> None:
    selection = AsyncMock(
        return_value=tool_selection(
            ToolCall(id="first", name="search_codebase", arguments='{"query":"first"}'),
            ToolCall(id="second", name="search_knowledge", arguments='{"query":"second"}'),
            ToolCall(id="third", name="read_file", arguments='{"file_path":"third.py"}'),
        )
    )
    execute = AsyncMock(return_value={"content": "x" * 30_000})
    completion = AsyncMock(return_value=CompletionResult(content="bounded", model="model"))
    monkeypatch.setattr(agent, "select_tools", selection)
    monkeypatch.setattr(agent, "execute_tool", execute)
    monkeypatch.setattr(agent, "create_completion", completion)

    asyncio.run(agent.create_agent_completion([{"role": "user", "content": "查找"}]))

    execute.assert_awaited_once_with("search_codebase", '{"query":"first"}')
    final_messages = completion.await_args.args[0]
    tool_contents = [
        message["content"] for message in final_messages if message["role"] == "tool"
    ]
    assert len(tool_contents) == 1
    assert sum(len(content) for content in tool_contents) <= agent.MAX_AGENT_TOOL_OUTPUT_CHARS
    assert all(json.loads(content) for content in tool_contents)
    assistant_calls = [
        message["tool_calls"]
        for message in final_messages
        if message["role"] == "assistant" and "tool_calls" in message
    ]
    assert [call["id"] for call in assistant_calls[0]] == ["first"]
