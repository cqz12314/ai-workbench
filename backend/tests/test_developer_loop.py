import asyncio
from pathlib import Path
from unittest.mock import AsyncMock

import pytest
from sqlalchemy import create_engine, select
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.db.base import Base
from app.models import DevelopmentRun, DevelopmentTask
from app.services import planner, task_executor, test_runner
from app.services.llm import ToolCall, ToolSelectionResult
from app.services.tools import ToolExecutionError


@pytest.fixture
def development_db(monkeypatch):
    engine = create_engine(
        "sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool
    )
    sessions = sessionmaker(bind=engine, autoflush=False, autocommit=False)
    Base.metadata.create_all(engine)
    monkeypatch.setattr(planner, "SessionLocal", sessions)
    monkeypatch.setattr(task_executor, "SessionLocal", sessions)
    yield sessions
    Base.metadata.drop_all(engine)
    engine.dispose()


def test_planner_creates_steps_and_status(development_db):
    result = planner.create_plan("修改项目并运行测试")
    assert result["status"] == "pending"
    assert [step["action"] for step in result["steps"]] == [
        "analyze_repository",
        "propose_change",
        "run_tests",
    ]
    record, _ = planner.get_plan(result["task_id"])
    assert record.status == "pending"
    planner.update_task(result["task_id"], "running", "analyze")
    record, _ = planner.get_plan(result["task_id"])
    assert record.status == "running"
    assert record.current_step == "analyze"


def test_executor_requires_existing_plan(development_db):
    with pytest.raises(task_executor.TaskExecutorError, match="does not exist"):
        asyncio.run(task_executor.execute_task(999))


def test_executor_records_runs_and_completes(development_db, monkeypatch):
    plan = planner.create_plan("运行测试")
    monkeypatch.setattr(
        task_executor,
        "run_test",
        lambda _name: {"passed": True, "returncode": 0, "test": "pytest"},
    )
    result = asyncio.run(task_executor.execute_task(plan["task_id"]))
    assert result["status"] == "completed"
    with development_db() as session:
        runs = session.scalars(select(DevelopmentRun)).all()
        task = session.get(DevelopmentTask, plan["task_id"])
        assert runs[-1].status == "completed"
        assert task.status == "completed"


def test_test_runner_only_allows_fixed_commands(monkeypatch, tmp_path: Path):
    monkeypatch.setattr(test_runner, "workspace_root", lambda: tmp_path)
    calls = []

    class Result:
        returncode = 0
        stdout = "API_KEY=secret-value"
        stderr = ""

    def run(command, **kwargs):
        calls.append((command, kwargs))
        return Result()

    monkeypatch.setattr(test_runner.subprocess, "run", run)
    result = test_runner.run_test("ruff check")
    assert calls[0][0] == ["ruff", "check", "."]
    assert "secret-value" not in result["stdout"]
    with pytest.raises(test_runner.TestRunnerError):
        test_runner.run_test("rm -rf /")


def test_auto_fix_loop_stops_after_three_rounds():
    tests = AsyncMock(return_value={"passed": False, "stderr": "failure"})
    analyze = AsyncMock(return_value="diagnosis")
    propose = AsyncMock(return_value="proposal")
    apply = AsyncMock()
    result = asyncio.run(
        task_executor.run_auto_fix_loop(tests, analyze, propose, apply, lambda _: True)
    )
    assert result["status"] == "failed"
    assert tests.await_count == 3
    assert apply.await_count == 3


def test_auto_fix_waits_for_confirmation():
    tests = AsyncMock(return_value={"passed": False})
    result = asyncio.run(
        task_executor.run_auto_fix_loop(
            tests,
            AsyncMock(return_value="error"),
            AsyncMock(return_value="proposal"),
            AsyncMock(),
            lambda _: False,
        )
    )
    assert result["status"] == "awaiting_confirmation"


def test_agent_cannot_run_unknown_task(monkeypatch, development_db):
    from app.services import agent

    monkeypatch.setattr(
        agent,
        "select_tools",
        AsyncMock(
            return_value=ToolSelectionResult(
                content=None,
                tool_calls=[ToolCall(id="run", name="run_task", arguments='{"task_id":999}')],
                model="model",
            )
        ),
    )
    with pytest.raises(ToolExecutionError):
        asyncio.run(agent.create_agent_completion([{"role": "user", "content": "执行任务"}]))
