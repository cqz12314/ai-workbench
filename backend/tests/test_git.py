import asyncio
import subprocess

import pytest

from app.core.config import settings
from app.services import git
from app.services.tools import ToolExecutionError, execute_tool


@pytest.fixture
def git_workspace(tmp_path, monkeypatch):
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=tmp_path, check=True)
    subprocess.run(["git", "config", "user.name", "Test User"], cwd=tmp_path, check=True)
    (tmp_path / "main.py").write_text("print('old')\n", encoding="utf-8")
    subprocess.run(["git", "add", "--", "main.py"], cwd=tmp_path, check=True)
    subprocess.run(["git", "commit", "-qm", "initial"], cwd=tmp_path, check=True)
    (tmp_path / "main.py").write_text("print('new')\n", encoding="utf-8")
    monkeypatch.setattr(settings, "git_enabled", True)
    monkeypatch.setattr(settings, "workspace_enabled", True)
    monkeypatch.setattr(settings, "workspace_root", str(tmp_path))
    return tmp_path


def test_git_status_and_diff(git_workspace):
    status = git.git_status()
    diff = git.git_diff()
    assert status["clean"] is False
    assert status["changes"][0]["path"] == "main.py"
    assert "print('new')" in diff["diff"]


def test_git_create_branch(git_workspace):
    result = git.git_create_branch("feature/safe-change")
    assert result["branch"] == "feature/safe-change"
    assert git.git_status()["branch"] == "feature/safe-change"


def test_git_commit_requires_confirmation_and_returns_audit(git_workspace):
    with pytest.raises(git.GitValidationError, match="confirmation"):
        git.git_commit("update", ["main.py"], False)
    result = git.git_commit("update main", ["main.py"], True, proposal_id="proposal-1")
    assert len(result["commit_hash"]) == 40
    assert result["file_paths"] == ["main.py"]
    assert result["proposal_id"] == "proposal-1"
    assert git.git_status()["clean"] is True


def test_git_commit_rejects_deleted_files(git_workspace):
    (git_workspace / "main.py").unlink()
    with pytest.raises(git.GitValidationError, match="deleted"):
        git.git_commit("delete", ["main.py"], True)


def test_git_disabled_and_forbidden_tools(monkeypatch, git_workspace):
    monkeypatch.setattr(settings, "git_enabled", False)
    with pytest.raises(ToolExecutionError, match="disabled"):
        asyncio.run(execute_tool("git_status", {}))

    from app.services.tools import TOOL_REGISTRY

    assert not {"git_push", "git_pull", "git_merge", "git_delete_branch", "shell"}.intersection(
        TOOL_REGISTRY
    )
