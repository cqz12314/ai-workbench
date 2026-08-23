import asyncio
from unittest.mock import AsyncMock

import pytest

from app.services import repository_analyzer
from app.services.github import GitHubAPIError, GitHubBinaryFileError, GitHubFileTooLargeError


def tree(*paths: str):
    return {"sha": "tree-sha", "tree": [{"path": path, "type": "blob"} for path in paths]}


def test_python_fastapi_project_identification(monkeypatch) -> None:
    monkeypatch.setattr(
        repository_analyzer,
        "get_repository_tree",
        AsyncMock(
            return_value=tree("pyproject.toml", "app/main.py", "Dockerfile", "README.md")
        ),
    )
    monkeypatch.setattr(
        repository_analyzer,
        "get_file",
        AsyncMock(
            side_effect=lambda _owner, _repo, path, _ref: {
                "content": (
                    "[project]\ndependencies=['fastapi']"
                    if path == "pyproject.toml"
                    else "from fastapi import FastAPI"
                )
            }
        ),
    )

    result = asyncio.run(repository_analyzer.analyze_repository("octo", "demo"))

    assert result["language"] == "Python"
    assert result["framework"] == "FastAPI"
    assert "Docker" in result["detected_stack"]
    assert result["project_structure"]["file_count"] == 4


def test_node_react_project_identification(monkeypatch) -> None:
    monkeypatch.setattr(
        repository_analyzer,
        "get_repository_tree",
        AsyncMock(return_value=tree("package.json", "src/App.tsx", "next.config.js")),
    )
    monkeypatch.setattr(
        repository_analyzer,
        "get_file",
        AsyncMock(
            return_value={"content": '{"dependencies":{"react":"19.0.0","next":"15.0.0"}}'}
        ),
    )

    result = asyncio.run(repository_analyzer.analyze_repository("octo", "web"))

    assert result["language"] == "JavaScript/TypeScript"
    assert "React" in result["detected_stack"]
    assert "Next.js" in result["detected_stack"]
    assert "react" in result["dependency_summary"]["node"]


def test_analyzer_skips_large_and_binary_files(monkeypatch) -> None:
    monkeypatch.setattr(
        repository_analyzer,
        "get_repository_tree",
        AsyncMock(return_value=tree("README.md", "logo.png", "huge.txt")),
    )

    async def get_file(_owner, _repo, path, _ref):
        if path == "logo.png":
            raise GitHubBinaryFileError("Binary GitHub files are not supported")
        if path == "huge.txt":
            raise GitHubFileTooLargeError("GitHub file exceeds the size limit")
        return {"content": "safe"}

    monkeypatch.setattr(repository_analyzer, "get_file", get_file)
    result = asyncio.run(repository_analyzer.analyze_repository("octo", "demo"))

    assert [item["path"] for item in result["important_files"]] == ["README.md"]


def test_analyzer_propagates_github_errors(monkeypatch) -> None:
    monkeypatch.setattr(
        repository_analyzer,
        "get_repository_tree",
        AsyncMock(side_effect=GitHubAPIError("GitHub access was forbidden")),
    )

    with pytest.raises(GitHubAPIError):
        asyncio.run(repository_analyzer.analyze_repository("octo", "private"))
