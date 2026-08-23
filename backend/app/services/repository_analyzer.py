import json
from typing import Any

from app.services.github import (
    GitHubBinaryFileError,
    GitHubFileTooLargeError,
    get_file,
    get_repository_tree,
)

MAX_ANALYZER_FILES = 12
MAX_TOTAL_CHARS = 40_000
MAX_FILE_CHARS = 12_000
IMPORTANT_NAMES = {
    "README",
    "README.md",
    "pyproject.toml",
    "requirements.txt",
    "package.json",
    "package-lock.json",
    "pnpm-lock.yaml",
    "yarn.lock",
    "Dockerfile",
    "docker-compose.yml",
    "docker-compose.yaml",
    "next.config.js",
    "next.config.mjs",
    "vite.config.ts",
    "vite.config.js",
    "tsconfig.json",
    "setup.py",
}


def _basename(path: str) -> str:
    return path.rsplit("/", 1)[-1]


def _select_important_paths(tree: list[dict[str, Any]]) -> list[str]:
    selected: list[str] = []
    for entry in tree:
        path = entry.get("path")
        if not isinstance(path, str) or entry.get("type") != "blob":
            continue
        name = _basename(path)
        if name in IMPORTANT_NAMES or path.lower().endswith("/requirements.txt"):
            selected.append(path)
        if len(selected) >= MAX_ANALYZER_FILES:
            break
    return selected


def _detect_stack(paths: set[str], contents: dict[str, str]) -> tuple[str, str, list[str]]:
    lower_paths = {path.lower() for path in paths}
    all_text = "\n".join(contents.values()).lower()
    stack: list[str] = []
    languages: list[str] = []
    if any(path.endswith((".py", ".pyi")) for path in lower_paths) or any(
        name in lower_paths for name in ("pyproject.toml", "requirements.txt", "setup.py")
    ):
        languages.append("Python")
        stack.append("Python")
    if "package.json" in lower_paths or any(
        path.endswith((".js", ".jsx", ".ts", ".tsx")) for path in lower_paths
    ):
        languages.append("JavaScript/TypeScript")
        stack.append("Node.js")
    if "package.json" in lower_paths and any(
        marker in all_text for marker in ('"react"', "from 'react'", 'from \"react\"')
    ):
        stack.append("React")
    if any("next.config" in path for path in lower_paths) or '"next"' in all_text:
        stack.append("Next.js")
    if "fastapi" in all_text:
        stack.append("FastAPI")
    if any(
        _basename(path).lower() in {"dockerfile", "docker-compose.yml", "docker-compose.yaml"}
        for path in paths
    ):
        stack.append("Docker")
    framework = " + ".join(item for item in ("FastAPI", "Next.js", "React") if item in stack)
    language = " + ".join(languages) or "Unknown"
    return language, framework or "Unknown", list(dict.fromkeys(stack))


def _dependency_summary(contents: dict[str, str]) -> dict[str, list[str]]:
    summary: dict[str, list[str]] = {}
    package = contents.get("package.json")
    if package:
        try:
            data = json.loads(package)
            dependencies = {**data.get("dependencies", {}), **data.get("devDependencies", {})}
            summary["node"] = sorted(dependencies)[:50]
        except (TypeError, ValueError):
            summary["node"] = []
    for filename in ("requirements.txt", "pyproject.toml"):
        if filename in contents:
            lines = [line.strip() for line in contents[filename].splitlines()]
            summary["python"] = [line for line in lines if line and not line.startswith("#")][:50]
            break
    return summary


async def analyze_repository(
    owner: str, repo: str, ref: str | None = None, review: bool = False
) -> dict[str, Any]:
    tree_payload = await get_repository_tree(owner, repo, ref)
    tree = tree_payload["tree"]
    paths = {entry["path"] for entry in tree if isinstance(entry.get("path"), str)}
    important_paths = _select_important_paths(tree)
    contents: dict[str, str] = {}
    total_chars = 0
    for path in important_paths:
        try:
            result = await get_file(owner, repo, path, ref)
        except (GitHubBinaryFileError, GitHubFileTooLargeError):
            continue
        content = result.get("content")
        if not isinstance(content, str):
            continue
        remaining = MAX_TOTAL_CHARS - total_chars
        if remaining <= 0:
            break
        clipped = content[: min(MAX_FILE_CHARS, remaining)]
        contents[path] = clipped
        total_chars += len(clipped)

    language, framework, detected_stack = _detect_stack(paths, contents)
    directories = sorted(
        {path.split("/", 1)[0] for path in paths if "/" in path}
    )[:100]
    structure = {
        "file_count": len(paths),
        "directory_count": len(directories),
        "top_level_directories": directories,
        "sample_paths": sorted(paths)[:100],
        "tree_sha": tree_payload.get("sha"),
    }
    return {
        "repository": f"{owner}/{repo}",
        "ref": ref,
        "language": language,
        "framework": framework,
        "detected_stack": detected_stack,
        "project_structure": structure,
        "important_files": [
            {"path": path, "content": content} for path, content in contents.items()
        ],
        "dependency_summary": _dependency_summary(contents),
        "analysis_mode": "review" if review else "overview",
        "content_limits": {
            "max_files": MAX_ANALYZER_FILES,
            "max_file_chars": MAX_FILE_CHARS,
            "max_total_chars": MAX_TOTAL_CHARS,
        },
    }
