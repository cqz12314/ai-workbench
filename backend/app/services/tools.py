import json
import logging
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator
from sqlalchemy import select
from sqlalchemy.exc import SQLAlchemyError

from app.db.session import SessionLocal
from app.models import Document
from app.services.git import (
    GitError,
    git_commit,
    git_create_branch,
    git_diff,
    git_status,
)
from app.services.github import (
    GitHubError,
    get_file,
    list_issues,
    list_repositories,
    search_code,
)
from app.services.github_write import GitHubWriteError
from app.services.github_write import create_branch as github_create_branch
from app.services.github_write import create_pull_request as github_create_pull_request
from app.services.github_write import push_branch as github_push_branch
from app.services.planner import PlannerError, create_plan
from app.services.rag import RAGRetrievalError, search_knowledge
from app.services.repository_analyzer import analyze_repository
from app.services.task_executor import TaskExecutorError, execute_task
from app.services.workspace import (
    WorkspaceError,
    apply_change,
    propose_change,
    read_file,
)

logger = logging.getLogger(__name__)


class ToolError(RuntimeError):
    """Base error for safe, read-only Agent tools."""


class ToolNotFoundError(ToolError):
    """Raised when a model requests a tool outside the allowlist."""


class ToolArgumentsError(ToolError):
    """Raised when a model supplies invalid tool arguments."""


class ToolExecutionError(ToolError):
    """Raised when an allowed tool cannot complete its read-only operation."""


class SearchKnowledgeArguments(BaseModel):
    model_config = ConfigDict(extra="forbid")

    query: str = Field(min_length=1, max_length=1000)
    limit: int = Field(default=5, ge=1, le=20)

    @field_validator("query")
    @classmethod
    def normalize_query(cls, value: str) -> str:
        query = value.strip()
        if not query:
            raise ValueError("query must not be blank")
        return query


class ListDocumentsArguments(BaseModel):
    model_config = ConfigDict(extra="forbid")


class GitHubPaginationArguments(BaseModel):
    model_config = ConfigDict(extra="forbid")

    per_page: int = Field(default=20, ge=1, le=20)
    page: int = Field(default=1, ge=1, le=100)


class GitHubRepositoryArguments(GitHubPaginationArguments):
    pass


class GitHubFileArguments(BaseModel):
    model_config = ConfigDict(extra="forbid")

    owner: str = Field(min_length=1, max_length=100)
    repo: str = Field(min_length=1, max_length=100)
    path: str = Field(min_length=1, max_length=1000)
    ref: str | None = Field(default=None, max_length=255)


class GitHubCodeSearchArguments(GitHubPaginationArguments):
    query: str = Field(min_length=1, max_length=500)

    @field_validator("query")
    @classmethod
    def normalize_query(cls, value: str) -> str:
        query = value.strip()
        if not query:
            raise ValueError("query must not be blank")
        return query


class GitHubIssuesArguments(GitHubPaginationArguments):
    owner: str = Field(min_length=1, max_length=100)
    repo: str = Field(min_length=1, max_length=100)
    state: str = Field(default="open", pattern="^(open|closed|all)$")


class RepositoryArguments(BaseModel):
    model_config = ConfigDict(extra="forbid")

    owner: str = Field(min_length=1, max_length=100)
    repo: str = Field(min_length=1, max_length=100)
    ref: str | None = Field(default=None, max_length=255)


class WorkspaceFileArguments(BaseModel):
    model_config = ConfigDict(extra="forbid")

    file_path: str = Field(min_length=1, max_length=1000)


class ProposeChangeArguments(WorkspaceFileArguments):
    proposed_content: str = Field(max_length=100_000)


class ApplyChangeArguments(BaseModel):
    model_config = ConfigDict(extra="forbid")

    proposal_id: str = Field(min_length=1, max_length=64)
    confirmation: bool = False


class GitDiffArguments(BaseModel):
    model_config = ConfigDict(extra="forbid")

    staged: bool = False


class GitBranchArguments(BaseModel):
    model_config = ConfigDict(extra="forbid")

    branch_name: str = Field(min_length=1, max_length=100)


class GitCommitArguments(BaseModel):
    model_config = ConfigDict(extra="forbid")

    message: str = Field(min_length=1, max_length=200)
    file_paths: list[str] = Field(min_length=1, max_length=50)
    confirmation: bool = False
    proposal_id: str | None = Field(default=None, max_length=64)


class GitHubWriteBranchArguments(BaseModel):
    model_config = ConfigDict(extra="forbid")

    owner: str = Field(min_length=1, max_length=100)
    repo: str = Field(min_length=1, max_length=100)
    feature_name: str = Field(min_length=1, max_length=100)
    base_sha: str = Field(min_length=40, max_length=40)
    confirmation: bool = False


class GitHubPushBranchArguments(BaseModel):
    model_config = ConfigDict(extra="forbid")

    owner: str = Field(min_length=1, max_length=100)
    repo: str = Field(min_length=1, max_length=100)
    branch: str = Field(min_length=1, max_length=100)
    commit_sha: str = Field(min_length=40, max_length=40)
    confirmation: bool = False


class GitHubPullRequestArguments(BaseModel):
    model_config = ConfigDict(extra="forbid")

    owner: str = Field(min_length=1, max_length=100)
    repo: str = Field(min_length=1, max_length=100)
    title: str = Field(min_length=1, max_length=256)
    description: str = Field(default="", max_length=20_000)
    source_branch: str = Field(min_length=1, max_length=100)
    target_branch: str = Field(default="main", min_length=1, max_length=100)
    confirmation: bool = False


class TaskPlanArguments(BaseModel):
    model_config = ConfigDict(extra="forbid")

    task: str = Field(min_length=1, max_length=10_000)


class RunTaskArguments(BaseModel):
    model_config = ConfigDict(extra="forbid")

    task_id: int = Field(gt=0)
    test_name: Literal["pytest", "ruff check", "npm run build", "npm run lint"] = "pytest"


ToolArguments = (
    SearchKnowledgeArguments
    | ListDocumentsArguments
    | GitHubRepositoryArguments
    | GitHubFileArguments
    | GitHubCodeSearchArguments
    | GitHubIssuesArguments
    | RepositoryArguments
    | WorkspaceFileArguments
    | ProposeChangeArguments
    | ApplyChangeArguments
    | GitDiffArguments
    | GitBranchArguments
    | GitCommitArguments
    | GitHubWriteBranchArguments
    | GitHubPushBranchArguments
    | GitHubPullRequestArguments
    | TaskPlanArguments
    | RunTaskArguments
)
ToolExecutor = Callable[[ToolArguments], dict[str, Any] | Awaitable[dict[str, Any]]]


@dataclass(frozen=True)
class ToolDefinition:
    name: str
    description: str
    arguments_model: type[BaseModel]
    executor: ToolExecutor

    def as_llm_tool(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.arguments_model.model_json_schema(),
            },
        }


def execute_search_knowledge(arguments: ToolArguments) -> dict[str, Any]:
    if not isinstance(arguments, SearchKnowledgeArguments):
        raise ToolArgumentsError("Invalid arguments for search_knowledge")
    try:
        results = search_knowledge(arguments.query, arguments.limit)
    except RAGRetrievalError as exc:
        logger.exception("Agent knowledge search failed")
        raise ToolExecutionError("Knowledge-base search is unavailable") from exc
    return {
        "results": [
            {
                "chunk_id": result.chunk_id,
                "document_id": result.document_id,
                "chunk_index": result.chunk_index,
                "filename": result.filename,
                "content": result.content,
                "distance": result.distance,
            }
            for result in results
        ]
    }


def execute_list_documents(arguments: ToolArguments) -> dict[str, Any]:
    if not isinstance(arguments, ListDocumentsArguments):
        raise ToolArgumentsError("Invalid arguments for list_documents")
    try:
        with SessionLocal() as session:
            documents = list(
                session.scalars(
                    select(Document).order_by(Document.created_at.desc(), Document.id.desc())
                )
            )
    except SQLAlchemyError as exc:
        logger.exception("Agent document listing failed")
        raise ToolExecutionError("Document listing is unavailable") from exc
    return {
        "documents": [
            {
                "id": document.id,
                "filename": document.filename,
                "file_type": document.file_type,
                "created_at": document.created_at.isoformat(),
            }
            for document in documents
        ]
    }


async def execute_github_list_repositories(arguments: ToolArguments) -> dict[str, Any]:
    if not isinstance(arguments, GitHubRepositoryArguments):
        raise ToolArgumentsError("Invalid arguments for github_list_repositories")
    try:
        return await list_repositories(arguments.per_page, arguments.page)
    except GitHubError as exc:
        raise ToolExecutionError(str(exc)) from exc


async def execute_github_get_file(arguments: ToolArguments) -> dict[str, Any]:
    if not isinstance(arguments, GitHubFileArguments):
        raise ToolArgumentsError("Invalid arguments for github_get_file")
    try:
        return await get_file(arguments.owner, arguments.repo, arguments.path, arguments.ref)
    except GitHubError as exc:
        raise ToolExecutionError(str(exc)) from exc


async def execute_github_search_code(arguments: ToolArguments) -> dict[str, Any]:
    if not isinstance(arguments, GitHubCodeSearchArguments):
        raise ToolArgumentsError("Invalid arguments for github_search_code")
    try:
        return await search_code(arguments.query, arguments.per_page, arguments.page)
    except GitHubError as exc:
        raise ToolExecutionError(str(exc)) from exc


async def execute_github_list_issues(arguments: ToolArguments) -> dict[str, Any]:
    if not isinstance(arguments, GitHubIssuesArguments):
        raise ToolArgumentsError("Invalid arguments for github_list_issues")
    try:
        return await list_issues(
            arguments.owner,
            arguments.repo,
            arguments.state,
            arguments.per_page,
            arguments.page,
        )
    except GitHubError as exc:
        raise ToolExecutionError(str(exc)) from exc


async def execute_analyze_repository(arguments: ToolArguments) -> dict[str, Any]:
    if not isinstance(arguments, RepositoryArguments):
        raise ToolArgumentsError("Invalid arguments for analyze_repository")
    try:
        return await analyze_repository(arguments.owner, arguments.repo, arguments.ref)
    except GitHubError as exc:
        raise ToolExecutionError(str(exc)) from exc


async def execute_review_repository(arguments: ToolArguments) -> dict[str, Any]:
    if not isinstance(arguments, RepositoryArguments):
        raise ToolArgumentsError("Invalid arguments for review_repository")
    try:
        return await analyze_repository(arguments.owner, arguments.repo, arguments.ref, review=True)
    except GitHubError as exc:
        raise ToolExecutionError(str(exc)) from exc


def execute_read_file(arguments: ToolArguments) -> dict[str, Any]:
    if not isinstance(arguments, WorkspaceFileArguments):
        raise ToolArgumentsError("Invalid arguments for read_file")
    try:
        return read_file(arguments.file_path)
    except WorkspaceError as exc:
        raise ToolExecutionError(str(exc)) from exc


def execute_propose_change(arguments: ToolArguments) -> dict[str, Any]:
    if not isinstance(arguments, ProposeChangeArguments):
        raise ToolArgumentsError("Invalid arguments for propose_change")
    try:
        return propose_change(arguments.file_path, arguments.proposed_content)
    except WorkspaceError as exc:
        raise ToolExecutionError(str(exc)) from exc


def execute_apply_change(arguments: ToolArguments) -> dict[str, Any]:
    if not isinstance(arguments, ApplyChangeArguments):
        raise ToolArgumentsError("Invalid arguments for apply_change")
    try:
        return apply_change(arguments.proposal_id, arguments.confirmation)
    except WorkspaceError as exc:
        raise ToolExecutionError(str(exc)) from exc


def execute_git_status(arguments: ToolArguments) -> dict[str, Any]:
    if not isinstance(arguments, ListDocumentsArguments):
        raise ToolArgumentsError("Invalid arguments for git_status")
    try:
        return git_status()
    except GitError as exc:
        raise ToolExecutionError(str(exc)) from exc


def execute_git_diff(arguments: ToolArguments) -> dict[str, Any]:
    if not isinstance(arguments, GitDiffArguments):
        raise ToolArgumentsError("Invalid arguments for git_diff")
    try:
        return git_diff(arguments.staged)
    except GitError as exc:
        raise ToolExecutionError(str(exc)) from exc


def execute_git_create_branch(arguments: ToolArguments) -> dict[str, Any]:
    if not isinstance(arguments, GitBranchArguments):
        raise ToolArgumentsError("Invalid arguments for git_create_branch")
    try:
        return git_create_branch(arguments.branch_name)
    except GitError as exc:
        raise ToolExecutionError(str(exc)) from exc


def execute_git_commit(arguments: ToolArguments) -> dict[str, Any]:
    if not isinstance(arguments, GitCommitArguments):
        raise ToolArgumentsError("Invalid arguments for git_commit")
    try:
        return git_commit(
            arguments.message,
            arguments.file_paths,
            arguments.confirmation,
            arguments.proposal_id,
        )
    except GitError as exc:
        raise ToolExecutionError(str(exc)) from exc


async def execute_github_create_branch(arguments: ToolArguments) -> dict[str, Any]:
    if not isinstance(arguments, GitHubWriteBranchArguments):
        raise ToolArgumentsError("Invalid arguments for github_create_branch")
    try:
        return await github_create_branch(
            arguments.owner,
            arguments.repo,
            arguments.feature_name,
            arguments.base_sha,
            arguments.confirmation,
        )
    except GitHubWriteError as exc:
        raise ToolExecutionError(str(exc)) from exc


async def execute_github_push_branch(arguments: ToolArguments) -> dict[str, Any]:
    if not isinstance(arguments, GitHubPushBranchArguments):
        raise ToolArgumentsError("Invalid arguments for github_push_branch")
    try:
        return await github_push_branch(
            arguments.owner,
            arguments.repo,
            arguments.branch,
            arguments.commit_sha,
            arguments.confirmation,
        )
    except GitHubWriteError as exc:
        raise ToolExecutionError(str(exc)) from exc


async def execute_github_create_pull_request(arguments: ToolArguments) -> dict[str, Any]:
    if not isinstance(arguments, GitHubPullRequestArguments):
        raise ToolArgumentsError("Invalid arguments for github_create_pull_request")
    try:
        return await github_create_pull_request(
            arguments.owner,
            arguments.repo,
            arguments.title,
            arguments.description,
            arguments.source_branch,
            arguments.target_branch,
            arguments.confirmation,
        )
    except GitHubWriteError as exc:
        raise ToolExecutionError(str(exc)) from exc


def execute_task_plan(arguments: ToolArguments) -> dict[str, Any]:
    if not isinstance(arguments, TaskPlanArguments):
        raise ToolArgumentsError("Invalid arguments for task_plan")
    try:
        return create_plan(arguments.task)
    except PlannerError as exc:
        raise ToolExecutionError(str(exc)) from exc


async def execute_run_task(arguments: ToolArguments) -> dict[str, Any]:
    if not isinstance(arguments, RunTaskArguments):
        raise ToolArgumentsError("Invalid arguments for run_task")
    try:
        return await execute_task(arguments.task_id, arguments.test_name)
    except TaskExecutorError as exc:
        raise ToolExecutionError(str(exc)) from exc


TOOL_REGISTRY = {
    "search_knowledge": ToolDefinition(
        name="search_knowledge",
        description="Search uploaded AI Workbench documents for relevant text chunks.",
        arguments_model=SearchKnowledgeArguments,
        executor=execute_search_knowledge,
    ),
    "list_documents": ToolDefinition(
        name="list_documents",
        description="List uploaded documents without exposing server filesystem paths.",
        arguments_model=ListDocumentsArguments,
        executor=execute_list_documents,
    ),
    "github_list_repositories": ToolDefinition(
        name="github_list_repositories",
        description="List repositories accessible to the authenticated GitHub user.",
        arguments_model=GitHubRepositoryArguments,
        executor=execute_github_list_repositories,
    ),
    "github_get_file": ToolDefinition(
        name="github_get_file",
        description="Read a UTF-8 text file from a GitHub repository without modifying it.",
        arguments_model=GitHubFileArguments,
        executor=execute_github_get_file,
    ),
    "github_search_code": ToolDefinition(
        name="github_search_code",
        description="Search GitHub code using a read-only code search query.",
        arguments_model=GitHubCodeSearchArguments,
        executor=execute_github_search_code,
    ),
    "github_list_issues": ToolDefinition(
        name="github_list_issues",
        description="List issues in a GitHub repository without changing them.",
        arguments_model=GitHubIssuesArguments,
        executor=execute_github_list_issues,
    ),
    "analyze_repository": ToolDefinition(
        name="analyze_repository",
        description="Analyze a GitHub repository's read-only structure, stack, and key files.",
        arguments_model=RepositoryArguments,
        executor=execute_analyze_repository,
    ),
    "review_repository": ToolDefinition(
        name="review_repository",
        description="Review a GitHub repository's read-only architecture and maintenance risks.",
        arguments_model=RepositoryArguments,
        executor=execute_review_repository,
    ),
    "read_file": ToolDefinition(
        name="read_file",
        description="Read an allowed text file inside the configured workspace.",
        arguments_model=WorkspaceFileArguments,
        executor=execute_read_file,
    ),
    "propose_change": ToolDefinition(
        name="propose_change",
        description="Generate a unified diff for a workspace file without writing it.",
        arguments_model=ProposeChangeArguments,
        executor=execute_propose_change,
    ),
    "apply_change": ToolDefinition(
        name="apply_change",
        description="Apply an existing confirmed workspace proposal after backup and hash checks.",
        arguments_model=ApplyChangeArguments,
        executor=execute_apply_change,
    ),
    "git_status": ToolDefinition(
        name="git_status",
        description="Read the controlled workspace Git status.",
        arguments_model=ListDocumentsArguments,
        executor=execute_git_status,
    ),
    "git_diff": ToolDefinition(
        name="git_diff",
        description="Read the controlled workspace Git diff.",
        arguments_model=GitDiffArguments,
        executor=execute_git_diff,
    ),
    "git_create_branch": ToolDefinition(
        name="git_create_branch",
        description="Create a validated local Git branch in the controlled workspace.",
        arguments_model=GitBranchArguments,
        executor=execute_git_create_branch,
    ),
    "git_commit": ToolDefinition(
        name="git_commit",
        description="Commit explicitly selected workspace files after explicit confirmation.",
        arguments_model=GitCommitArguments,
        executor=execute_git_commit,
    ),
    "github_create_branch": ToolDefinition(
        name="github_create_branch",
        description="Create a confirmed ai/* branch on GitHub from an existing commit SHA.",
        arguments_model=GitHubWriteBranchArguments,
        executor=execute_github_create_branch,
    ),
    "github_push_branch": ToolDefinition(
        name="github_push_branch",
        description="Update a confirmed ai/* GitHub branch to an existing remote commit SHA.",
        arguments_model=GitHubPushBranchArguments,
        executor=execute_github_push_branch,
    ),
    "github_create_pull_request": ToolDefinition(
        name="github_create_pull_request",
        description="Create a confirmed pull request from an ai/* branch.",
        arguments_model=GitHubPullRequestArguments,
        executor=execute_github_create_pull_request,
    ),
    "task_plan": ToolDefinition(
        name="task_plan",
        description="Create a persisted development plan before any task execution.",
        arguments_model=TaskPlanArguments,
        executor=execute_task_plan,
    ),
    "run_task": ToolDefinition(
        name="run_task",
        description="Execute an existing development task plan using only safe allowlisted steps.",
        arguments_model=RunTaskArguments,
        executor=execute_run_task,
    ),
}


def get_llm_tools() -> list[dict[str, Any]]:
    return [definition.as_llm_tool() for definition in TOOL_REGISTRY.values()]


async def execute_tool(name: str, arguments: str | dict[str, Any]) -> dict[str, Any]:
    definition = TOOL_REGISTRY.get(name)
    if definition is None:
        raise ToolNotFoundError("The requested tool is not available")

    try:
        raw_arguments = json.loads(arguments) if isinstance(arguments, str) else arguments
    except json.JSONDecodeError as exc:
        raise ToolArgumentsError("The tool arguments are invalid") from exc
    if not isinstance(raw_arguments, dict):
        raise ToolArgumentsError("The tool arguments are invalid")

    try:
        validated = definition.arguments_model.model_validate(raw_arguments)
    except ValidationError as exc:
        raise ToolArgumentsError("The tool arguments are invalid") from exc
    result = definition.executor(validated)
    if isinstance(result, Awaitable):
        return await result
    return result
