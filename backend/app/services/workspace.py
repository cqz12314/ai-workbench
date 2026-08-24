import difflib
import hashlib
import secrets
import tempfile
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from app.core.config import settings
from app.db.session import SessionLocal
from app.models import ChangeHistory

MAX_WORKSPACE_FILE_BYTES = 1_048_576
MAX_CHANGE_CONTENT_CHARS = 100_000
BACKUP_DIRECTORY = ".ai_workbench_backups"
SYSTEM_PATH_PREFIXES = (
    "/bin",
    "/boot",
    "/dev",
    "/etc",
    "/proc",
    "/root",
    "/sbin",
    "/sys",
    "/usr",
    "/var",
)
SENSITIVE_DIRECTORY_NAMES = {".ssh", ".git", ".env", "certificates"}
CREDENTIAL_SUFFIXES = {".key", ".pem", ".p12", ".crt", ".cer"}
CREDENTIAL_FILENAMES = {
    "id_rsa",
    "id_ed25519",
    "credentials",
    "credentials.json",
    "private_key",
    "private-key",
    "secret",
    "secrets",
    "token",
    "tokens",
}


class WorkspaceError(RuntimeError):
    """Base error for controlled workspace operations."""


class WorkspaceDisabledError(WorkspaceError):
    pass


class WorkspacePathError(WorkspaceError):
    pass


class WorkspaceFileError(WorkspaceError):
    pass


class ProposalError(WorkspaceError):
    pass


@dataclass(frozen=True)
class ChangeProposal:
    proposal_id: str
    file_path: str
    original_hash: str
    proposed_hash: str
    diff: str
    proposed_content: str


_proposals: dict[str, ChangeProposal] = {}


def _workspace_root() -> Path:
    if not settings.workspace_enabled:
        raise WorkspaceDisabledError("Workspace changes are disabled")
    if not settings.workspace_root:
        raise WorkspaceDisabledError("Workspace root is not configured")
    root = Path(settings.workspace_root).expanduser().resolve()
    if not root.exists() or not root.is_dir():
        raise WorkspacePathError("Workspace root is unavailable")
    if root == Path(root.anchor) or any(
        str(root) == prefix or str(root).startswith(f"{prefix}/") for prefix in SYSTEM_PATH_PREFIXES
    ):
        raise WorkspacePathError("System paths cannot be used as a workspace")
    return root


def workspace_root() -> Path:
    """Return the validated configured workspace root."""
    return _workspace_root()


def is_sensitive_relative_path(relative_path: str | Path) -> bool:
    """Return whether a workspace-relative path is unsafe to expose or index."""
    path = Path(relative_path)
    lowered_parts = tuple(part.casefold() for part in path.parts)
    if any(part in SENSITIVE_DIRECTORY_NAMES or part.startswith(".env.") for part in lowered_parts):
        return True
    name = path.name.casefold()
    if name == ".env" or name.startswith(".env."):
        return True
    if name in CREDENTIAL_FILENAMES or path.suffix.casefold() in CREDENTIAL_SUFFIXES:
        return True
    stem = path.stem.casefold().replace("-", "_")
    return stem in {
        "credential",
        "credentials",
        "private_key",
        "secret",
        "secrets",
        "token",
        "tokens",
    }


def _contains_symlink(root: Path, supplied_path: Path) -> bool:
    current = root
    for part in supplied_path.parts:
        if part in {"", "."}:
            continue
        current /= part
        if current.is_symlink():
            return True
    return False


def resolve_path(file_path: str) -> Path:
    if not isinstance(file_path, str) or not file_path.strip():
        raise WorkspacePathError("File path must not be blank")
    root = _workspace_root()
    supplied = Path(file_path.strip())
    candidate = (root / supplied).resolve()
    if any(
        str(candidate) == prefix or str(candidate).startswith(f"{prefix}/")
        for prefix in SYSTEM_PATH_PREFIXES
    ):
        raise WorkspacePathError("System paths are not allowed")
    try:
        candidate.relative_to(root)
    except ValueError as exc:
        raise WorkspacePathError("File path is outside the workspace") from exc
    relative = candidate.relative_to(root)
    if is_sensitive_relative_path(relative):
        raise WorkspacePathError("Sensitive files are not allowed")
    if _contains_symlink(root, supplied):
        raise WorkspacePathError("Symlink paths are not allowed")
    return candidate


def resolve_indexable_file(relative_path: str | Path) -> Path:
    """Resolve a regular, non-symlink file using the shared workspace policy."""
    path = resolve_path(Path(relative_path).as_posix())
    if not path.exists() or not path.is_file():
        raise WorkspaceFileError("Workspace file does not exist")
    return path


def _hash_content(content: str) -> str:
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def _read_path(path: Path) -> str:
    if not path.exists() or not path.is_file():
        raise WorkspaceFileError("Workspace file does not exist")
    if path.stat().st_size > MAX_WORKSPACE_FILE_BYTES:
        raise WorkspaceFileError("Workspace file exceeds the size limit")
    try:
        content = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        raise WorkspaceFileError("Workspace file is unavailable or not text") from exc
    if "\x00" in content:
        raise WorkspaceFileError("Binary workspace files are not supported")
    return content


def read_file(file_path: str) -> dict[str, str | int]:
    path = resolve_path(file_path)
    content = _read_path(path)
    return {"path": file_path, "content": content, "size": len(content.encode("utf-8"))}


def _record_history(
    proposal: ChangeProposal,
    status: str,
    modified_hash: str | None = None,
    backup_path: str | None = None,
) -> None:
    with SessionLocal() as session:
        session.add(
            ChangeHistory(
                proposal_id=proposal.proposal_id,
                file_path=proposal.file_path,
                original_hash=proposal.original_hash,
                modified_hash=modified_hash,
                status=status,
                backup_path=backup_path,
            )
        )
        session.commit()


def propose_change(file_path: str, proposed_content: str) -> dict[str, str]:
    if not isinstance(proposed_content, str) or len(proposed_content) > MAX_CHANGE_CONTENT_CHARS:
        raise WorkspaceFileError("Proposed content exceeds the size limit")
    path = resolve_path(file_path)
    original = _read_path(path)
    proposal_id = secrets.token_urlsafe(24)
    proposal = ChangeProposal(
        proposal_id=proposal_id,
        file_path=file_path,
        original_hash=_hash_content(original),
        proposed_hash=_hash_content(proposed_content),
        diff="".join(
            difflib.unified_diff(
                original.splitlines(keepends=True),
                proposed_content.splitlines(keepends=True),
                fromfile=file_path,
                tofile=file_path,
            )
        ),
        proposed_content=proposed_content,
    )
    _proposals[proposal_id] = proposal
    _record_history(proposal, "proposed")
    return {
        "proposal_id": proposal.proposal_id,
        "file_path": proposal.file_path,
        "original_hash": proposal.original_hash,
        "proposed_hash": proposal.proposed_hash,
        "diff": proposal.diff,
        "status": "proposed",
    }


def apply_change(proposal_id: str, confirmation: bool) -> dict[str, str]:
    if confirmation is not True:
        raise ProposalError("Explicit confirmation=true is required")
    proposal = _proposals.get(proposal_id)
    if proposal is None:
        raise ProposalError("Proposal is invalid or expired")
    path = resolve_path(proposal.file_path)
    current = _read_path(path)
    if _hash_content(current) != proposal.original_hash:
        raise ProposalError("Workspace file changed since the proposal was created")

    root = _workspace_root()
    backup_dir = root / BACKUP_DIRECTORY
    backup_dir.mkdir(exist_ok=True)
    backup_name = f"{datetime.now(UTC).strftime('%Y%m%dT%H%M%S%fZ')}-{secrets.token_hex(8)}.bak"
    backup_path = backup_dir / backup_name
    backup_path.write_text(current, encoding="utf-8")
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=path.parent, delete=False
        ) as temporary:
            temporary.write(proposal.proposed_content)
            temporary_path = Path(temporary.name)
        temporary_path.replace(path)
    except OSError as exc:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink(missing_ok=True)
        raise WorkspaceFileError("Workspace file could not be updated") from exc

    modified_hash = _hash_content(proposal.proposed_content)
    _record_history(proposal, "applied", modified_hash, str(backup_path.relative_to(root)))
    del _proposals[proposal_id]
    return {
        "proposal_id": proposal_id,
        "file_path": proposal.file_path,
        "original_hash": proposal.original_hash,
        "modified_hash": modified_hash,
        "backup_path": str(backup_path.relative_to(root)),
        "status": "applied",
    }
