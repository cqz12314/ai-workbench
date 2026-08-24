import asyncio

import pytest
from sqlalchemy import create_engine, select
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.db.base import Base
from app.models import ChangeHistory
from app.services import workspace
from app.services.tools import ToolExecutionError, execute_tool


@pytest.fixture
def workspace_context(tmp_path, monkeypatch):
    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    sessions = sessionmaker(bind=engine, autoflush=False, autocommit=False)
    Base.metadata.create_all(engine)
    monkeypatch.setattr(workspace.settings, "workspace_enabled", True)
    monkeypatch.setattr(workspace.settings, "workspace_root", str(tmp_path))
    monkeypatch.setattr(workspace, "SessionLocal", sessions)
    workspace._proposals.clear()
    (tmp_path / "main.py").write_text("print('old')\n", encoding="utf-8")
    yield tmp_path, sessions
    workspace._proposals.clear()
    Base.metadata.drop_all(engine)
    engine.dispose()


def test_read_file_and_path_traversal_are_restricted(workspace_context):
    root, _ = workspace_context
    result = workspace.read_file("main.py")
    assert result["content"] == "print('old')\n"
    with pytest.raises(workspace.WorkspacePathError):
        workspace.read_file("../outside.txt")
    with pytest.raises(workspace.WorkspacePathError):
        workspace.read_file(str(root.parent / "outside.txt"))


@pytest.mark.parametrize(
    "name",
    [
        ".env",
        ".env.local",
        ".ssh/config",
        ".git/config",
        "server.pem",
        "server.key",
        "server.p12",
        "server.crt",
        "server.cer",
        "id_rsa",
        "id_ed25519",
        "credentials.json",
        "private_key.json",
        "certificates/bundle.json",
    ],
)
def test_sensitive_files_are_rejected(workspace_context, name: str):
    with pytest.raises(workspace.WorkspacePathError):
        workspace.resolve_path(name)


def test_normal_source_file_with_token_in_name_is_allowed(workspace_context):
    assert workspace.resolve_path("token_service.py").name == "token_service.py"


def test_symlink_file_and_parent_are_rejected(workspace_context, tmp_path):
    root, _ = workspace_context
    outside = tmp_path.parent / "outside-workspace.txt"
    outside.write_text("secret", encoding="utf-8")
    (root / "linked.txt").symlink_to(outside)
    (root / "linked-dir").symlink_to(tmp_path.parent, target_is_directory=True)

    with pytest.raises(workspace.WorkspacePathError, match="outside|Symlink"):
        workspace.resolve_path("linked.txt")
    with pytest.raises(workspace.WorkspacePathError, match="outside|Symlink"):
        workspace.resolve_path("linked-dir/outside-workspace.txt")


def test_proposal_diff_and_change_history(workspace_context):
    root, sessions = workspace_context
    proposal = workspace.propose_change("main.py", "print('new')\n")
    assert proposal["status"] == "proposed"
    assert "-print('old')" in proposal["diff"]
    assert "+print('new')" in proposal["diff"]
    with sessions() as session:
        history = session.scalars(select(ChangeHistory)).all()
        assert history[0].proposal_id == proposal["proposal_id"]
        assert history[0].status == "proposed"

    result = workspace.apply_change(proposal["proposal_id"], confirmation=True)
    assert (root / "main.py").read_text(encoding="utf-8") == "print('new')\n"
    assert (root / result["backup_path"]).read_text(encoding="utf-8") == "print('old')\n"
    with sessions() as session:
        statuses = [row.status for row in session.scalars(select(ChangeHistory)).all()]
        assert statuses == ["proposed", "applied"]


def test_confirmation_and_hash_conflict_are_required(workspace_context):
    proposal = workspace.propose_change("main.py", "print('new')\n")
    with pytest.raises(workspace.ProposalError, match="confirmation"):
        workspace.apply_change(proposal["proposal_id"], confirmation=False)
    workspace.resolve_path("main.py").write_text("print('external')\n", encoding="utf-8")
    with pytest.raises(workspace.ProposalError, match="changed"):
        workspace.apply_change(proposal["proposal_id"], confirmation=True)


def test_workspace_tools_are_disabled_by_default(monkeypatch):
    monkeypatch.setattr(workspace.settings, "workspace_enabled", False)
    monkeypatch.setattr(workspace.settings, "workspace_root", None)
    with pytest.raises(ToolExecutionError, match="disabled"):
        asyncio.run(execute_tool("read_file", {"file_path": "main.py"}))


def test_system_root_cannot_be_configured_as_workspace(monkeypatch):
    monkeypatch.setattr(workspace.settings, "workspace_enabled", True)
    monkeypatch.setattr(workspace.settings, "workspace_root", "/")

    with pytest.raises(workspace.WorkspacePathError, match="System paths"):
        workspace.workspace_root()
