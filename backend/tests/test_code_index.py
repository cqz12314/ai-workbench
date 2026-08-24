import hashlib
import io
from dataclasses import replace
from types import SimpleNamespace

import pytest

from app.services import code_index
from app.services.code_vector_store import CodeSearchResult


class FakeCodeStore:
    def __init__(self) -> None:
        self.files: dict[str, str] = {}
        self.embedded: list[list] = []
        self.replaced: list[str] = []
        self.deleted: list[str] = []
        self.search_results: list[CodeSearchResult] = []

    def indexed_files(self, _workspace_id: str) -> dict[str, str]:
        return dict(self.files)

    def embed_chunks(self, chunks):
        self.embedded.append(chunks)
        return [[1.0] for _ in chunks]

    def replace_file(self, _workspace_id, relative_path, chunks, _embeddings):
        self.replaced.append(relative_path)
        self.files[relative_path] = chunks[0].file_hash
        return [str(index) for index in range(len(chunks))]

    def delete_file(self, _workspace_id: str, relative_path: str) -> None:
        self.deleted.append(relative_path)
        self.files.pop(relative_path, None)

    def search(self, _workspace_id: str, _query: str, _limit: int):
        return self.search_results


@pytest.fixture
def enabled_workspace(tmp_path, monkeypatch):
    monkeypatch.setattr(code_index.settings, "code_index_enabled", True)
    monkeypatch.setattr(code_index.settings, "workspace_enabled", True)
    monkeypatch.setattr(code_index.settings, "workspace_root", str(tmp_path))
    return tmp_path


def chunk_by(chunks, symbol_name: str):
    return [chunk for chunk in chunks if chunk.symbol_name == symbol_name]


def test_python_ast_chunks_module_symbols_nested_and_decorators() -> None:
    source = '''import os

VALUE = 1

def decorator(fn):
    return fn

@decorator
async def fetch():
    return 1

class Service:
    """Service docs."""

    def login(self):
        def nested():
            return True
        return nested()

    class Config:
        pass
'''
    digest = hashlib.sha256(source.encode()).hexdigest()

    chunks = code_index.chunk_python(source, "demo", "service.py", digest)

    assert chunk_by(chunks, "<module>")[0].symbol_type == "module"
    assert chunk_by(chunks, "decorator")[0].symbol_type == "function"
    fetch = chunk_by(chunks, "fetch")[0]
    assert fetch.symbol_type == "async_function"
    assert fetch.start_line == 8
    assert chunk_by(chunks, "Service")[0].symbol_type == "class"
    assert chunk_by(chunks, "Service.login")[0].symbol_type == "method"
    assert chunk_by(chunks, "Service.login.nested")[0].symbol_type == "function"
    assert chunk_by(chunks, "Service.Config")[0].symbol_type == "class"
    assert all(len(chunk.content) <= code_index.MAX_CHUNK_CHARS for chunk in chunks)


def test_python_syntax_error_uses_text_fallback() -> None:
    chunks = code_index.chunk_python("def broken(:\n", "demo", "bad.py", "hash")
    assert chunks[0].symbol_type == "text_block"


def test_python_symbols_take_priority_over_many_module_statements(monkeypatch) -> None:
    monkeypatch.setattr(code_index, "MAX_CHUNKS_PER_FILE", 5)
    statements = "\n".join(f"VALUE_{index} = {index}" for index in range(150))
    source = f"""{statements}

def important_function():
    return True

class ImportantClass:
    def important_method(self):
        return True
"""

    chunks = code_index.chunk_python(source, "demo", "crowded.py", "hash")

    names = {chunk.symbol_name for chunk in chunks}
    assert "important_function" in names
    assert "ImportantClass" in names
    assert "ImportantClass.important_method" in names
    assert len(chunks) <= 5


def test_nested_python_chunk_preserves_original_indentation() -> None:
    source = """def outer():
    def nested():
        class NestedClass:
            pass
        return NestedClass
    return nested()
"""

    chunks = code_index.chunk_python(source, "demo", "nested.py", "hash")

    nested = chunk_by(chunks, "outer.nested")[0]
    nested_class = chunk_by(chunks, "outer.nested.NestedClass")[0]
    assert nested.content.startswith("    def nested():")
    assert nested_class.content.startswith("        class NestedClass:")


@pytest.mark.parametrize(
    ("extension", "language", "source"),
    [
        ("js", "javascript", "export function login() { return true; }\n"),
        ("ts", "typescript", "export async function login(): Promise<boolean> { return true; }\n"),
        ("jsx", "jsx", "const Login = () => { return <button>Login</button>; };\n"),
        ("tsx", "tsx", "export const Login = (): JSX.Element => { return <div />; };\n"),
    ],
)
def test_javascript_family_uses_structured_or_safe_fallback(
    extension: str, language: str, source: str
) -> None:
    chunks = code_index.chunk_code(source, "demo", f"Login.{extension}", language, "hash")
    assert chunks
    assert chunks[0].language == language
    assert chunks[0].symbol_type in {"function", "text_block"}
    assert all(len(chunk.content) <= code_index.MAX_CHUNK_CHARS for chunk in chunks)


def test_js_brace_match_ignores_strings_templates_and_comments() -> None:
    source = """function render() {
  const a = "}";
  const b = `template }`;
  // }
  /* } */
  return true;
}
"""
    chunks = code_index.chunk_javascript(source, "demo", "app.js", "javascript", "hash")
    assert chunks[0].symbol_name == "render"
    assert chunks[0].end_line == 7


def test_concise_arrow_does_not_capture_a_later_unrelated_block() -> None:
    source = """const double = x => x * 2;
if (enabled) {
  run();
}
"""

    chunks = code_index.chunk_javascript(source, "demo", "app.ts", "typescript", "hash")

    assert not chunk_by(chunks, "double")
    text = "\n".join(chunk.content for chunk in chunks if chunk.symbol_type == "text_block")
    assert "const double = x => x * 2;" in text
    assert "if (enabled)" in text


def test_incomplete_function_does_not_capture_a_later_unrelated_block() -> None:
    source = """function missing()
if (enabled) {
  run();
}
"""

    chunks = code_index.chunk_javascript(source, "demo", "broken.js", "javascript", "hash")

    assert not chunk_by(chunks, "missing")
    residual = "\n".join(chunk.content for chunk in chunks)
    assert "function missing()" in residual
    assert "if (enabled)" in residual


def test_javascript_body_finder_skips_default_object_braces() -> None:
    source = "function configure(options = {}) { return options; }\n"

    chunks = code_index.chunk_javascript(source, "demo", "app.js", "javascript", "hash")

    function = chunk_by(chunks, "configure")[0]
    assert function.content == source.rstrip()


def test_partial_tsx_structure_keeps_uncovered_code_as_text_block() -> None:
    source = """interface Props {
  title: string;
}

export function Header() {
  return <h1>Title</h1>;
}

const concise = (value: number) => value * 2;
"""

    chunks = code_index.chunk_javascript(source, "demo", "Header.tsx", "tsx", "hash")

    assert chunk_by(chunks, "Header")
    residual = "\n".join(
        chunk.content for chunk in chunks if chunk.symbol_type == "text_block"
    )
    assert "interface Props" in residual
    assert "const concise" in residual
    assert "return <h1>Title</h1>;" not in residual


def test_disabled_flags_reject_index_and_search(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(code_index.settings, "code_index_enabled", False)
    monkeypatch.setattr(code_index.settings, "workspace_enabled", True)
    monkeypatch.setattr(code_index.settings, "workspace_root", str(tmp_path))
    with pytest.raises(code_index.CodeIndexDisabledError):
        code_index.index_codebase(FakeCodeStore())
    with pytest.raises(code_index.CodeIndexDisabledError):
        code_index.search_codebase("login", store=FakeCodeStore())

    monkeypatch.setattr(code_index.settings, "code_index_enabled", True)
    monkeypatch.setattr(code_index.settings, "workspace_enabled", False)
    with pytest.raises(code_index.CodeIndexDisabledError):
        code_index.index_codebase(FakeCodeStore())


def test_incremental_new_unchanged_modified_and_stale(enabled_workspace) -> None:
    path = enabled_workspace / "main.py"
    path.write_text("def first():\n    return 1\n", encoding="utf-8")
    store = FakeCodeStore()

    first = code_index.index_codebase(store)
    assert first["indexed"] == 1
    assert store.replaced == ["main.py"]

    second = code_index.index_codebase(store)
    assert second["skipped"] == 1
    assert len(store.embedded) == 1

    path.write_text("def second():\n    return 2\n", encoding="utf-8")
    new_file = enabled_workspace / "new.ts"
    new_file.write_text("export function added() { return 2; }\n", encoding="utf-8")
    third = code_index.index_codebase(store)
    assert third["reindexed"] == 1
    assert third["indexed"] == 1

    new_file.unlink()
    fourth = code_index.index_codebase(store)
    assert fourth["stale_removed"] == 1
    assert "new.ts" in store.deleted


def test_incomplete_scan_never_cleans_stale(enabled_workspace, monkeypatch) -> None:
    (enabled_workspace / "a.py").write_text("x = 1\n", encoding="utf-8")
    (enabled_workspace / "b.py").write_text("x = 2\n", encoding="utf-8")
    store = FakeCodeStore()
    store.files["deleted.py"] = "old"
    monkeypatch.setattr(code_index, "MAX_CANDIDATE_FILES", 1)

    result = code_index.index_codebase(store)

    assert result["limit_reached"] is True
    assert store.deleted == []


def test_candidate_limit_stops_scandir_early(enabled_workspace, monkeypatch) -> None:
    for index in range(10):
        (enabled_workspace / f"{index}.py").write_text("x = 1\n", encoding="utf-8")
    unvisited = enabled_workspace / "zzz-child"
    unvisited.mkdir()
    (unvisited / "never.py").write_text("x = 1\n", encoding="utf-8")
    original_scandir = code_index.os.scandir
    scanned_directories = []

    class CountingScandir:
        def __init__(self, path):
            scanned_directories.append(code_index.Path(path))
            self._iterator = original_scandir(path)

        def __enter__(self):
            self._iterator.__enter__()
            return self

        def __exit__(self, *args):
            return self._iterator.__exit__(*args)

        def __iter__(self):
            yield from self._iterator

    monkeypatch.setattr(code_index.os, "scandir", CountingScandir)
    monkeypatch.setattr(code_index, "MAX_CANDIDATE_FILES", 2)

    candidates, complete, _ignored = code_index._candidate_paths(enabled_workspace)

    assert len(candidates) == 2
    assert complete is False
    assert scanned_directories == [enabled_workspace]


def test_candidate_order_is_deterministic_across_scandir_orders(
    enabled_workspace, monkeypatch
) -> None:
    (enabled_workspace / "z.py").write_text("z = 1\n", encoding="utf-8")
    (enabled_workspace / "a.py").write_text("a = 1\n", encoding="utf-8")
    alpha = enabled_workspace / "alpha"
    omega = enabled_workspace / "omega"
    alpha.mkdir()
    omega.mkdir()
    (alpha / "b.py").write_text("b = 1\n", encoding="utf-8")
    (alpha / "c.py").write_text("c = 1\n", encoding="utf-8")
    (omega / "never.py").write_text("x = 1\n", encoding="utf-8")
    original_scandir = code_index.os.scandir

    class OrderedScandir:
        def __init__(self, path, reverse):
            with original_scandir(path) as entries:
                self._entries = list(entries)
            if reverse:
                self._entries.reverse()

        def __enter__(self):
            return iter(self._entries)

        def __exit__(self, *_args):
            return False

    monkeypatch.setattr(code_index, "MAX_CANDIDATE_FILES", 3)
    monkeypatch.setattr(
        code_index.os,
        "scandir",
        lambda path: OrderedScandir(path, reverse=False),
    )
    forward = code_index._candidate_paths(enabled_workspace)
    monkeypatch.setattr(
        code_index.os,
        "scandir",
        lambda path: OrderedScandir(path, reverse=True),
    )
    reversed_order = code_index._candidate_paths(enabled_workspace)

    assert forward == reversed_order
    assert forward[0] == [
        code_index.Path("a.py"),
        code_index.Path("alpha/b.py"),
        code_index.Path("z.py"),
    ]
    assert forward[1] is False


def test_character_budget_prevents_stale_cleanup(enabled_workspace, monkeypatch) -> None:
    (enabled_workspace / "a.py").write_text("value = 'too much'\n", encoding="utf-8")
    store = FakeCodeStore()
    store.files["deleted.py"] = "old"
    monkeypatch.setattr(code_index, "MAX_INDEX_TOTAL_CHARS", 4)

    result = code_index.index_codebase(store)

    assert result["limit_reached"] is True
    assert store.deleted == []


def test_index_failure_prevents_stale_cleanup(enabled_workspace, monkeypatch) -> None:
    (enabled_workspace / "a.py").write_text("value = 1\n", encoding="utf-8")
    store = FakeCodeStore()
    store.files["deleted.py"] = "old"
    monkeypatch.setattr(
        code_index,
        "_read_candidate",
        lambda _relative: (_ for _ in ()).throw(OSError("read failed")),
    )

    with pytest.raises(code_index.CodeIndexError):
        code_index.index_codebase(store)
    assert store.deleted == []


def test_chunk_count_and_size_are_hard_limited(monkeypatch) -> None:
    monkeypatch.setattr(code_index, "MAX_CHUNK_CHARS", 10)
    monkeypatch.setattr(code_index, "MAX_CHUNKS_PER_FILE", 3)

    chunks = code_index.chunk_text("line\n" * 100, "demo", "notes.md", "markdown", "hash")

    assert len(chunks) == 3
    assert all(len(chunk.content) <= 10 for chunk in chunks)


def test_ignored_sensitive_binary_large_directories_and_symlinks(
    enabled_workspace, monkeypatch
) -> None:
    (enabled_workspace / "safe.py").write_text("token_service = True\n", encoding="utf-8")
    (enabled_workspace / ".env").write_text("SECRET=x", encoding="utf-8")
    (enabled_workspace / "private.pem").write_text("KEY", encoding="utf-8")
    (enabled_workspace / "credentials.json").write_text("{}", encoding="utf-8")
    (enabled_workspace / "binary.py").write_bytes(b"a\x00b")
    (enabled_workspace / "large.py").write_text("x" * 100, encoding="utf-8")
    for directory in ("node_modules", ".git", ".venv"):
        target = enabled_workspace / directory
        target.mkdir()
        (target / "bad.py").write_text("bad = True", encoding="utf-8")
    outside = enabled_workspace.parent / "outside-index.py"
    outside.write_text("outside = True", encoding="utf-8")
    (enabled_workspace / "linked.py").symlink_to(outside)
    monkeypatch.setattr(code_index, "MAX_FILE_BYTES", 32)
    store = FakeCodeStore()

    code_index.index_codebase(store)

    assert store.replaced == ["safe.py"]


def test_actual_bytes_are_hard_limited_even_when_stat_is_stale(monkeypatch) -> None:
    monkeypatch.setattr(code_index, "MAX_FILE_BYTES", 8)
    raw = b"x" * 20
    read_sizes: list[int] = []

    class TrackingReader(io.BytesIO):
        def read(self, size=-1):
            value = super().read(size)
            read_sizes.append(len(value))
            return value

    class GrowingPath:
        def stat(self):
            return SimpleNamespace(st_size=1)

        def open(self, _mode):
            return TrackingReader(raw)

    monkeypatch.setattr(code_index, "resolve_indexable_file", lambda _relative: GrowingPath())

    assert code_index._read_candidate(code_index.Path("growing.py")) is None
    assert sum(read_sizes) == code_index.MAX_FILE_BYTES + 1
    read_sizes.clear()
    assert code_index._hash_candidate(code_index.Path("growing.py")) is None
    assert sum(read_sizes) == code_index.MAX_FILE_BYTES + 1


def test_search_applies_result_and_total_content_limits(enabled_workspace) -> None:
    store = FakeCodeStore()
    template = CodeSearchResult("a.py", "python", "a", "function", 1, 2, "x" * 10_000, 0.1)
    store.search_results = [replace(template, relative_path=f"{index}.py") for index in range(10)]

    results = code_index.search_codebase("login", limit=10, store=store)

    assert len(results) == 4
    assert all(len(result.content) <= code_index.MAX_RESULT_CHARS for result in results)
    assert sum(len(result.content) for result in results) <= code_index.MAX_SEARCH_TOTAL_CHARS
    with pytest.raises(code_index.CodeIndexError):
        code_index.search_codebase("login", limit=11, store=store)
