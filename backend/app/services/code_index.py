import ast
import hashlib
import os
import re
from dataclasses import dataclass, replace
from pathlib import Path

from app.core.config import settings
from app.services.code_vector_store import (
    CodeSearchResult,
    CodeVectorChunk,
    CodeVectorStore,
    get_code_vector_store,
)
from app.services.vector_store import VectorStoreError
from app.services.workspace import (
    WorkspaceError,
    is_sensitive_relative_path,
    resolve_indexable_file,
    workspace_root,
)

MAX_CANDIDATE_FILES = 2_000
MAX_FILE_BYTES = 512 * 1024
MAX_CHUNK_CHARS = 4_000
MAX_CHUNKS_PER_FILE = 100
MAX_INDEX_TOTAL_CHARS = 2_000_000
DEFAULT_SEARCH_LIMIT = 5
MAX_SEARCH_LIMIT = 10
MAX_RESULT_CHARS = 4_000
MAX_SEARCH_TOTAL_CHARS = 16_000

SUPPORTED_EXTENSIONS = {
    ".py": "python",
    ".js": "javascript",
    ".jsx": "jsx",
    ".ts": "typescript",
    ".tsx": "tsx",
    ".json": "json",
    ".toml": "toml",
    ".yaml": "yaml",
    ".yml": "yaml",
    ".md": "markdown",
}
IGNORED_DIRECTORIES = {
    ".git",
    ".venv",
    "venv",
    "node_modules",
    "dist",
    "build",
    "__pycache__",
    "coverage",
    ".ai_workbench_backups",
}
IGNORED_PATHS = {("frontend", "dist"), ("backend", "data")}
JS_DECLARATION = re.compile(
    r"^\s*(?:export\s+(?:default\s+)?)?(?:async\s+)?"
    r"(?:(function|class)\s+([A-Za-z_$][\w$]*)|"
    r"(?:const|let|var)\s+([A-Za-z_$][\w$]*)[^=]*=\s*(?:async\s*)?(?:\([^)]*\)|[A-Za-z_$][\w$]*)\s*=>)"
)
JS_METHOD = re.compile(r"^\s*(?:static\s+)?(?:async\s+)?([A-Za-z_$][\w$]*)\s*\([^;{}]*\)\s*\{")


class CodeIndexError(RuntimeError):
    """Raised when the controlled code index cannot complete."""


class CodeIndexDisabledError(CodeIndexError):
    pass


@dataclass(frozen=True)
class CodeChunk:
    project: str
    relative_path: str
    language: str
    symbol_name: str
    symbol_type: str
    start_line: int
    end_line: int
    content: str
    file_hash: str
    chunk_index: int = 0


def _require_enabled() -> Path:
    if not settings.code_index_enabled:
        raise CodeIndexDisabledError("Code index is disabled")
    try:
        return workspace_root()
    except WorkspaceError as exc:
        raise CodeIndexDisabledError(str(exc)) from exc


def workspace_identifier(root: Path) -> str:
    return hashlib.sha256(str(root.resolve()).encode()).hexdigest()


def _start_line(node: ast.AST) -> int:
    starts = [int(getattr(node, "lineno", 1))]
    starts.extend(
        int(decorator.lineno)
        for decorator in getattr(node, "decorator_list", [])
        if hasattr(decorator, "lineno")
    )
    return min(starts)


def _bounded_parts(content: str, start_line: int) -> list[tuple[int, int, str]]:
    lines = content.splitlines(keepends=True)
    if not lines and content:
        lines = [content]
    parts: list[tuple[int, int, str]] = []
    current: list[str] = []
    current_chars = 0
    part_start = start_line
    line_number = start_line
    for line in lines:
        pieces = [
            line[index : index + MAX_CHUNK_CHARS] for index in range(0, len(line), MAX_CHUNK_CHARS)
        ]
        for piece_index, piece in enumerate(pieces or [line]):
            if current and current_chars + len(piece) > MAX_CHUNK_CHARS:
                parts.append(
                    (
                        part_start,
                        line_number - (1 if piece_index == 0 else 0),
                        "".join(current).rstrip(),
                    )
                )
                current = []
                current_chars = 0
                part_start = line_number
            current.append(piece)
            current_chars += len(piece)
            if len(piece) == MAX_CHUNK_CHARS and current_chars == MAX_CHUNK_CHARS:
                parts.append((part_start, line_number, "".join(current).rstrip()))
                current = []
                current_chars = 0
                part_start = line_number
        line_number += 1
    if current:
        parts.append((part_start, max(part_start, line_number - 1), "".join(current).rstrip()))
    return [part for part in parts if part[2].strip()]


def _append_bounded(
    chunks: list[CodeChunk],
    base: CodeChunk,
    content: str,
    start_line: int,
) -> None:
    for part_start, part_end, part in _bounded_parts(content, start_line):
        if len(chunks) >= MAX_CHUNKS_PER_FILE:
            return
        chunks.append(replace(base, start_line=part_start, end_line=part_end, content=part))


def _source_lines(source: str, start: int, end: int) -> str:
    return "".join(source.splitlines(keepends=True)[start - 1 : end])


def chunk_python(source: str, project: str, relative_path: str, file_hash: str) -> list[CodeChunk]:
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return chunk_text(source, project, relative_path, "python", file_hash, "text_block")

    chunks: list[CodeChunk] = []
    base_values = dict(
        project=project,
        relative_path=relative_path,
        language="python",
        file_hash=file_hash,
    )
    def visit(nodes: list[ast.stmt], parents: list[tuple[str, str]]) -> None:
        for node in nodes:
            if not isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            name = ".".join([parent[0] for parent in parents] + [node.name])
            inside_class = bool(parents and parents[-1][1] == "class")
            if isinstance(node, ast.ClassDef):
                symbol_type = "class"
                nested_nodes = node.body
                end = int(getattr(node, "end_lineno", node.lineno))
                first_nested = next(
                    (
                        child
                        for child in node.body
                        if isinstance(child, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef))
                    ),
                    None,
                )
                if first_nested is not None:
                    end = max(_start_line(node), _start_line(first_nested) - 1)
            else:
                nested_nodes = node.body
                if isinstance(node, ast.AsyncFunctionDef):
                    symbol_type = "async_method" if inside_class else "async_function"
                else:
                    symbol_type = "method" if inside_class else "function"
                end = int(getattr(node, "end_lineno", node.lineno))
            start = _start_line(node)
            content = _source_lines(source, start, end)
            _append_bounded(
                chunks,
                CodeChunk(
                    symbol_name=name,
                    symbol_type=symbol_type,
                    start_line=start,
                    end_line=end,
                    content="",
                    **base_values,
                ),
                content,
                start,
            )
            visit(
                nested_nodes,
                [*parents, (node.name, "class" if isinstance(node, ast.ClassDef) else "function")],
            )

    visit(tree.body, [])
    module_nodes = [
        node
        for node in tree.body
        if not isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef))
    ]
    if module_nodes and len(chunks) < MAX_CHUNKS_PER_FILE:
        start = min(int(node.lineno) for node in module_nodes)
        end = max(int(getattr(node, "end_lineno", node.lineno)) for node in module_nodes)
        module_content = "\n".join(
            _source_lines(
                source,
                int(node.lineno),
                int(getattr(node, "end_lineno", node.lineno)),
            ).rstrip()
            for node in module_nodes
        )
        _append_bounded(
            chunks,
            CodeChunk(
                symbol_name="<module>",
                symbol_type="module",
                start_line=start,
                end_line=end,
                content="",
                **base_values,
            ),
            module_content,
            start,
        )
    if not chunks:
        return chunk_text(source, project, relative_path, "python", file_hash, "module")
    return [replace(chunk, chunk_index=index) for index, chunk in enumerate(chunks)]


def _matching_brace(source: str, opening: int) -> int | None:
    depth = 0
    state = "code"
    escaped = False
    index = opening
    while index < len(source):
        char = source[index]
        following = source[index + 1] if index + 1 < len(source) else ""
        if state in {"single", "double", "template"}:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif (
                (state == "single" and char == "'")
                or (state == "double" and char == '"')
                or (state == "template" and char == "`")
            ):
                state = "code"
        elif state == "line_comment":
            if char == "\n":
                state = "code"
        elif state == "block_comment":
            if char == "*" and following == "/":
                state = "code"
                index += 1
        elif char == "/" and following == "/":
            state = "line_comment"
            index += 1
        elif char == "/" and following == "*":
            state = "block_comment"
            index += 1
        elif char == "'":
            state = "single"
        elif char == '"':
            state = "double"
        elif char == "`":
            state = "template"
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return index
        index += 1
    return None


def _skip_js_trivia(source: str, start: int) -> int:
    index = start
    while index < len(source):
        if source[index].isspace():
            index += 1
        elif source.startswith("//", index):
            newline = source.find("\n", index + 2)
            index = len(source) if newline < 0 else newline + 1
        elif source.startswith("/*", index):
            closing = source.find("*/", index + 2)
            index = len(source) if closing < 0 else closing + 2
        else:
            break
    return index


def _find_js_body(source: str, start: int, match: re.Match[str]) -> int | None:
    """Conservatively locate a declaration body without treating later braces as its body."""
    if match.group(3) is not None:
        candidate = _skip_js_trivia(source, start + match.end())
        return candidate if candidate < len(source) and source[candidate] == "{" else None

    if match.re is JS_METHOD:
        candidate = start + match.end() - 1
        return candidate if source[candidate] == "{" else None

    index = start + match.end()
    declaration_kind = match.group(1)
    parentheses = 0
    brackets = 0
    saw_parameters = False
    parameters_closed = False
    return_type = False
    state = "code"
    escaped = False
    while index < len(source):
        char = source[index]
        following = source[index + 1] if index + 1 < len(source) else ""
        if state in {"single", "double", "template"}:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif (
                (state == "single" and char == "'")
                or (state == "double" and char == '"')
                or (state == "template" and char == "`")
            ):
                state = "code"
        elif state == "line_comment":
            if char == "\n":
                state = "code"
        elif state == "block_comment":
            if char == "*" and following == "/":
                state = "code"
                index += 1
        elif char == "/" and following == "/":
            state = "line_comment"
            index += 1
        elif char == "/" and following == "*":
            state = "block_comment"
            index += 1
        elif char == "'":
            state = "single"
        elif char == '"':
            state = "double"
        elif char == "`":
            state = "template"
        elif char == "(":
            if parentheses == 0:
                saw_parameters = True
            parentheses += 1
        elif char == ")":
            parentheses = max(0, parentheses - 1)
            if saw_parameters and parentheses == 0:
                parameters_closed = True
        elif char == "[":
            brackets += 1
        elif char == "]":
            brackets = max(0, brackets - 1)
        elif char == ";" and parentheses == 0 and brackets == 0:
            return None
        elif (
            char == ":"
            and declaration_kind == "function"
            and parameters_closed
            and parentheses == 0
            and brackets == 0
        ):
            return_type = True
        elif char == "{" and parentheses == 0 and brackets == 0:
            closing = _matching_brace(source, index)
            if closing is None:
                return None
            following_code = _skip_js_trivia(source, closing + 1)
            if following_code < len(source) and source[following_code] == "{":
                index = following_code
                continue
            return index
        elif (
            declaration_kind == "function"
            and parameters_closed
            and not return_type
            and parentheses == 0
            and brackets == 0
            and not char.isspace()
        ):
            return None
        index += 1
    return None


def _merged_ranges(ranges: list[tuple[int, int]]) -> list[tuple[int, int]]:
    merged: list[list[int]] = []
    for start, end in sorted(ranges):
        if merged and start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    return [(start, end) for start, end in merged]


def chunk_javascript(
    source: str, project: str, relative_path: str, language: str, file_hash: str
) -> list[CodeChunk]:
    chunks: list[CodeChunk] = []
    covered_ranges: list[tuple[int, int]] = []
    offsets: list[int] = []
    position = 0
    for line in source.splitlines(keepends=True):
        offsets.append(position)
        position += len(line)
    lines = source.splitlines(keepends=True)
    for line_index, line in enumerate(lines):
        if len(chunks) >= MAX_CHUNKS_PER_FILE:
            break
        match = JS_DECLARATION.match(line)
        method_match = JS_METHOD.match(line) if not match else None
        if not match and not method_match:
            continue
        if match:
            kind, declared, arrow = match.groups()
            symbol_name = declared or arrow or "<anonymous>"
            symbol_type = "class" if kind == "class" else "function"
        else:
            symbol_name = method_match.group(1)
            if symbol_name in {"if", "for", "while", "switch", "catch"}:
                continue
            symbol_type = "method"
        absolute_start = offsets[line_index]
        opening = _find_js_body(source, absolute_start, match or method_match)
        if opening is None:
            continue
        closing = _matching_brace(source, opening)
        if closing is None:
            continue
        start_line = line_index + 1
        end_line = source.count("\n", 0, closing) + 1
        content = source[absolute_start : closing + 1]
        base = CodeChunk(
            project=project,
            relative_path=relative_path,
            language=language,
            symbol_name=symbol_name,
            symbol_type=symbol_type,
            start_line=start_line,
            end_line=end_line,
            content="",
            file_hash=file_hash,
        )
        chunks_before = len(chunks)
        _append_bounded(chunks, base, content, start_line)
        if len(chunks) > chunks_before:
            covered_ranges.append((absolute_start, closing + 1))
    residual_start = 0
    for covered_start, covered_end in [*_merged_ranges(covered_ranges), (len(source), len(source))]:
        if len(chunks) >= MAX_CHUNKS_PER_FILE:
            break
        residual = source[residual_start:covered_start]
        if residual.strip():
            start_line = source.count("\n", 0, residual_start) + 1
            _append_bounded(
                chunks,
                CodeChunk(
                    project=project,
                    relative_path=relative_path,
                    language=language,
                    symbol_name="<module>",
                    symbol_type="text_block",
                    start_line=start_line,
                    end_line=start_line + residual.count("\n"),
                    content="",
                    file_hash=file_hash,
                ),
                residual,
                start_line,
            )
        residual_start = max(residual_start, covered_end)
    if not chunks:
        return chunk_text(source, project, relative_path, language, file_hash, "text_block")
    return [replace(chunk, chunk_index=index) for index, chunk in enumerate(chunks)]


def chunk_text(
    source: str,
    project: str,
    relative_path: str,
    language: str,
    file_hash: str,
    symbol_type: str = "text_block",
) -> list[CodeChunk]:
    base = CodeChunk(
        project=project,
        relative_path=relative_path,
        language=language,
        symbol_name="<module>",
        symbol_type=symbol_type,
        start_line=1,
        end_line=max(1, source.count("\n") + 1),
        content="",
        file_hash=file_hash,
    )
    chunks: list[CodeChunk] = []
    _append_bounded(chunks, base, source, 1)
    return [replace(chunk, chunk_index=index) for index, chunk in enumerate(chunks)]


def chunk_code(
    source: str,
    project: str,
    relative_path: str,
    language: str,
    file_hash: str,
) -> list[CodeChunk]:
    if language == "python":
        return chunk_python(source, project, relative_path, file_hash)
    if language in {"javascript", "typescript", "jsx", "tsx"}:
        return chunk_javascript(source, project, relative_path, language, file_hash)
    return chunk_text(source, project, relative_path, language, file_hash)


def _ignored_directory(relative: Path) -> bool:
    parts = tuple(part.casefold() for part in relative.parts)
    return bool(parts and (parts[-1] in IGNORED_DIRECTORIES or parts in IGNORED_PATHS))


def _path_sort_key(path: Path) -> tuple[str, str]:
    value = path.as_posix()
    return value.casefold(), value


def _candidate_paths(root: Path) -> tuple[list[Path], bool, int]:
    candidates: list[Path] = []
    ignored = 0
    pending = [root]
    try:
        while pending:
            current_path = pending.pop()
            child_directories: list[Path] = []
            with os.scandir(current_path) as entries:
                ordered_entries = sorted(
                    entries,
                    key=lambda entry: (entry.name.casefold(), entry.name),
                )
                for entry in ordered_entries:
                    path = Path(entry.path)
                    relative = path.relative_to(root)
                    if entry.is_symlink():
                        ignored += 1
                    elif entry.is_dir(follow_symlinks=False):
                        if _ignored_directory(relative) or is_sensitive_relative_path(relative):
                            ignored += 1
                        else:
                            child_directories.append(path)
                    elif not entry.is_file(follow_symlinks=False):
                        ignored += 1
                    elif (
                        relative.suffix.casefold() not in SUPPORTED_EXTENSIONS
                        or is_sensitive_relative_path(relative)
                    ):
                        ignored += 1
                    elif len(candidates) >= MAX_CANDIDATE_FILES:
                        return sorted(candidates, key=_path_sort_key), False, ignored + 1
                    else:
                        candidates.append(relative)
            pending.extend(sorted(child_directories, key=_path_sort_key, reverse=True))
    except OSError:
        return sorted(candidates, key=_path_sort_key), False, ignored
    return sorted(candidates, key=_path_sort_key), True, ignored


def _read_candidate(relative: Path) -> tuple[str, str] | None:
    try:
        path = resolve_indexable_file(relative)
        if path.stat().st_size > MAX_FILE_BYTES:
            return None
        with path.open("rb") as source:
            raw = source.read(MAX_FILE_BYTES + 1)
        if len(raw) > MAX_FILE_BYTES:
            return None
        if b"\x00" in raw:
            return None
        file_hash = hashlib.sha256(raw).hexdigest()
        return file_hash, raw.decode("utf-8-sig")
    except UnicodeDecodeError:
        return None
    except WorkspaceError:
        return None
    except OSError as exc:
        raise CodeIndexError("Workspace file could not be read") from exc


def _hash_candidate(relative: Path) -> str | None:
    try:
        path = resolve_indexable_file(relative)
        if path.stat().st_size > MAX_FILE_BYTES:
            return None
        digest = hashlib.sha256()
        total_bytes = 0
        with path.open("rb") as source:
            while block := source.read(min(64 * 1024, MAX_FILE_BYTES + 1 - total_bytes)):
                total_bytes += len(block)
                if total_bytes > MAX_FILE_BYTES:
                    return None
                if b"\x00" in block:
                    return None
                digest.update(block)
    except WorkspaceError:
        return None
    except OSError as exc:
        raise CodeIndexError("Workspace file could not be hashed") from exc
    return digest.hexdigest()


def _vector_chunks(workspace_id: str, chunks: list[CodeChunk]) -> list[CodeVectorChunk]:
    return [CodeVectorChunk(workspace_id=workspace_id, **chunk.__dict__) for chunk in chunks]


def index_codebase(store: CodeVectorStore | None = None) -> dict[str, int | bool]:
    root = _require_enabled()
    active_store = store or get_code_vector_store()
    workspace_id = workspace_identifier(root)
    project = root.name
    try:
        indexed = active_store.indexed_files(workspace_id)
        candidates, scan_complete, ignored = _candidate_paths(root)
    except (OSError, VectorStoreError, WorkspaceError) as exc:
        raise CodeIndexError("Codebase scan failed") from exc

    seen: set[str] = set()
    stats = {
        "scanned": 0,
        "indexed": 0,
        "reindexed": 0,
        "skipped": 0,
        "ignored": ignored,
        "stale_removed": 0,
        "chunks_written": 0,
        "total_chars": 0,
        "limit_reached": not scan_complete,
    }
    try:
        for relative in candidates:
            relative_name = relative.as_posix()
            stats["scanned"] += 1
            file_hash = _hash_candidate(relative)
            if file_hash is None:
                stats["ignored"] += 1
                continue
            if indexed.get(relative_name) == file_hash:
                seen.add(relative_name)
                stats["skipped"] += 1
                continue
            payload = _read_candidate(relative)
            if payload is None:
                stats["ignored"] += 1
                continue
            confirmed_hash, source = payload
            if confirmed_hash != file_hash:
                scan_complete = False
                stats["limit_reached"] = True
                continue
            seen.add(relative_name)
            if stats["total_chars"] + len(source) > MAX_INDEX_TOTAL_CHARS:
                scan_complete = False
                stats["limit_reached"] = True
                break
            language = SUPPORTED_EXTENSIONS[relative.suffix.casefold()]
            chunks = chunk_code(source, project, relative_name, language, file_hash)
            vectors = _vector_chunks(workspace_id, chunks)
            embeddings = active_store.embed_chunks(vectors)
            active_store.replace_file(workspace_id, relative_name, vectors, embeddings)
            stats["total_chars"] += len(source)
            stats["chunks_written"] += len(chunks)
            if relative_name in indexed:
                stats["reindexed"] += 1
            else:
                stats["indexed"] += 1
    except (OSError, VectorStoreError, WorkspaceError) as exc:
        raise CodeIndexError("Codebase indexing failed; stale cleanup was skipped") from exc

    if scan_complete:
        try:
            for relative_name in sorted(set(indexed) - seen):
                active_store.delete_file(workspace_id, relative_name)
                stats["stale_removed"] += 1
        except VectorStoreError as exc:
            raise CodeIndexError("Stale code index cleanup failed") from exc
    return stats


def search_codebase(
    query: str,
    limit: int = DEFAULT_SEARCH_LIMIT,
    store: CodeVectorStore | None = None,
) -> list[CodeSearchResult]:
    root = _require_enabled()
    normalized = query.strip() if isinstance(query, str) else ""
    if not normalized or len(normalized) > 1_000:
        raise CodeIndexError("Code search query is invalid")
    if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= MAX_SEARCH_LIMIT:
        raise CodeIndexError("Code search limit must be between 1 and 10")
    try:
        results = (store or get_code_vector_store()).search(
            workspace_identifier(root), normalized, limit
        )
    except VectorStoreError as exc:
        raise CodeIndexError("Codebase search failed") from exc
    bounded: list[CodeSearchResult] = []
    remaining = MAX_SEARCH_TOTAL_CHARS
    for result in results[:limit]:
        allowed = min(MAX_RESULT_CHARS, remaining)
        if allowed <= 0:
            break
        content = result.content[:allowed]
        bounded.append(replace(result, content=content))
        remaining -= len(content)
    return bounded
