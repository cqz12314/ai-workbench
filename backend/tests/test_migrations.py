from sqlalchemy import create_engine, inspect, text

from app.db.migrations import migrate_database


def test_migration_adds_titles_without_losing_existing_data(tmp_path) -> None:
    database_path = tmp_path / "legacy.db"
    engine = create_engine(f"sqlite:///{database_path}")
    with engine.begin() as connection:
        connection.execute(
            text(
                """
                CREATE TABLE conversations (
                    id INTEGER NOT NULL PRIMARY KEY,
                    created_at DATETIME NOT NULL,
                    updated_at DATETIME NOT NULL
                )
                """
            )
        )
        connection.execute(
            text(
                """
                CREATE TABLE messages (
                    id INTEGER NOT NULL PRIMARY KEY,
                    conversation_id INTEGER NOT NULL,
                    role VARCHAR(16) NOT NULL,
                    content TEXT NOT NULL,
                    created_at DATETIME NOT NULL,
                    FOREIGN KEY(conversation_id) REFERENCES conversations (id) ON DELETE CASCADE
                )
                """
            )
        )
        connection.execute(
            text(
                """
                INSERT INTO conversations (id, created_at, updated_at)
                VALUES (1, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP),
                       (2, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)
                """
            )
        )
        connection.execute(
            text(
                """
                INSERT INTO messages (id, conversation_id, role, content, created_at)
                VALUES (1, 1, 'assistant', '欢迎', CURRENT_TIMESTAMP),
                       (2, 1, 'user', '保留下来的第一条用户消息', CURRENT_TIMESTAMP)
                """
            )
        )

    migrate_database(engine)
    migrate_database(engine)

    assert "title" in {column["name"] for column in inspect(engine).get_columns("conversations")}
    with engine.connect() as connection:
        conversations = connection.execute(
            text("SELECT id, title FROM conversations ORDER BY id")
        ).all()
        message_count = connection.scalar(text("SELECT COUNT(*) FROM messages"))

    assert conversations == [(1, "保留下来的第一条用户消息"), (2, "新对话")]
    assert message_count == 2
    engine.dispose()
