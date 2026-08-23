from sqlalchemy import Engine, inspect, text

DEFAULT_CONVERSATION_TITLE = "新对话"
MAX_GENERATED_TITLE_LENGTH = 60


def migrate_database(engine: Engine) -> None:
    """Apply small, idempotent SQLite schema upgrades for existing installations."""
    inspector = inspect(engine)
    if "conversations" not in inspector.get_table_names():
        return

    columns = {column["name"] for column in inspector.get_columns("conversations")}
    if "title" in columns:
        return

    escaped_default = DEFAULT_CONVERSATION_TITLE.replace("'", "''")
    with engine.begin() as connection:
        connection.execute(
            text(
                "ALTER TABLE conversations "
                f"ADD COLUMN title VARCHAR(200) NOT NULL DEFAULT '{escaped_default}'"
            )
        )
        connection.execute(
            text(
                """
                UPDATE conversations
                SET title = COALESCE(
                    (
                        SELECT substr(trim(messages.content), 1, :title_length)
                        FROM messages
                        WHERE messages.conversation_id = conversations.id
                          AND messages.role = 'user'
                          AND trim(messages.content) <> ''
                        ORDER BY messages.id
                        LIMIT 1
                    ),
                    :default_title
                )
                """
            ),
            {
                "title_length": MAX_GENERATED_TITLE_LENGTH,
                "default_title": DEFAULT_CONVERSATION_TITLE,
            },
        )
