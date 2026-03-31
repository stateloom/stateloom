"""Alembic environment for StateLoom schema migrations.

Uses raw SQL via op.execute() — no SQLAlchemy ORM models.
"""

from alembic import context

# No ORM metadata — all migrations use raw SQL via op.execute()
target_metadata = None


def run_migrations_online() -> None:
    """Run migrations using a live database connection."""
    from sqlalchemy import create_engine

    url = context.config.get_main_option("sqlalchemy.url") or ""
    connectable = create_engine(url)

    with connectable.connect() as connection:
        context.configure(connection=connection, target_metadata=target_metadata)
        with context.begin_transaction():
            context.run_migrations()

    connectable.dispose()


run_migrations_online()
