import os
from contextlib import contextmanager
from typing import Generator
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker, Session
from chronorag.config import settings
from chronorag.database.models import Base

_engine = None
_SessionFactory = None


def get_engine():
    global _engine
    if _engine is None:
        url = settings.database_url
        if url.startswith("sqlite"):
            # Enable SQLite foreign keys & check_same_thread=False
            _engine = create_engine(
                url,
                connect_args={"check_same_thread": False},
                echo=settings.debug,
            )
            # Enable SQLite foreign key constraints
            from sqlalchemy import event
            @event.listens_for(_engine, "connect")
            def set_sqlite_pragma(dbapi_connection, connection_record):
                cursor = dbapi_connection.cursor()
                cursor.execute("PRAGMA foreign_keys=ON")
                cursor.close()
        else:
            _engine = create_engine(
                url,
                echo=settings.debug,
                pool_pre_ping=True,
            )
    return _engine


def get_session_factory():
    global _SessionFactory
    if _SessionFactory is None:
        _SessionFactory = sessionmaker(
            bind=get_engine(),
            autocommit=False,
            autoflush=False,
            expire_on_commit=False,
        )
    return _SessionFactory


@contextmanager
def get_db_session() -> Generator[Session, None, None]:
    """Context manager for database sessions with automatic commit/rollback."""
    factory = get_session_factory()
    session = factory()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def is_postgres() -> bool:
    engine = get_engine()
    return engine.dialect.name == "postgresql"


def init_db():
    """Initializes schema and tables."""
    engine = get_engine()
    Base.metadata.create_all(bind=engine)

    # Initialize full-text search table for SQLite if applicable
    if not is_postgres():
        with engine.begin() as conn:
            conn.execute(text("""
                CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
                    chunk_id UNINDEXED,
                    chunk_text,
                    tokenize='porter unicode61'
                );
            """))
            conn.execute(text("""
                CREATE VIRTUAL TABLE IF NOT EXISTS claims_fts USING fts5(
                    claim_id UNINDEXED,
                    subject_text,
                    predicate,
                    object_value,
                    tokenize='porter unicode61'
                );
            """))
    else:
        # PostgreSQL pgvector and FTS initialization
        with engine.begin() as conn:
            conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector;"))
            conn.execute(text("CREATE EXTENSION IF NOT EXISTS btree_gist;"))
