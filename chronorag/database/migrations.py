import logging
from chronorag.database.connection import init_db, get_engine, is_postgres
from sqlalchemy import text

logger = logging.getLogger(__name__)


def run_migrations():
    """Initializes the database schema and indexes."""
    logger.info("Running database migrations...")
    init_db()
    logger.info("Database schema initialized successfully.")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    run_migrations()
