import logging
import pandas as pd
from sqlalchemy import create_engine, text
from sqlalchemy.exc import SQLAlchemyError

from config.settings import DB_CONFIG, DB_TABLE

logger = logging.getLogger(__name__)


def _build_engine():
    """Build a SQLAlchemy engine from the DB_CONFIG dict."""
    host     = DB_CONFIG["host"]
    user     = DB_CONFIG["user"]
    password = DB_CONFIG["password"]
    database = DB_CONFIG["database"]
    url = f"mysql+mysqlconnector://{user}:{password}@{host}/{database}"
    return create_engine(url, pool_pre_ping=True)


def fetch_incident_data() -> pd.DataFrame:
    """
    Fetch the full incident prediction dataset from MySQL via SQLAlchemy.

    Returns
    -------
    pd.DataFrame
        Raw incident data with all columns.
    """
    try:
        engine = _build_engine()
        with engine.connect() as conn:
            logger.info("Database connection established successfully.")
            df = pd.read_sql(text(f"SELECT * FROM {DB_TABLE}"), conn)
        logger.info("Fetched %d records from table '%s'.", len(df), DB_TABLE)
        return df
    except SQLAlchemyError as exc:
        logger.error("Database operation failed: %s", exc)
        raise
