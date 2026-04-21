# """Load and cache incident data for predictions (mirrors Streamlit `load_data`)."""

# from __future__ import annotations

# from functools import lru_cache

# import pandas as pd

# from config.settings import DATE_COLUMN
# from data.db_connector import fetch_incident_data
# from data.preprocessor import preprocess


# @lru_cache(maxsize=1)
# def get_processed_data() -> tuple[pd.DataFrame, pd.Timestamp]:
#     df_raw = fetch_incident_data()
#     df_processed, _encoders = preprocess(df_raw, fit=False)
#     last_training_date = df_processed[DATE_COLUMN].max()
#     return df_processed, last_training_date


# def list_departments() -> list[str]:
#     df_processed, _ = get_processed_data()
#     return sorted(df_processed["department_name"].unique().tolist())

# def list_companies() -> list[str]:
#     df_processed, _ = get_processed_data()
#     return sorted(df_processed["company_name"].unique().tolist())

from __future__ import annotations

from functools import lru_cache
import pandas as pd

from config.settings import DATE_COLUMN, DEPARTMENT_COLUMN, COMPANY_COLUMN, PLANT_COLUMN
from data.db_connector import fetch_incident_data
from data.preprocessor import preprocess


# ─────────────────────────────────────────────────────────────
# Load Data
# ─────────────────────────────────────────────────────────────
@lru_cache(maxsize=1)
def get_processed_data() -> tuple[pd.DataFrame, pd.Timestamp]:
    df_raw = fetch_incident_data()
    df_processed, _ = preprocess(df_raw, fit=False)
    last_training_date = df_processed[DATE_COLUMN].max()
    return df_processed, last_training_date


# ─────────────────────────────────────────────────────────────
# Dropdown Data
# ─────────────────────────────────────────────────────────────
def list_departments() -> list[str]:
    df_processed, _ = get_processed_data()
    return sorted(df_processed[DEPARTMENT_COLUMN].dropna().unique().tolist())


def list_companies() -> list[str]:
    df_processed, _ = get_processed_data()
    return sorted(df_processed[COMPANY_COLUMN].dropna().unique().tolist())


def list_plants() -> list[str]:
    df_processed, _ = get_processed_data()
    return sorted(df_processed[PLANT_COLUMN].dropna().unique().tolist())


# ─────────────────────────────────────────────────────────────
# Filtering Logic
# ─────────────────────────────────────────────────────────────
def filter_data(
    department: str | None = None,
    company: str | None = None,
    plant: str | None = None,
) -> pd.DataFrame:
    df_processed, _ = get_processed_data()
    df_filtered = df_processed.copy()

    if department:
        df_filtered = df_filtered[df_filtered[DEPARTMENT_COLUMN] == department]

    if company:
        df_filtered = df_filtered[df_filtered[COMPANY_COLUMN] == company]

    if plant:
        df_filtered = df_filtered[df_filtered[PLANT_COLUMN] == plant]

    return df_filtered


# ─────────────────────────────────────────────────────────────
# Effective Input Logic
# ─────────────────────────────────────────────────────────────
def resolve_effective_inputs(
    df_filtered: pd.DataFrame,
    department: str | None = None,
    company: str | None = None,
    plant: str | None = None,
) -> tuple[str, str, str]:

    if df_filtered.empty:
        raise ValueError("No data available for selected filters")

    if department is None:
        department = df_filtered[DEPARTMENT_COLUMN].value_counts().idxmax()

    if company is None:
        company = df_filtered[COMPANY_COLUMN].value_counts().idxmax()

    if plant is None:
        plant = df_filtered[PLANT_COLUMN].value_counts().idxmax()

    return department, company, plant