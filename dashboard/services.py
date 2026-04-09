"""Load and cache incident data for predictions (mirrors Streamlit `load_data`)."""

from __future__ import annotations

from functools import lru_cache

import pandas as pd

from config.settings import DATE_COLUMN
from data.db_connector import fetch_incident_data
from data.preprocessor import preprocess


@lru_cache(maxsize=1)
def get_processed_data() -> tuple[pd.DataFrame, pd.Timestamp]:
    df_raw = fetch_incident_data()
    df_processed, _encoders = preprocess(df_raw, fit=False)
    last_training_date = df_processed[DATE_COLUMN].max()
    return df_processed, last_training_date


def list_departments() -> list[str]:
    df_processed, _ = get_processed_data()
    return sorted(df_processed["department_name"].unique().tolist())
