import os
import logging
import joblib
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

from config.settings import (
    DATE_COLUMN,
    DEPARTMENT_COLUMN,
    TARGET_INCIDENT_TYPE,
    TARGET_SEVERITY_TYPE,
    SEASON_MAP,
    SEASON_ENCODING,
    ARTIFACTS_DIR,
    DEPT_ENCODER_PATH,
    INCIDENT_ENCODER_PATH,
    SEVERITY_ENCODER_PATH,
)

logger = logging.getLogger(__name__)

# ─── Required columns ─────────────────────────────────────────────────────────
REQUIRED_COLUMNS = [
    DATE_COLUMN,
    DEPARTMENT_COLUMN,
    TARGET_INCIDENT_TYPE,
    TARGET_SEVERITY_TYPE,
]


def validate_columns(df: pd.DataFrame) -> None:
    """Raise ValueError if any required column is missing from the DataFrame."""
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Parse dates, drop rows with nulls in critical columns,
    remove duplicates, and strip whitespace from string fields.
    """
    df = df.copy()
    df[DATE_COLUMN] = pd.to_datetime(df[DATE_COLUMN], errors="coerce")

    before = len(df)
    df.dropna(subset=REQUIRED_COLUMNS, inplace=True)
    df.drop_duplicates(inplace=True)
    after = len(df)

    logger.info("Cleaned data: %d → %d rows (dropped %d).", before, after, before - after)

    for col in [DEPARTMENT_COLUMN, TARGET_INCIDENT_TYPE, TARGET_SEVERITY_TYPE]:
        df[col] = df[col].astype(str).str.strip()

    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Derive temporal and categorical features from the incident date.

    New columns added
    -----------------
    month_num       : calendar month (1-12)
    day_of_week     : weekday index Mon=0 … Sun=6
    day_of_year     : ordinal day within the year (1-366)
    quarter         : calendar quarter (1-4)
    is_weekend      : 1 if Sat/Sun, else 0
    season          : human-readable season label
    season_encoded  : integer-encoded season
    """
    df = df.copy()
    df["month_num"]    = df[DATE_COLUMN].dt.month
    df["day_of_week"]  = df[DATE_COLUMN].dt.dayofweek
    df["day_of_year"]  = df[DATE_COLUMN].dt.dayofyear
    df["quarter"]      = df[DATE_COLUMN].dt.quarter
    df["is_weekend"]   = (df["day_of_week"] >= 5).astype(int)
    df["season"]       = df["month_num"].map(SEASON_MAP)
    df["season_encoded"] = df["season"].map(SEASON_ENCODING)
    return df


def _fit_or_load_encoder(
    series: pd.Series,
    path: str,
    fit: bool,
) -> tuple[np.ndarray, LabelEncoder]:
    """Fit a new LabelEncoder or load a saved one, then transform the series."""
    le = LabelEncoder()
    if fit:
        encoded = le.fit_transform(series)
        joblib.dump(le, path)
        logger.debug("Saved encoder for '%s' → %s", series.name, path)
    else:
        le = joblib.load(path)
        encoded = le.transform(series)
    return encoded, le


def encode_labels(
    df: pd.DataFrame,
    fit: bool = True,
) -> tuple[pd.DataFrame, dict[str, LabelEncoder]]:
    """
    Label-encode department, incident type, and severity type columns.

    Parameters
    ----------
    df  : DataFrame with cleaned + engineered features.
    fit : True during training (fits + saves encoders),
          False during inference (loads saved encoders).

    Returns
    -------
    df        : DataFrame with additional *_encoded columns.
    encoders  : Dict mapping column name → fitted LabelEncoder.
    """
    os.makedirs(ARTIFACTS_DIR, exist_ok=True)
    encoders: dict[str, LabelEncoder] = {}

    encoding_targets = [
        (DEPARTMENT_COLUMN,    DEPT_ENCODER_PATH),
        (TARGET_INCIDENT_TYPE, INCIDENT_ENCODER_PATH),
        (TARGET_SEVERITY_TYPE, SEVERITY_ENCODER_PATH),
    ]

    for col, path in encoding_targets:
        encoded_vals, le = _fit_or_load_encoder(df[col].astype(str), path, fit)
        df[f"{col}_encoded"] = encoded_vals
        encoders[col] = le

    return df, encoders


def preprocess(
    df: pd.DataFrame,
    fit: bool = True,
) -> tuple[pd.DataFrame, dict[str, LabelEncoder]]:
    """
    Full preprocessing pipeline: validate → clean → engineer → encode.

    Parameters
    ----------
    df  : Raw DataFrame from the database.
    fit : Whether to fit and save new encoders (training mode).

    Returns
    -------
    df_processed : Feature-ready DataFrame.
    encoders     : Dict of fitted LabelEncoders.
    """
    logger.info("Starting preprocessing pipeline (fit=%s)…", fit)
    validate_columns(df)
    df = clean_data(df)
    df = engineer_features(df)
    df, encoders = encode_labels(df, fit=fit)
    logger.info("Preprocessing complete — final shape: %s.", df.shape)
    return df, encoders
