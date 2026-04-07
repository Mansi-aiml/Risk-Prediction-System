import logging
from datetime import timedelta

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

from config.settings import (
    DATE_COLUMN,
    FEATURE_COLUMNS,
    MONTH_NAMES,
    SEASON_ENCODING,
    SEASON_MAP,
    DEPT_ENCODER_PATH,
    INCIDENT_ENCODER_PATH,
    INCIDENT_MODEL_PATH,
    SEVERITY_ENCODER_PATH,
    SEVERITY_MODEL_PATH,
)
from utils.risk_analyzer import get_recommendation, get_risk_level, get_warning
from utils.llm_advisor import generate_llm_advice

logger = logging.getLogger(__name__)


# ─── Artifact loading ─────────────────────────────────────────────────────────

def _load_artifacts() -> tuple:
    """Load all persisted models and encoders from the artifacts directory."""
    logger.info("Loading model artifacts…")
    return (
        joblib.load(INCIDENT_MODEL_PATH),
        joblib.load(SEVERITY_MODEL_PATH),
        joblib.load(DEPT_ENCODER_PATH),
        joblib.load(INCIDENT_ENCODER_PATH),
        joblib.load(SEVERITY_ENCODER_PATH),
    )


# ─── Feature builder ──────────────────────────────────────────────────────────

def _build_future_feature_rows(
    last_date: pd.Timestamp,
    forecast_days: int,
    dept_encoded: int,
) -> pd.DataFrame:
    """
    Generate one feature row per day in the forecast window.

    The department encoding is replicated across every row so the model
    receives a consistent department context for each future date.
    """
    rows = []
    for offset in range(1, forecast_days + 1):
        future_date = last_date + timedelta(days=offset)
        month = future_date.month
        season_label = SEASON_MAP[month]
        rows.append(
            {
                "date":                    future_date,
                "month_num":               month,
                "day_of_week":             future_date.dayofweek,
                "day_of_year":             future_date.timetuple().tm_yday,
                "season_encoded":          SEASON_ENCODING[season_label],
                "department_name_encoded": dept_encoded,
                "is_weekend":              int(future_date.dayofweek >= 5),
                "quarter":                 (month - 1) // 3 + 1,
            }
        )
    return pd.DataFrame(rows)


# ─── Prediction helpers ───────────────────────────────────────────────────────

def _dominant_prediction(
    proba_matrix: np.ndarray,
    encoder: LabelEncoder,
) -> tuple[str, float]:
    """
    Average class probabilities across all forecast days and return the
    class label with the highest mean probability.

    Returns
    -------
    label       : Decoded string label of the dominant class.
    probability : Rounded probability value.
    """
    avg_proba = proba_matrix.mean(axis=0)
    top_idx = int(np.argmax(avg_proba))
    label = encoder.inverse_transform([top_idx])[0]
    probability = round(float(avg_proba[top_idx]), 2)
    return label, probability


def _forecast_midpoint_month(
    last_date: pd.Timestamp,
    forecast_days: int,
) -> int:
    """Return the calendar month at the midpoint of the forecast window."""
    midpoint = last_date + timedelta(days=forecast_days // 2 + 1)
    return midpoint.month


# ─── Public API ───────────────────────────────────────────────────────────────

def predict_future_risks(
    department: str,
    forecast_days: int,
    last_training_date: pd.Timestamp,
) -> dict:
    """
    Predict the dominant incident type and severity for a given department
    over the next *forecast_days* days following the last training date.

    Parameters
    ----------
    department          : Target department name (must exist in training data).
    forecast_days       : Number of days to forecast ahead.
    last_training_date  : Latest date present in the training dataset.

    Returns
    -------
    dict with keys:
        department, forecast_days, last_training_date, month, season,
        incident_type, probability, severity_type, risk_level,
        recommendation, warning
    """
    incident_model, severity_model, dept_enc, incident_enc, severity_enc = _load_artifacts()

    # Department validation
    known_depts = list(dept_enc.classes_)
    if department not in known_depts:
        raise ValueError(
            f"Department '{department}' was not seen during training.\n"
            f"  Available departments: {known_depts}"
        )

    dept_encoded = int(dept_enc.transform([department])[0])

    # Build future feature matrix — keep as DataFrame so feature names match
    # what the model stored at fit time (avoids sklearn UserWarning).
    future_df = _build_future_feature_rows(last_training_date, forecast_days, dept_encoded)
    X_future = future_df[FEATURE_COLUMNS]

    # Predict probabilities across forecast window
    incident_proba = incident_model.predict_proba(X_future)
    severity_proba = severity_model.predict_proba(X_future)

    # Resolve dominant predictions
    incident_type, probability = _dominant_prediction(incident_proba, incident_enc)
    severity_type, _           = _dominant_prediction(severity_proba, severity_enc)

    # Contextual metadata
    mid_month  = _forecast_midpoint_month(last_training_date, forecast_days)
    season     = SEASON_MAP[mid_month]
    month_name = MONTH_NAMES[mid_month]

    #risk_level     = get_risk_level(severity_type)
    #recommendation = get_recommendation(incident_type)
    #warning        = get_warning(incident_type, risk_level)
    risk_level = get_risk_level(severity_type)

   
    from utils.llm_advisor import generate_llm_advice

    warning, recommendation = generate_llm_advice(
    department,
    incident_type,
    severity_type,
    risk_level
    )

    logger.info(
        "Prediction → dept=%s | type=%s (%.2f) | severity=%s | risk=%s",
        department, incident_type, probability, severity_type, risk_level,
    )

    return {
        "department":         department,
        "forecast_days":      forecast_days,
        "last_training_date": last_training_date.strftime("%Y-%m-%d"),
        "month":              month_name,
        "season":             season,
        "incident_type":      incident_type,
        "probability":        probability,
        "severity_type":      severity_type,
        "risk_level":         risk_level,
       "recommendation":     recommendation,
        "warning":            warning,
        
    }
