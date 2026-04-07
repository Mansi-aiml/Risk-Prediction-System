"""
Time-Series Incident Frequency Forecaster
==========================================

Granularity strategy
--------------------
1. **Daily**   — groupby normalized incident_date; used when ≥ TS_MIN_HISTORY
                  unique calendar dates exist for the department.
2. **Monthly** — groupby (year × incident_month from DB); fallback when daily
                  dates are sparse (common when the DB stores a reporting date
                  that many records share).
3. **Synthetic-monthly** — repeated annual cycle over 3 simulated years;
                  last resort when even months lack variety.

All three paths share the same feature-engineering and LightGBM-Regressor
training code.  Only the step size (1 day vs ~30 days) differs in the
recursive forecaster.
"""

import logging
import math
from collections import deque
from datetime import timedelta

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor

from config.settings import (
    DATE_COLUMN,
    DEPARTMENT_COLUMN,
    SEASON_ENCODING,
    SEASON_MAP,
    TARGET_INCIDENT_TYPE,
    TS_LAG_DAYS,
    TS_MIN_HISTORY,
    TS_ROLL_WINDOWS,
    TS_TOP_N_TYPES,
    TS_TREND_DOWN_THRESHOLD,
    TS_TREND_UP_THRESHOLD,
    LGBM_TS_PARAMS,
)

logger = logging.getLogger(__name__)

# ─── Canonical feature columns (train ↔ predict must be identical) ────────────
_TS_FEATURE_COLS: list[str] = (
    [f"lag_{l}"         for l in TS_LAG_DAYS]
    + [f"roll_mean_{w}" for w in TS_ROLL_WINDOWS]
    + ["month", "day_of_week", "day_of_year", "quarter", "is_weekend", "season_encoded"]
)

_MONTHLY_STEP_DAYS = 30   # approximate days per forecasting step in monthly mode
_MONTHLY_MIN_ROWS  = 3    # minimum monthly buckets to attempt TS training


# ─── Series builders ──────────────────────────────────────────────────────────

def _daily_series(dept_df: pd.DataFrame) -> pd.Series:
    """
    Aggregate incidents by normalized calendar date (time stripped).
    Returns a contiguous daily series with 0-filled gaps.
    """
    dates = dept_df[DATE_COLUMN].dt.normalize()           # strip time component
    counts = dept_df.groupby(dates).size().rename("count")
    counts.index = pd.DatetimeIndex(counts.index)
    full_rng = pd.date_range(counts.index.min(), counts.index.max(), freq="D")
    return counts.reindex(full_rng, fill_value=0)


def _monthly_series(dept_df: pd.DataFrame) -> pd.Series:
    """
    Aggregate incidents by (year × incident_month) to build a monthly series.

    Uses the DB's ``incident_month`` column when present (more reliable than
    incident_date.month when dates are clustered around a reporting date),
    otherwise falls back to incident_date.month.
    """
    df = dept_df.copy()
    df["_year"] = df[DATE_COLUMN].dt.year

    if "incident_month" in df.columns:
        df["_month"] = (
            pd.to_numeric(df["incident_month"], errors="coerce")
            .fillna(df[DATE_COLUMN].dt.month)
            .astype(int)
            .clip(1, 12)
        )
    else:
        df["_month"] = df[DATE_COLUMN].dt.month

    df["_ym"] = pd.to_datetime(
        df["_year"].astype(str) + "-" + df["_month"].astype(str).str.zfill(2) + "-01",
        errors="coerce",
    )
    df.dropna(subset=["_ym"], inplace=True)

    counts = df.groupby("_ym").size().rename("count")
    counts.index = pd.DatetimeIndex(counts.index)
    full_rng = pd.date_range(counts.index.min(), counts.index.max(), freq="MS")
    return counts.reindex(full_rng, fill_value=0)


def _synthetic_monthly_series(dept_df: pd.DataFrame) -> pd.Series:
    """
    Build a 3-year synthetic monthly series by repeating the department's
    observed monthly incident pattern.  Used when neither daily nor real
    monthly data provides enough rows to train the TS model.
    """
    df = dept_df.copy()

    if "incident_month" in df.columns:
        month_col = pd.to_numeric(df["incident_month"], errors="coerce").fillna(
            df[DATE_COLUMN].dt.month
        ).astype(int).clip(1, 12)
    else:
        month_col = df[DATE_COLUMN].dt.month

    month_pattern = month_col.value_counts().sort_index()
    base_year = int(df[DATE_COLUMN].dt.year.max()) - 2

    records = [
        {"_ym": pd.Timestamp(year=base_year + yr, month=m, day=1),
         "count": int(month_pattern.get(m, 1))}
        for yr in range(3)
        for m in range(1, 13)
    ]
    synth = pd.DataFrame(records).set_index("_ym")["count"]
    synth.index = pd.DatetimeIndex(synth.index)
    return synth


def _build_time_series(
    df: pd.DataFrame,
    department: str,
) -> tuple[pd.Series, str]:
    """
    Choose the best available granularity for the department.

    Returns
    -------
    (series, granularity)
        series      — DatetimeIndex, values = incident counts per period.
        granularity — ``'daily'`` or ``'monthly'``.
    """
    dept_df = df[df[DEPARTMENT_COLUMN] == department].copy()
    if dept_df.empty:
        raise ValueError(f"No records found for department '{department}'.")

    # ── 1. Daily ──────────────────────────────────────────────────────────────
    daily = _daily_series(dept_df)
    if len(daily) >= TS_MIN_HISTORY:
        logger.info("[TS] Using daily granularity (%d days).", len(daily))
        return daily, "daily"

    logger.info(
        "[TS] Only %d unique dates for '%s' — trying monthly granularity.",
        len(daily), department,
    )

    # ── 2. Monthly (real year × month) ────────────────────────────────────────
    monthly = _monthly_series(dept_df)
    if len(monthly) >= _MONTHLY_MIN_ROWS:
        logger.info("[TS] Using monthly granularity (%d months).", len(monthly))
        return monthly, "monthly"

    logger.info(
        "[TS] Only %d unique year-months — using synthetic cyclical pattern.",
        len(monthly),
    )

    # ── 3. Synthetic monthly pattern ──────────────────────────────────────────
    synth = _synthetic_monthly_series(dept_df)
    logger.info("[TS] Using synthetic monthly series (%d rows).", len(synth))
    return synth, "monthly"


# ─── Feature engineering ──────────────────────────────────────────────────────

def _build_feature_df(series: pd.Series) -> pd.DataFrame:
    """
    Convert any time series (daily or monthly) into a lag/rolling feature
    DataFrame ready for LightGBM training.

    Lag and rolling-mean features use ``shift(1)`` so they reflect values
    known *before* each period (no look-ahead leakage).
    """
    idx = pd.DatetimeIndex(series.index)
    fdf = pd.DataFrame({"count": series.values}, index=idx)

    for lag in TS_LAG_DAYS:
        fdf[f"lag_{lag}"] = fdf["count"].shift(lag)

    for window in TS_ROLL_WINDOWS:
        fdf[f"roll_mean_{window}"] = fdf["count"].shift(1).rolling(window).mean()

    fdf["month"]          = idx.month
    fdf["day_of_week"]    = idx.dayofweek
    fdf["day_of_year"]    = idx.dayofyear
    fdf["quarter"]        = idx.quarter
    fdf["is_weekend"]     = (idx.dayofweek >= 5).astype(int)
    fdf["season_encoded"] = (
        pd.Series(idx.month, index=idx)
        .map(SEASON_MAP)
        .map(SEASON_ENCODING)
        .values
    )

    fdf.dropna(inplace=True)
    return fdf


# ─── Model training ───────────────────────────────────────────────────────────

def _train_ts_model(feature_df: pd.DataFrame) -> LGBMRegressor:
    """Fit and return a LightGBM Regressor on the engineered feature matrix."""
    X = feature_df[_TS_FEATURE_COLS]
    y = feature_df["count"].values
    model = LGBMRegressor(**LGBM_TS_PARAMS)
    model.fit(X, y)
    return model


# ─── Recursive forecasting ────────────────────────────────────────────────────

def _make_feature_row(
    future_date: pd.Timestamp,
    buffer: deque,
) -> dict:
    """
    Build one feature dict for *future_date* using the rolling *buffer*
    of recent counts (real observations + previously predicted values).
    """
    buf = list(buffer)
    row: dict = {}

    for lag in TS_LAG_DAYS:
        row[f"lag_{lag}"] = buf[-lag] if lag <= len(buf) else 0.0

    for window in TS_ROLL_WINDOWS:
        slice_ = buf[-window:] if window <= len(buf) else buf
        row[f"roll_mean_{window}"] = float(np.mean(slice_)) if slice_ else 0.0

    row["month"]          = future_date.month
    row["day_of_week"]    = future_date.dayofweek
    row["day_of_year"]    = future_date.timetuple().tm_yday
    row["quarter"]        = (future_date.month - 1) // 3 + 1
    row["is_weekend"]     = int(future_date.dayofweek >= 5)
    row["season_encoded"] = SEASON_ENCODING[SEASON_MAP[future_date.month]]
    return row


def _recursive_forecast(
    model: LGBMRegressor,
    history: pd.Series,
    anchor_date: pd.Timestamp,
    forecast_steps: int,
    step_days: int = 1,
) -> list[float]:
    """
    Recursively predict *forecast_steps* steps starting from *anchor_date*.

    Parameters
    ----------
    step_days : 1 for daily forecasting, ~30 for monthly forecasting.
                Each predicted value feeds the lag/rolling buffer for
                subsequent steps.
    """
    max_lookback = max(max(TS_LAG_DAYS), max(TS_ROLL_WINDOWS))
    buffer: deque = deque(
        history.values[-max_lookback:].tolist(),
        maxlen=max_lookback,
    )

    predictions: list[float] = []
    for offset in range(1, forecast_steps + 1):
        future_date = anchor_date + timedelta(days=offset * step_days)
        row_dict = _make_feature_row(future_date, buffer)
        X_row = pd.DataFrame([row_dict])[_TS_FEATURE_COLS]
        pred = max(0.0, float(model.predict(X_row)[0]))
        predictions.append(pred)
        buffer.append(pred)

    return predictions


# ─── Analytics -------------------------------------------

def _detect_trend(predictions: list[float]) -> str:
    """
    Compare first-half vs second-half mean of the forecast to assign a
    directional trend label.
    """
    if len(predictions) < 2:
        return "Stable"
    mid         = len(predictions) // 2
    first_half  = np.mean(predictions[:mid])
    second_half = np.mean(predictions[mid:])
    pct_change  = (second_half - first_half) / (first_half + 1e-9)
    if pct_change >= TS_TREND_UP_THRESHOLD:
        return "Increasing"
    if pct_change <= TS_TREND_DOWN_THRESHOLD:
        return "Decreasing"
    return "Stable"


def _highest_risk_week(predictions: list[float]) -> str:
    """Return the 7-day block label (Week N) with the highest cumulative count."""
    week_totals: dict[int, float] = {}
    for i, val in enumerate(predictions):
        week_num = i // 7 + 1
        week_totals[week_num] = week_totals.get(week_num, 0.0) + val
    if not week_totals:
        return "Week 1"
    return f"Week {max(week_totals, key=lambda k: week_totals[k])}"


def _expected_distribution(
    df: pd.DataFrame,
    department: str,
    total_incidents: int,
) -> dict[str, int]:
    """
    Distribute *total_incidents* across the top incident types observed
    historically for *department*, using their proportional frequency.
    """
    dept_df = df[df[DEPARTMENT_COLUMN] == department]
    if dept_df.empty or total_incidents == 0:
        return {}

    proportions = dept_df[TARGET_INCIDENT_TYPE].value_counts(normalize=True)
    top = proportions.head(TS_TOP_N_TYPES)

    distribution: dict[str, int] = {}
    remaining = total_incidents
    for i, (inc_type, prop) in enumerate(top.items()):
        if i == len(top) - 1:
            distribution[inc_type] = max(0, remaining)
        else:
            count = round(prop * total_incidents)
            distribution[inc_type] = count
            remaining -= count

    return {k: v for k, v in distribution.items() if v > 0}


# ─── Public API ───────────────────────────────────────────────────────────────

def run_ts_forecast(
    df: pd.DataFrame,
    department: str,
    forecast_days: int,
    last_training_date: pd.Timestamp,
) -> dict:
    """
    End-to-end time-series incident frequency forecast for *department*.

    Automatically selects daily or monthly granularity based on data
    availability, trains a LightGBM Regressor, and forecasts the given
    number of days ahead using recursive prediction.

    Parameters
    ----------
    df                  : Preprocessed DataFrame (all departments, all dates).
    department          : Target department name.
    forecast_days       : Number of calendar days to predict ahead.
    last_training_date  : Latest date in the training dataset (for reference).

    Returns
    -------
    dict with keys:
        total_incidents, trend, high_risk_week, distribution, daily_forecasts
    """
    series, granularity = _build_time_series(df, department)
    anchor_date = series.index.max()

    logger.info(
        "[TS] Granularity=%s | History=%d periods | Anchor=%s",
        granularity, len(series), anchor_date.date(),
    )

    feature_df = _build_feature_df(series)

    if feature_df.empty:
        raise ValueError(
            f"Insufficient history for '{department}' even after granularity "
            f"fallback ({len(series)} periods). Cannot train TS model."
        )

    model = _train_ts_model(feature_df)

    # ── Forecast ──────────────────────────────────────────────────────────────
    if granularity == "daily":
        raw_preds    = _recursive_forecast(model, series, anchor_date, forecast_days, step_days=1)
        daily_preds  = raw_preds
        total_raw    = sum(raw_preds)

    else:  # monthly
        months_needed = max(1, math.ceil(forecast_days / 30.44))
        monthly_preds = _recursive_forecast(
            model, series, anchor_date, months_needed, step_days=_MONTHLY_STEP_DAYS
        )
        # Distribute monthly total evenly across forecast days
        monthly_total = sum(monthly_preds)
        daily_rate    = monthly_total / (months_needed * 30.44)
        daily_preds   = [round(daily_rate, 2)] * forecast_days
        total_raw     = daily_rate * forecast_days

    total_incidents = max(1, round(total_raw))

    trend          = _detect_trend(daily_preds)
    high_risk_week = _highest_risk_week(daily_preds)
    distribution   = _expected_distribution(df, department, total_incidents)

    logger.info(
        "[TS] → total=%d | trend=%s | peak=%s | granularity=%s",
        total_incidents, trend, high_risk_week, granularity,
    )

    return {
        "total_incidents": total_incidents,
        "trend":           trend,
        "high_risk_week":  high_risk_week,
        "distribution":    distribution,
        "daily_forecasts": [round(p, 2) for p in daily_preds],
        "granularity":     granularity,
    }
