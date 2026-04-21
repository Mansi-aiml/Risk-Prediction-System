import os
import sys
import logging

import pandas as pd

from config.settings import (
    DATE_COLUMN,
    INCIDENT_MODEL_PATH,
    SEVERITY_MODEL_PATH,
)
from data.db_connector import fetch_incident_data
from data.preprocessor import preprocess
from models.trainer import train_pipeline
from models.predictor import predict_future_risks
from models.ts_forecaster import run_ts_forecast

# ─── Logging setup ────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  [%(levelname)-8s]  %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

# ─── Risk-level colour map (ANSI) ─────────────────────────────────────────────
_RISK_COLOUR = {
    "CRITICAL": "\033[91m",  # bright red
    "HIGH":     "\033[93m",  # bright yellow
    "MEDIUM":   "\033[94m",  # bright blue
    "LOW":      "\033[92m",  # bright green
}
_RESET = "\033[0m"


# ─── UI helpers ───────────────────────────────────────────────────────────────

def _banner() -> None:
    print("\n" + "=" * 68)
    print("        INCIDENT PREDICTIVE ANALYSIS SYSTEM ")
    print("=" * 68)


def _section(title: str) -> None:
    print(f"\n  {'─' * 60}")
    print(f"  {title}")
    print(f"  {'─' * 60}")


def _coloured_risk(risk_level: str) -> str:
    colour = _RISK_COLOUR.get(risk_level, "")
    return f"{colour}{risk_level}{_RESET}"


def _print_report(result: dict) -> None:
    risk_display = _coloured_risk(result["risk_level"])

    print("\n" + "═" * 68)
    print(f"  Future Risk Predictions for {result['department']} Department")
    print(
        f"  (Forecast: {result['forecast_days']} days after last training date:"
        f" {result['last_training_date']})"
    )
    print("─" * 68)
    print(f"  {'Month':<22}: {result['month']}")
    print(f"  {'Season':<22}: {result['season']}")
    print(f"  {'Predicted Incident Type':<22}: {result['incident_type']}")
    print(f"  {'Probability':<22}: {result['probability']}")
    print(f"  {'Predicted Severity':<22}: {result['severity_type']}")
    print(f"  {'Risk Level':<22}: {risk_display}")
    print("─" * 68)
    print(f"  Recommended Action : {result['recommendation']}")
    print(f"  Warning            : {result['warning']}")
    print("═" * 68 + "\n")


def _print_ts_report(
    ts: dict,
    department: str,
    forecast_days: int,
    prediction: dict,
) -> None:
    """Append the Time-Series Forecast section after the existing risk report."""
    risk_display = _coloured_risk(prediction["risk_level"])

    # ── Time-Series Forecast block ────────────────────────────────────────────
    trend_colour = {
        "Increasing": "\033[91m",   # red
        "Stable":     "\033[93m",   # yellow
        "Decreasing": "\033[92m",   # green
    }
    tc = trend_colour.get(ts["trend"], "")

    print("─" * 68)
    print(f"  TIME-SERIES FORECAST  —  {department} Department  (Next {forecast_days} days)")
    print("─" * 68)
    print(f"  {'Predicted Incident Type':<28}: {prediction['incident_type']}")
    print(f"  {'Severity':<28}: {prediction['severity_type']}")
    print(f"  {'Risk Level':<28}: {risk_display}")
    print(f"  {'Probability':<28}: {prediction['probability']}")
    print()
    print(f"  {'Next ' + str(forecast_days) + ' days incidents':<28}: {ts['total_incidents']}")
    print(f"  {'Trend':<28}: {tc}{ts['trend']}{_RESET}")
    print(f"  {'High Risk Week':<28}: {ts['high_risk_week']}")

    # ── Expected Incident Distribution ────────────────────────────────────────
    print()
    print("  Expected Incident Distribution")
    print("  " + "─" * 34)
    if ts["distribution"]:
        for inc_type, count in ts["distribution"].items():
            bar = "█" * count
            print(f"    {inc_type:<28}: {count:>3}  {bar}")
    else:
        print("    (Insufficient distribution data)")

    # ── Recommendations & Warning ──────────────────────────────────────────────
    print()
    print(f"  Recommended Action : {prediction['recommendation']}")
    print(f"  Warning            : {prediction['warning']}")
    print("═" * 68 + "\n")


def _models_exist() -> bool:
    return os.path.exists(INCIDENT_MODEL_PATH) and os.path.exists(SEVERITY_MODEL_PATH)


def _prompt_retrain() -> bool:
    answer = input("\n  Trained models already exist. Retrain? (y/n) [n]: ").strip().lower()
    return answer == "y"


# ─── Main pipeline ────────────────────────────────────────────────────────────

def main() -> None:
    _banner()

    # ── Stage 1: Data ingestion ───────────────────────────────────────────────
    _section("Stage 1/4 — Fetching incident data from database")
    try:
        df_raw = fetch_incident_data()
    except Exception as exc:
        logger.error("Failed to fetch data: %s", exc)
        sys.exit(1)
    print(f"  Records loaded  : {len(df_raw):,}")
    print(f"  Columns present : {list(df_raw.columns)}")

    # ── Stage 2: Preprocessing ────────────────────────────────────────────────
    _section("Stage 2/4 — Preprocessing & feature engineering")
    try:
        df_processed, encoders = preprocess(df_raw, fit=True)
    except Exception as exc:
        logger.error("Preprocessing failed: %s", exc)
        sys.exit(1)

    last_training_date: pd.Timestamp = df_processed[DATE_COLUMN].max()
    print(f"  Processed rows        : {len(df_processed):,}")
    print(f"  Unique departments    : {df_processed['department_name'].nunique()}")
    print(f"  Unique incident types : {df_processed['incident_type'].nunique()}")
    print(f"  Unique severity types : {df_processed['severity_type'].nunique()}")
    print(f"  Last training date    : {last_training_date.date()}")

    # ── Stage 3: Model training ───────────────────────────────────────────────
    _section("Stage 3/4 — Model training (LightGBM)")
    if _models_exist() and not _prompt_retrain():
        print("\n  Using existing trained models.")
    else:
        try:
            train_pipeline(df_processed)
        except Exception as exc:
            logger.error("Model training failed: %s", exc)
            sys.exit(1)

    # ── Stage 4: Prediction ───────────────────────────────────────────────────
    _section("Stage 4/4 — Incident Risk Prediction")

    # Show available departments, companies, and plants for user reference
    available_depts    = sorted(df_processed["department_name"].unique())
    available_companies = sorted(df_processed["company_name"].unique())
    available_plants   = sorted(df_processed["plant_name"].unique())
    print(f"\n  Available departments:\n  {available_depts}\n")
    print(f"\n  Available companies:\n  {available_companies}\n")
    print(f"\n  Available plants:\n  {available_plants}\n")

    department    = input("  Enter Department   : ").strip()
    company       = input("  Enter Company      : ").strip()
    plant         = input("  Enter Plant        : ").strip()
    forecast_days = int(input("  Enter Forecast Days: ").strip())

    try:
        result = predict_future_risks(department, company, plant, forecast_days, last_training_date)
        _print_report(result)
    except ValueError as exc:
        print(f"\n  [ERROR] {exc}")
        sys.exit(1)
    except Exception as exc:
        logger.error("Prediction error: %s", exc)
        sys.exit(1)

    # ── Stage 5: Time-Series Frequency Forecast (additive) ───────────────────
    _section("Stage 5/5 — Time-Series Incident Frequency Forecast (LightGBM)")
    try:
        ts_result = run_ts_forecast(
            df=df_processed,
            department=department,
            forecast_days=forecast_days,
            last_training_date=last_training_date,
        )
        _print_ts_report(ts_result, department, forecast_days, result)
    except ValueError as exc:
        print(f"\n  [TS WARNING] {exc}")
        print("  Time-series forecast skipped — existing prediction remains valid.\n")
    except Exception as exc:
        logger.error("Time-series forecast error: %s", exc)
        print("  Time-series forecast encountered an error and was skipped.\n")


if __name__ == "__main__":
    main()
