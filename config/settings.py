import os
from dotenv import load_dotenv

load_dotenv()

# ─── Database Configuration ───────────────────────────────────────────────────
DB_CONFIG = {
    "host": os.getenv("DB_HOST", "localhost"),
    "user": os.getenv("DB_USER", "root"),
    "password": os.getenv("DB_PASSWORD", ""),
    "database": os.getenv("DB_NAME", "wps_erp"),
}
DB_TABLE = os.getenv("DB_TABLE", "incident_prediction_dataset")

# ─── Column Names ─────────────────────────────────────────────────────────────
DATE_COLUMN = "incident_date"
TARGET_INCIDENT_TYPE = "incident_type"
TARGET_SEVERITY_TYPE = "severity_type"
DEPARTMENT_COLUMN = "department_name"
COMPANY_COLUMN    = "company_name"


# ─── Feature Columns Used for Training ────────────────────────────────────────
FEATURE_COLUMNS = [
    "month_num",
    "day_of_week",
    "day_of_year",
    "season_encoded",
    "department_name_encoded",
    "company_name_encoded",
    "is_weekend",
    "quarter",
]

# ─── Indian Climate Season Mapping ────────────────────────────────────────────
SEASON_MAP = {
    12: "Winter",       1: "Winter",       2: "Winter",
    3:  "Summer",       4: "Summer",       5: "Summer",
    6:  "Monsoon",      7: "Monsoon",      8: "Monsoon",      9: "Monsoon",
    10: "Post-Monsoon", 11: "Post-Monsoon",
}

SEASON_ENCODING = {
    "Winter": 0,
    "Summer": 1,
    "Monsoon": 2,
    "Post-Monsoon": 3,
}

# ─── Risk Level Classification ────────────────────────────────────────────────
RISK_LEVELS = {
    "CRITICAL": ["Fatal", "Fatality", "Critical", "Death"],
    "HIGH":     ["Major", "Severe", "Serious", "Hospitalized"],
    "MEDIUM":   ["Moderate", "Emergency", "Medical Treatment"],
    "LOW":      ["Minor", "Near Miss", "First Aid", "No Injury"],
}

# ─── Month Name Lookup ────────────────────────────────────────────────────────
MONTH_NAMES = {
    1: "January",   2: "February",  3: "March",     4: "April",
    5: "May",       6: "June",      7: "July",       8: "August",
    9: "September", 10: "October",  11: "November",  12: "December",
}

# ─── Model Artifacts ──────────────────────────────────────────────────────────
ARTIFACTS_DIR = "artifacts"
INCIDENT_MODEL_PATH  = os.path.join(ARTIFACTS_DIR, "incident_type_model.pkl")
SEVERITY_MODEL_PATH  = os.path.join(ARTIFACTS_DIR, "severity_type_model.pkl")
DEPT_ENCODER_PATH    = os.path.join(ARTIFACTS_DIR, "dept_encoder.pkl")
COMPANY_ENCODER_PATH = os.path.join(ARTIFACTS_DIR, "company_encoder.pkl")
INCIDENT_ENCODER_PATH = os.path.join(ARTIFACTS_DIR, "incident_encoder.pkl")
SEVERITY_ENCODER_PATH = os.path.join(ARTIFACTS_DIR, "severity_encoder.pkl")

# ─── Training Sample Cap ─────────────────────────────────────────────────────
# Stratified sample drawn from the full dataset before training.
# 50 K rows capture all temporal/department patterns without the 372 K overhead.
MAX_TRAIN_SAMPLES = 300000

# ─── Time-Series Forecasting Settings ────────────────────────────────────────
# Lag windows (days) and rolling-mean windows used as features.
TS_LAG_DAYS     = [1, 3, 7, 14]
TS_ROLL_WINDOWS = [7, 14, 30]
# Minimum days of department history required to fit the TS model.
TS_MIN_HISTORY  = 30
# Maximum incident types shown in the distribution table.
TS_TOP_N_TYPES  = 5
# Trend thresholds: % change between first-half / second-half of forecast.
TS_TREND_UP_THRESHOLD   = 0.10   # ≥10 % increase  → Increasing
TS_TREND_DOWN_THRESHOLD = -0.10  # ≥10 % decrease  → Decreasing

# LightGBM Regressor params for the time-series model.
# Uses standard gbdt (small per-dept series, ~730-1100 rows — GOSS not needed).
LGBM_TS_PARAMS = {
    "boosting_type":     "gbdt",
    "n_estimators":      200,
    "learning_rate":     0.05,
    "max_depth":         4,
    "num_leaves":        15,
    "min_child_samples": 5,
    "subsample":         0.8,
    "colsample_bytree":  0.8,
    "random_state":      42,
    "n_jobs":            -1,
    "verbose":           -1,
}

# ─── LightGBM Hyperparameters ─────────────────────────────────────────────────
# boosting_type="goss" — Gradient-based One-Side Sampling:
#   keeps all high-gradient instances, randomly drops low-gradient ones each
#   iteration, giving 3-4× speed-up over standard gbdt on large datasets.
LGBM_PARAMS = {
    "boosting_type":     "goss",
    "n_estimators":      100,
    "learning_rate":     0.1,
    "max_depth":         5,
    "num_leaves":        25,
    "min_child_samples": 20,
    "colsample_bytree":  0.8,
    "random_state":      42,
    "n_jobs":            -1,
    "verbose":           -1,
}
