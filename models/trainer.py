import os
import logging
import joblib
import pandas as pd
import numpy as np
from lightgbm import LGBMClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, f1_score

from config.settings import (
    FEATURE_COLUMNS,
    TARGET_INCIDENT_TYPE,
    TARGET_SEVERITY_TYPE,
    LGBM_PARAMS,
    ARTIFACTS_DIR,
    INCIDENT_MODEL_PATH,
    SEVERITY_MODEL_PATH,
    MAX_TRAIN_SAMPLES,
)

logger = logging.getLogger(__name__)

# ─── Internal helpers ─────────────────────────────────────────────────────────

def _build_feature_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract the training feature matrix as a named DataFrame so LightGBM
    stores feature names during fit — eliminating the sklearn UserWarning
    about feature name mismatches at predict time.
    """
    return df[FEATURE_COLUMNS]


def _train_single_model(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    model_path: str,
    label: str,
) -> LGBMClassifier:
    """
    Fit one LGBMClassifier, persist it to disk, and return the fitted model.
    """
    os.makedirs(ARTIFACTS_DIR, exist_ok=True)
    model = LGBMClassifier(**LGBM_PARAMS)
    model.fit(X_train, y_train)
    joblib.dump(model, model_path)
    logger.info("[%s] Model saved → %s", label, model_path)
    return model


def _evaluate(
    model: LGBMClassifier,
    X_test: pd.DataFrame,
    y_test: np.ndarray,
    label: str,
) -> None:
    """Print accuracy, weighted F1, and a full classification report."""
    y_pred = model.predict(X_test)
    acc  = accuracy_score(y_test, y_pred)
    f1   = f1_score(y_test, y_pred, average="weighted", zero_division=0)

    print(f"\n  ┌─ [{label}] Evaluation ─────────────────────────────")
    print(f"  │  Accuracy  : {acc:.4f}")
    print(f"  │  F1 Score  : {f1:.4f}  (weighted)")
    print(f"  └────────────────────────────────────────────────────")
    print(classification_report(y_test, y_pred, zero_division=0))
    logger.info("[%s] Accuracy=%.4f  F1=%.4f", label, acc, f1)


# ─── Public API ───────────────────────────────────────────────────────────────

def _stratified_sample(
    df: pd.DataFrame,
    target_col: str,
    n_samples: int,
) -> pd.DataFrame:
    """
    Draw a stratified random sample from df, capped at n_samples.
    Each class retains its proportional share so rare incident types
    are not dropped.  Returns df unchanged if it is already small enough.
    """
    if len(df) <= n_samples:
        return df
    frac = n_samples / len(df)
    # Iterate groups explicitly — avoids the pandas FutureWarning about
    # groupby.apply operating on grouping columns.
    sampled = pd.concat(
        [g.sample(frac=frac, random_state=42)
         for _, g in df.groupby(target_col, group_keys=False)],
        ignore_index=True,
    )
    return sampled


def train_pipeline(
    df: pd.DataFrame,
    test_size: float = 0.20,
) -> tuple[LGBMClassifier, LGBMClassifier]:
    """
    Train two LightGBM classifiers:
      1. Incident Type  (multi-class)
      2. Severity Type  (multi-class)

    A stratified sample (MAX_TRAIN_SAMPLES rows) is drawn first so training
    stays fast regardless of dataset size.  An 80/20 holdout evaluates
    accuracy and weighted F1 on unseen data.

    Parameters
    ----------
    df        : Preprocessed DataFrame (output of data.preprocessor.preprocess).
    test_size : Fraction of the sample reserved for evaluation.

    Returns
    -------
    incident_model, severity_model : Fitted LGBMClassifier objects.
    """
    results = {}
    for target, path, label in [
        (f"{TARGET_INCIDENT_TYPE}_encoded", INCIDENT_MODEL_PATH, "Incident Type"),
        (f"{TARGET_SEVERITY_TYPE}_encoded", SEVERITY_MODEL_PATH, "Severity Type"),
    ]:
        print(f"\n  Training [{label}] model…")

        # Stratified sample — fast and representative
        df_sample = _stratified_sample(df, target, MAX_TRAIN_SAMPLES)
        logger.info(
            "[%s] Training on %d / %d rows (stratified sample).",
            label, len(df_sample), len(df),
        )
        print(f"  Sample size : {len(df_sample):,} / {len(df):,} rows")

        X = _build_feature_matrix(df_sample)
        y = df_sample[target].values

        X_tr, X_te, y_tr, y_te = train_test_split(
            X, y,
            test_size=test_size,
            random_state=42,
            stratify=y if len(np.unique(y)) > 1 else None,
        )

        model = _train_single_model(X_tr, y_tr, path, label)
        _evaluate(model, X_te, y_te, label)
        results[label] = model

    return results["Incident Type"], results["Severity Type"]
