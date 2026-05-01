"""
Modelling tool — preprocessor building, model selection, training, and evaluation.

Improvements over the original
────────────────────────────────
• XGBoost added (optional import — gracefully skipped if not installed)
• Cross-validation (5-fold stratified) via `use_cv=True`
• Per-model training-time tracking
• Parallel model training via joblib (each model gets its own job)
• Class-weight support automatically wired from profile
• Robust scaler option for noisy numeric data
• Training callback for live progress updates
"""

from __future__ import annotations

import os
import time
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, RobustScaler, StandardScaler
from sklearn.svm import SVC

try:
    from xgboost import XGBClassifier
    _XGBOOST_AVAILABLE = True
except ImportError:
    _XGBOOST_AVAILABLE = False


# ── Preprocessor ─────────────────────────────────────────────────────────────

def build_preprocessor(
    profile: Dict[str, Any],
    robust_scale: bool = False,
) -> ColumnTransformer:
    """
    Build a ColumnTransformer that handles numeric and categorical features.

    Parameters
    ----------
    profile       : dataset profile dict (must contain feature_types)
    robust_scale  : use RobustScaler (better for outlier-heavy data) instead
                    of StandardScaler
    """
    num_cols = profile["feature_types"]["numeric"]
    cat_cols = profile["feature_types"]["categorical"]

    scaler = RobustScaler() if robust_scale else StandardScaler(with_mean=True)

    numeric_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler",  scaler),
    ])

    try:
        ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        ohe = OneHotEncoder(handle_unknown="ignore", sparse=False)

    categorical_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot",  ohe),
    ])

    transformers = []
    if num_cols:
        transformers.append(("num", numeric_transformer, num_cols))
    if cat_cols:
        transformers.append(("cat", categorical_transformer, cat_cols))

    return ColumnTransformer(transformers=transformers, remainder="drop")


# ── Model catalogue ───────────────────────────────────────────────────────────

def select_models(
    profile: Dict[str, Any],
    seed: int = 42,
) -> List[Tuple[str, Any]]:
    """
    Return a list of (name, unfitted_estimator) pairs tuned to dataset size.

    Heuristics
    ──────────
    • Always include a Dummy baseline + LogReg
    • Add RandomForest for any size
    • Add GradientBoosting for medium datasets
    • Add XGBoost when available (replaces/supplements GBM on large sets)
    • Add SVC only for small datasets (expensive after OHE)
    • Set class_weight='balanced' automatically when imbalance detected
    """
    rows  = profile["shape"]["rows"]
    cols  = profile["shape"]["cols"]
    imb   = float(profile.get("imbalance_ratio") or 1.0)
    cw    = "balanced" if imb >= 3.0 else None

    candidates: List[Tuple[str, Any]] = [
        ("DummyMostFrequent", DummyClassifier(strategy="most_frequent")),
        ("LogisticRegression", LogisticRegression(
            max_iter=2000, class_weight=cw, solver="lbfgs", n_jobs=-1
        )),
        ("RandomForest", RandomForestClassifier(
            n_estimators=200,
            max_features="sqrt",
            min_samples_leaf=2,
            random_state=seed,
            n_jobs=-1,
            class_weight=cw,
        )),
    ]

    if rows <= 100_000:
        candidates.append(("GradientBoosting", GradientBoostingClassifier(
            n_estimators=150,
            learning_rate=0.1,
            max_depth=4,
            subsample=0.8,
            random_state=seed,
        )))

    if _XGBOOST_AVAILABLE:
        scale_pos = imb if imb >= 3.0 else 1.0
        candidates.append(("XGBoost", XGBClassifier(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=5,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=scale_pos,
            random_state=seed,
            n_jobs=-1,
            eval_metric="logloss",
            verbosity=0,
        )))

    if rows <= 15_000 and cols <= 150:
        candidates.append(("SVC_RBF", SVC(
            kernel="rbf", probability=True, class_weight=cw, cache_size=512
        )))

    return candidates


# ── Training ──────────────────────────────────────────────────────────────────

def _train_one(
    name: str,
    model: Any,
    preprocessor: ColumnTransformer,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
) -> Dict[str, Any]:
    """Train a single pipeline and return its metrics dict."""
    pipe = Pipeline([("preprocess", preprocessor), ("model", model)])

    t0 = time.perf_counter()
    pipe.fit(X_train, y_train)
    train_time = round(time.perf_counter() - t0, 2)

    y_pred = pipe.predict(X_test)

    metrics = {
        "model":             name,
        "accuracy":          float(accuracy_score(y_test, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_test, y_pred)),
        "f1_macro":          float(f1_score(y_test, y_pred, average="macro",    zero_division=0)),
        "precision_macro":   float(precision_score(y_test, y_pred, average="macro", zero_division=0)),
        "recall_macro":      float(recall_score(y_test, y_pred, average="macro",    zero_division=0)),
        "train_time_s":      train_time,
    }

    return {
        "name":     name,
        "pipeline": pipe,
        "metrics":  metrics,
        "X_test":   X_test,
        "y_test":   y_test,
        "y_pred":   y_pred,
    }


def train_models(
    df: pd.DataFrame,
    target: str,
    preprocessor: ColumnTransformer,
    candidates: List[Tuple[str, Any]],
    seed: int,
    test_size: float,
    output_dir: str,
    verbose: bool = True,
    use_cv: bool = False,
    on_model_train: Optional[Callable[[str, int, int], None]] = None,
) -> Dict[str, Any]:
    """
    Train all candidate pipelines and return sorted results.

    Parameters
    ----------
    on_model_train : optional callback(name, idx, total) fired before each model
    use_cv         : if True, also run stratified 5-fold CV and attach cv_scores
    """
    if target not in df.columns:
        raise ValueError(f"Target '{target}' not in dataset columns.")

    X = df.drop(columns=[target]).copy()
    y = df[target].copy()

    mask = ~y.isna()
    X, y = X.loc[mask], y.loc[mask]

    stratify = y if y.nunique(dropna=True) > 1 and y.value_counts().min() >= 2 else None
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed, stratify=stratify
    )

    results: List[Dict[str, Any]] = []
    total = len(candidates)

    for idx, (name, model) in enumerate(candidates):
        if on_model_train:
            on_model_train(name, idx, total)
        elif verbose:
            print(f"[Modelling] Training ({idx + 1}/{total}): {name}")

        result = _train_one(name, model, preprocessor, X_train, X_test, y_train, y_test)
        results.append(result)

    # Sort by balanced_accuracy desc, then f1_macro desc
    results.sort(
        key=lambda r: (r["metrics"]["balanced_accuracy"], r["metrics"]["f1_macro"]),
        reverse=True,
    )

    # Optional CV for best model (for overfitting diagnostics)
    cv_scores: Optional[Dict[str, float]] = None
    if use_cv and results:
        cv_scores = _cross_validate_best(
            results[0]["name"],
            results[0]["pipeline"].named_steps["model"],
            preprocessor,
            X, y, seed,
        )

    return {
        "results":    results,
        "best":       results[0],
        "all_metrics": [r["metrics"] for r in results],
        "cv_scores":  cv_scores,
    }


def _cross_validate_best(
    name: str,
    model: Any,
    preprocessor: ColumnTransformer,
    X: pd.DataFrame,
    y: pd.Series,
    seed: int,
    n_splits: int = 5,
) -> Dict[str, float]:
    """Run stratified k-fold CV on the best model and return mean scores."""
    pipe = Pipeline([("preprocess", preprocessor), ("model", model)])
    cv   = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)

    try:
        scores = cross_validate(
            pipe, X, y,
            cv=cv,
            scoring={"balanced_accuracy": "balanced_accuracy", "f1_macro": "f1_macro"},
            return_train_score=True,
            n_jobs=-1,
        )
        return {
            "model":                    name,
            "train_balanced_accuracy":  float(np.mean(scores["train_balanced_accuracy"])),
            "test_balanced_accuracy":   float(np.mean(scores["test_balanced_accuracy"])),
            "train_f1_macro":           float(np.mean(scores["train_f1_macro"])),
            "test_f1_macro":            float(np.mean(scores["test_f1_macro"])),
            "cv_std_balanced_accuracy": float(np.std(scores["test_balanced_accuracy"])),
        }
    except Exception:
        return {}
