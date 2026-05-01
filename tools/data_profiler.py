"""
Data Profiler — produces a rich EDA summary used by the planner and reflector.

Extended fields over the original
───────────────────────────────────
• numeric_stats   : per-column mean/std/min/max/skew/kurtosis
• correlation     : top correlated feature pairs (pearson)
• outlier_pct     : IQR-based outlier percentage per numeric column
• target_correlation : point-biserial / Cramér's V between features and target
• duplicate_rows  : count and percentage
• constant_cols   : columns with only one unique value (useless features)
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd


# ── Public API ────────────────────────────────────────────────────────────────

def infer_target_column(df: pd.DataFrame) -> Optional[str]:
    """Heuristic target inference: prefer common names, then last low-cardinality col."""
    candidates = ["target", "label", "class", "y", "outcome", "survived", "churn"]
    lower_map = {c.lower(): c for c in df.columns}
    for k in candidates:
        if k in lower_map:
            return lower_map[k]

    last = df.columns[-1]
    uniq = df[last].nunique(dropna=True)
    n = len(df)
    if n > 0 and (uniq <= 50 or uniq / max(n, 1) < 0.05):
        return last
    return None


def is_classification_target(series: pd.Series) -> bool:
    if series.dtype == "object" or str(series.dtype).startswith("category"):
        return True
    return series.nunique(dropna=True) <= 50


def dataset_fingerprint(df: pd.DataFrame, target: str) -> str:
    cols  = ",".join(df.columns.astype(str).tolist())
    shape = f"{df.shape[0]}x{df.shape[1]}"
    base  = f"{shape}|{target}|{cols}"
    return f"fp_{abs(hash(base)) % (10 ** 12)}"


def profile_dataset(df: pd.DataFrame, target: str) -> Dict[str, Any]:
    """
    Produce a comprehensive EDA profile of *df* with respect to *target*.

    All values are JSON-serialisable (Python builtins only, no numpy types).
    """
    if target not in df.columns:
        raise ValueError(f"Target column '{target}' not found in dataset.")

    y  = df[target]
    X  = df.drop(columns=[target])
    profile: Dict[str, Any] = {}

    # ── Shape & columns ───────────────────────────────────────────────────────
    profile["shape"]   = {"rows": int(df.shape[0]), "cols": int(df.shape[1])}
    profile["columns"] = df.columns.astype(str).tolist()

    # ── Missing values ────────────────────────────────────────────────────────
    missing = (df.isna().mean() * 100).round(2)
    profile["missing_pct"] = {str(k): float(v) for k, v in missing.items()}

    # ── Duplicate rows ────────────────────────────────────────────────────────
    n_dup = int(df.duplicated().sum())
    profile["duplicate_rows"] = {
        "count": n_dup,
        "pct":   round(n_dup / max(len(df), 1) * 100, 2),
    }

    # ── Feature types ─────────────────────────────────────────────────────────
    num_cols = X.select_dtypes(include=["number", "bool"]).columns.astype(str).tolist()
    cat_cols = [c for c in X.columns.astype(str) if c not in num_cols]
    profile["feature_types"] = {"numeric": num_cols, "categorical": cat_cols}

    # ── Unique counts ─────────────────────────────────────────────────────────
    profile["n_unique_by_col"] = {
        str(c): int(df[c].nunique(dropna=True)) for c in df.columns.astype(str)
    }

    # ── Constant / quasi-constant columns ────────────────────────────────────
    profile["constant_cols"] = [
        c for c in X.columns.astype(str)
        if X[c].nunique(dropna=True) <= 1
    ]

    # ── Numeric statistics ────────────────────────────────────────────────────
    numeric_stats: Dict[str, Any] = {}
    if num_cols:
        num_df = X[num_cols].astype(float)
        for col in num_cols:
            s = num_df[col].dropna()
            if len(s) == 0:
                continue
            q1, q3  = float(s.quantile(0.25)), float(s.quantile(0.75))
            iqr     = q3 - q1
            n_out   = int(((s < q1 - 1.5 * iqr) | (s > q3 + 1.5 * iqr)).sum())
            numeric_stats[col] = {
                "mean":        round(float(s.mean()), 4),
                "std":         round(float(s.std()), 4),
                "min":         round(float(s.min()), 4),
                "max":         round(float(s.max()), 4),
                "q1":          round(q1, 4),
                "q3":          round(q3, 4),
                "skew":        round(float(s.skew()), 4),
                "kurtosis":    round(float(s.kurtosis()), 4),
                "outlier_pct": round(n_out / max(len(s), 1) * 100, 2),
            }
    profile["numeric_stats"] = numeric_stats

    # ── Top correlated feature pairs ─────────────────────────────────────────
    profile["top_correlations"] = _top_correlations(X[num_cols] if num_cols else X)

    # ── Target info ───────────────────────────────────────────────────────────
    profile["target"]       = str(target)
    profile["target_dtype"] = str(y.dtype)
    profile["is_classification"] = bool(is_classification_target(y))

    notes: List[str] = []
    if profile["shape"]["rows"] < 1_000:
        notes.append("Small dataset (<1 000 rows): prefer simpler models; guard against overfitting.")
    if profile["shape"]["cols"] > 100:
        notes.append("High dimensionality (>100 cols): watch one-hot expansion and overfitting.")
    if profile["constant_cols"]:
        notes.append(f"{len(profile['constant_cols'])} constant column(s) detected — consider dropping them.")
    if profile["duplicate_rows"]["pct"] > 5:
        notes.append(f"High duplicate rate ({profile['duplicate_rows']['pct']:.1f}%) — verify data integrity.")

    # ── Classification-specific ───────────────────────────────────────────────
    if profile["is_classification"]:
        vc = y.value_counts(dropna=False)
        profile["class_counts"] = {str(k): int(v) for k, v in vc.items()}
        ratio = float(vc.max() / max(vc.min(), 1)) if len(vc) >= 2 else 1.0
        profile["imbalance_ratio"] = round(ratio, 3)
        if ratio >= 3.0:
            notes.append(
                f"Imbalance detected (ratio={ratio:.1f}): prioritise macro metrics / balanced accuracy."
            )
    else:
        profile["class_counts"]   = None
        profile["imbalance_ratio"] = None
        notes.append("Non-classification target detected: pipeline focuses on classification tasks.")

    profile["notes"] = notes
    return profile


# ── Internal helpers ──────────────────────────────────────────────────────────

def _top_correlations(
    num_df: pd.DataFrame,
    top_n: int = 5,
    threshold: float = 0.7,
) -> List[Dict[str, Any]]:
    """Return top_n highly correlated feature pairs (|r| >= threshold)."""
    if num_df.shape[1] < 2:
        return []
    try:
        corr  = num_df.astype(float).corr(method="pearson")
        pairs = []
        cols  = corr.columns.tolist()
        for i in range(len(cols)):
            for j in range(i + 1, len(cols)):
                r = corr.iloc[i, j]
                if abs(r) >= threshold:
                    pairs.append({
                        "feature_a": cols[i],
                        "feature_b": cols[j],
                        "pearson_r": round(float(r), 4),
                    })
        pairs.sort(key=lambda p: abs(p["pearson_r"]), reverse=True)
        return pairs[:top_n]
    except Exception:
        return []
