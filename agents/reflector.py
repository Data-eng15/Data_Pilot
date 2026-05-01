"""
Reflector Agent — evaluates results, diagnoses problems, and decides replanning.

Analysis layers (in order of application)
──────────────────────────────────────────
1. Baseline lift      — is the best model actually better than a dummy?
2. Model spread       — are all models performing similarly? (no diversity)
3. F1 / balanced-acc  — absolute performance thresholds
4. Imbalance check    — imbalance-specific advice
5. Missing data check — still-problematic missing values
6. Per-class gap      — detect minority-class collapse via class report
7. Overfitting probe  — train/test gap (when CV scores are provided)
8. Replan decision    — composite gate on severity of issues
"""

from __future__ import annotations

import statistics
from typing import Any, Dict, List, Optional, Tuple


# ── Thresholds ────────────────────────────────────────────────────────────────
_MIN_LIFT_OVER_DUMMY   = 0.05   # balanced accuracy must beat dummy by this much
_F1_POOR               = 0.50
_F1_ACCEPTABLE         = 0.65
_BAL_ACC_POOR          = 0.55
_MODEL_SPREAD_TIGHT    = 0.03   # all models within 3 pp → no diversity
_OVERFIT_GAP           = 0.10   # train − test > 10 pp → overfitting
_SEVERITY_REPLAN       = 2      # ≥N high-severity issues → recommend replan


def reflect(
    dataset_profile: Dict[str, Any],
    evaluation: Dict[str, Any],
    all_metrics: List[Dict[str, Any]],
    cv_scores: Optional[Dict[str, float]] = None,          # optional CV results
    classification_report_str: Optional[str] = None,       # optional full CR
) -> Dict[str, Any]:
    """
    Analyse training results and produce an actionable reflection.

    Returns
    -------
    {
        "status":             "ok" | "needs_attention",
        "best_model":         str,
        "severity":           int,          # 0-3 (0=fine, 3=critical)
        "issues":             List[str],
        "suggestions":        List[str],
        "diagnostics":        Dict[str, Any],
        "replan_recommended": bool,
    }
    """
    best_model  = evaluation.get("model", "Unknown")
    bal_acc     = float(evaluation.get("balanced_accuracy", 0.0))
    f1_macro    = float(evaluation.get("f1_macro", 0.0))
    accuracy    = float(evaluation.get("accuracy", 0.0))
    imb         = float(dataset_profile.get("imbalance_ratio") or 1.0)
    rows        = dataset_profile.get("shape", {}).get("rows", 0)
    miss        = dataset_profile.get("missing_pct", {})
    max_missing = max(miss.values(), default=0.0)

    issues: List[str]      = []
    suggestions: List[str] = []
    diagnostics: Dict[str, Any] = {}
    severity = 0  # cumulative severity score

    # ── 1. Baseline lift ──────────────────────────────────────────────────────
    dummy = next(
        (m for m in all_metrics if "Dummy" in m.get("model", "")), None
    )
    if dummy is not None:
        dummy_ba = float(dummy.get("balanced_accuracy", 0.0))
        lift = bal_acc - dummy_ba
        diagnostics["baseline_lift"] = round(lift, 4)
        if lift < _MIN_LIFT_OVER_DUMMY:
            issues.append(
                f"Best model only +{lift:.3f} balanced-acc above baseline — "
                "very weak signal."
            )
            suggestions.append(
                "Check for target leakage or verify the target column is correct. "
                "Consider richer feature engineering."
            )
            severity += 2

    # ── 2. Model spread ───────────────────────────────────────────────────────
    non_dummy = [m for m in all_metrics if "Dummy" not in m.get("model", "")]
    if len(non_dummy) >= 2:
        ba_scores = [float(m.get("balanced_accuracy", 0)) for m in non_dummy]
        spread = max(ba_scores) - min(ba_scores)
        diagnostics["model_spread"] = round(spread, 4)
        if spread < _MODEL_SPREAD_TIGHT:
            issues.append(
                f"All models within {spread:.3f} balanced-acc of each other — "
                "likely a data bottleneck, not a model choice issue."
            )
            suggestions.append(
                "Focus on feature engineering rather than trying more models."
            )
            severity += 1

    # ── 3. Absolute performance ───────────────────────────────────────────────
    diagnostics["f1_macro"]          = round(f1_macro, 4)
    diagnostics["balanced_accuracy"] = round(bal_acc, 4)

    if f1_macro < _F1_POOR:
        issues.append(f"Macro F1 {f1_macro:.3f} is poor (< {_F1_POOR}).")
        suggestions.append(
            "Try more expressive models (XGBoost, GBM), tune hyperparameters, "
            "or address data quality."
        )
        severity += 2
    elif f1_macro < _F1_ACCEPTABLE:
        issues.append(f"Macro F1 {f1_macro:.3f} is below acceptable threshold ({_F1_ACCEPTABLE}).")
        suggestions.append(
            "Light hyperparameter tuning (n_estimators, max_depth) may close the gap."
        )
        severity += 1

    if bal_acc < _BAL_ACC_POOR:
        issues.append(f"Balanced accuracy {bal_acc:.3f} is poor (< {_BAL_ACC_POOR}).")
        severity += 1

    # ── 4. Imbalance-specific advice ─────────────────────────────────────────
    if imb >= 10.0:
        suggestions.append(
            f"Severe imbalance ({imb:.1f}x): consider SMOTE oversampling, "
            "class_weight='balanced', and optimise the decision threshold on the ROC curve."
        )
    elif imb >= 3.0:
        suggestions.append(
            f"Moderate imbalance ({imb:.1f}x): ensure class_weight='balanced' is set "
            "and evaluate with balanced accuracy + macro F1 rather than raw accuracy."
        )

    # ── 5. Residual missing-data check ────────────────────────────────────────
    if max_missing > 20.0:
        issues.append(
            f"Max missing rate {max_missing:.1f}% — imputation may be noisy; "
            "consider indicator flags for missing columns."
        )
        suggestions.append(
            "Add binary missingness-indicator features alongside imputed values."
        )
        severity += 1

    # ── 6. Per-class collapse (from classification report string) ─────────────
    if classification_report_str:
        _check_per_class_collapse(
            classification_report_str, issues, suggestions, diagnostics
        )

    # ── 7. Overfitting probe (when CV scores available) ───────────────────────
    if cv_scores:
        train_score = cv_scores.get("train_balanced_accuracy")
        test_score  = cv_scores.get("test_balanced_accuracy", bal_acc)
        if train_score is not None:
            gap = float(train_score) - float(test_score)
            diagnostics["overfit_gap"] = round(gap, 4)
            if gap > _OVERFIT_GAP:
                issues.append(
                    f"Overfitting detected: train {train_score:.3f} vs test {test_score:.3f} "
                    f"(gap={gap:.3f})."
                )
                suggestions.append(
                    "Reduce model complexity, add regularisation (C, max_depth, min_samples_leaf), "
                    "or increase training data."
                )
                severity += 2

    # ── 8. Small-dataset warning ──────────────────────────────────────────────
    if rows < 500:
        issues.append(
            f"Only {rows} rows — results may be unstable; prefer CV over single split."
        )
        suggestions.append("Use stratified k-fold CV (k=5) for more reliable estimates.")
        severity += 1

    # ── Final status & replan decision ───────────────────────────────────────
    status             = "ok" if not issues else "needs_attention"
    replan_recommended = severity >= _SEVERITY_REPLAN

    # Deduplicate suggestions while preserving order
    seen: set = set()
    deduped_suggestions: List[str] = []
    for s in suggestions:
        key = s[:50]
        if key not in seen:
            seen.add(key)
            deduped_suggestions.append(s)

    return {
        "status":             status,
        "best_model":         best_model,
        "severity":           min(severity, 3),
        "issues":             issues,
        "suggestions":        deduped_suggestions,
        "diagnostics":        diagnostics,
        "replan_recommended": replan_recommended,
    }


def should_replan(reflection: Dict[str, Any]) -> bool:
    """True when the reflector recommends another training cycle."""
    return bool(reflection.get("replan_recommended", False))


def apply_replan_strategy(
    plan: List[str],
    dataset_profile: Dict[str, Any],
    reflection: Dict[str, Any],
) -> Tuple[List[str], Dict[str, Any]]:
    """
    Adjust the plan and profile in response to identified issues.

    Strategy mapping
    ────────────────
    severe low performance + no lift  → inject feature engineering + more models
    overfitting                        → inject regularisation step
    imbalance issues                   → promote SMOTE / threshold step
    data quality (missing)             → inject indicator-flag imputation
    """
    new_plan    = list(plan)
    new_profile = dict(dataset_profile)
    notes       = list(new_profile.get("notes", []))
    issues      = reflection.get("issues", [])
    diagnostics = reflection.get("diagnostics", {})

    lift  = diagnostics.get("baseline_lift", 1.0)
    gap   = diagnostics.get("overfit_gap", 0.0)
    f1    = diagnostics.get("f1_macro", 1.0)
    imb   = float(new_profile.get("imbalance_ratio") or 1.0)
    miss  = new_profile.get("missing_pct", {})
    max_m = max(miss.values(), default=0.0)

    # Inject steps before train_models if not already present
    def _ensure_before_train(step: str) -> None:
        idx = new_plan.index("train_models") if "train_models" in new_plan else len(new_plan)
        if step not in new_plan:
            new_plan.insert(idx, step)

    if gap > 0.10:
        _ensure_before_train("apply_regularization")
        notes.append("Replan: overfitting detected → added regularisation step.")

    if lift < 0.05 or f1 < 0.50:
        _ensure_before_train("apply_feature_engineering")
        notes.append("Replan: low performance → injecting feature engineering.")
        # Unlock XGBoost even on larger sets
        if "use_fast_models_only" in new_plan:
            new_plan.remove("use_fast_models_only")

    if imb >= 3.0 and "apply_severe_imbalance_strategy" not in new_plan:
        _ensure_before_train("apply_severe_imbalance_strategy")
        notes.append("Replan: imbalance → escalating to SMOTE + threshold tuning.")

    if max_m > 20.0 and "handle_severe_missing_data" not in new_plan:
        _ensure_before_train("handle_severe_missing_data")
        notes.append("Replan: high missing rate → adding indicator-flag imputation.")

    if "replan_attempt" not in new_plan:
        new_plan.append("replan_attempt")

    new_profile["notes"] = notes
    return new_plan, new_profile


# ── Internal helpers ──────────────────────────────────────────────────────────

def _check_per_class_collapse(
    report_str: str,
    issues: List[str],
    suggestions: List[str],
    diagnostics: Dict[str, Any],
) -> None:
    """
    Parse sklearn's classification_report string to detect classes where
    recall collapsed to 0 (minority class ignored by the model).
    """
    collapsed: List[str] = []
    for line in report_str.splitlines():
        parts = line.split()
        if len(parts) >= 4:
            label = parts[0]
            try:
                recall = float(parts[2])
            except ValueError:
                continue
            if recall == 0.0 and label not in ("accuracy", "macro", "weighted"):
                collapsed.append(label)

    if collapsed:
        diagnostics["collapsed_classes"] = collapsed
        issues.append(
            f"Zero recall for class(es) {collapsed} — model ignores minority class(es)."
        )
        suggestions.append(
            "Use class_weight='balanced' or SMOTE; tune the decision threshold "
            "to recover recall on under-represented classes."
        )
