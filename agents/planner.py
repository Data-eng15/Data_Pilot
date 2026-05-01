"""
Planner Agent — generates a context-aware execution plan from a dataset profile.

Decision tree
─────────────
1. Dataset size   → governs which model families are viable
2. Imbalance      → triggers class-weight / threshold strategies
3. Missing data   → adds targeted imputation steps
4. Cardinality    → switches from OHE to target-encoding for large cats
5. Dimensionality → triggers feature-selection / PCA step
6. Memory hint    → promotes previously successful models to the front
"""

from typing import Any, Dict, List, Optional


# ── Thresholds ────────────────────────────────────────────────────────────────
_SMALL_ROWS      = 1_000
_MEDIUM_ROWS     = 30_000
_LARGE_ROWS      = 200_000
_HIGH_DIM_COLS   = 100
_HIGH_CARD_UNIQ  = 30     # unique values above which a cat col is "high-card"
_IMBALANCE_MOD   = 3.0    # moderate imbalance
_IMBALANCE_SEV   = 10.0   # severe imbalance
_MISSING_WARN    = 10.0   # % missing that warrants a note
_MISSING_SEV     = 40.0   # % missing that warrants a dedicated imputation step


def create_plan(
    dataset_profile: Dict[str, Any],
    memory_hint: Optional[Dict[str, Any]] = None,
) -> List[str]:
    """
    Generate a prioritised execution plan from a dataset profile.

    Returns a list of named steps (strings) that the orchestrator will execute
    in order.  Each step name matches a phase in the pipeline:
      profile_dataset | build_preprocessor | select_models | train_models |
      evaluate | reflect | write_report
    plus optional enrichment steps inserted as needed.
    """
    plan: List[str] = ["profile_dataset"]

    rows  = dataset_profile.get("shape", {}).get("rows", 0)
    cols  = dataset_profile.get("shape", {}).get("cols", 0)
    imb   = float(dataset_profile.get("imbalance_ratio") or 1.0)
    miss  = dataset_profile.get("missing_pct", {})
    ftypes = dataset_profile.get("feature_types", {})
    cat_cols = ftypes.get("categorical", [])
    n_unique = dataset_profile.get("n_unique_by_col", {})

    # ── Missing data handling ─────────────────────────────────────────────────
    max_missing = max(miss.values(), default=0.0)
    if max_missing >= _MISSING_SEV:
        plan.append("handle_severe_missing_data")  # median + indicator flags
    elif max_missing >= _MISSING_WARN:
        plan.append("handle_moderate_missing_data")

    # ── Cardinality handling ──────────────────────────────────────────────────
    high_card = [c for c in cat_cols if n_unique.get(c, 0) > _HIGH_CARD_UNIQ]
    if high_card:
        plan.append("apply_target_encoding")        # replaces OHE for those cols

    # ── Dimensionality reduction ──────────────────────────────────────────────
    if cols > _HIGH_DIM_COLS:
        plan.append("apply_feature_selection")      # variance / chi2 filter

    # ── Imbalance strategy ────────────────────────────────────────────────────
    if imb >= _IMBALANCE_SEV:
        plan.append("apply_severe_imbalance_strategy")   # SMOTE + threshold tuning
    elif imb >= _IMBALANCE_MOD:
        plan.append("consider_imbalance_strategy")        # class_weight + balanced acc

    plan.append("build_preprocessor")
    plan.append("select_models")

    # ── Dataset-size aware model list ─────────────────────────────────────────
    if rows < _SMALL_ROWS:
        plan.append("apply_regularization")         # signal to select simpler models
        plan.append("use_cross_validation")         # CV beats single split for tiny sets
    elif rows < _MEDIUM_ROWS:
        plan.append("use_cross_validation")
    elif rows >= _LARGE_ROWS:
        plan.append("use_fast_models_only")         # skip expensive SVC/full GB

    plan.append("train_models")

    # ── Memory-guided model prioritisation ───────────────────────────────────
    if memory_hint and memory_hint.get("best_model"):
        best = memory_hint["best_model"]
        plan.append(f"prioritize_model:{best}")

    plan += ["evaluate", "reflect", "write_report"]

    return plan


# ── Introspection helpers (used by the CLI summary) ───────────────────────────

def plan_rationale(
    dataset_profile: Dict[str, Any],
    memory_hint: Optional[Dict[str, Any]] = None,
) -> List[str]:
    """Return a human-readable list of reasons behind the generated plan."""
    rows  = dataset_profile.get("shape", {}).get("rows", 0)
    cols  = dataset_profile.get("shape", {}).get("cols", 0)
    imb   = float(dataset_profile.get("imbalance_ratio") or 1.0)
    miss  = dataset_profile.get("missing_pct", {})
    ftypes = dataset_profile.get("feature_types", {})
    cat_cols = ftypes.get("categorical", [])
    n_unique = dataset_profile.get("n_unique_by_col", {})

    reasons: List[str] = []
    max_missing = max(miss.values(), default=0.0)

    if rows < _SMALL_ROWS:
        reasons.append(f"Small dataset ({rows} rows) → cross-validation + regularised models")
    elif rows >= _LARGE_ROWS:
        reasons.append(f"Large dataset ({rows} rows) → fast models only (skip SVC / full GBM)")
    else:
        reasons.append(f"Medium dataset ({rows} rows) → full model suite + CV")

    if imb >= _IMBALANCE_SEV:
        reasons.append(f"Severe class imbalance ({imb:.1f}x) → SMOTE + threshold tuning")
    elif imb >= _IMBALANCE_MOD:
        reasons.append(f"Moderate imbalance ({imb:.1f}x) → class_weight='balanced'")

    if max_missing >= _MISSING_SEV:
        reasons.append(f"Severe missing data (max {max_missing:.1f}%) → indicator-flag imputation")
    elif max_missing >= _MISSING_WARN:
        reasons.append(f"Moderate missing data (max {max_missing:.1f}%) → median/mode imputation")

    high_card = [c for c in cat_cols if n_unique.get(c, 0) > _HIGH_CARD_UNIQ]
    if high_card:
        reasons.append(f"{len(high_card)} high-cardinality column(s) → target encoding")

    if cols > _HIGH_DIM_COLS:
        reasons.append(f"High dimensionality ({cols} cols) → feature selection step")

    if memory_hint and memory_hint.get("best_model"):
        reasons.append(f"Memory hit → prioritising {memory_hint['best_model']} from prior run")

    return reasons
