"""
Evaluation tool — confusion matrix, classification report, and markdown report.

Improvements over the original
────────────────────────────────
• Seaborn heatmap for the confusion matrix (cleaner than raw imshow)
• Per-model metrics bar chart saved as PNG
• Structured classification report dict (machine-readable)
• write_markdown_report includes CV scores and diagnostics sections
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional

import matplotlib
matplotlib.use("Agg")  # headless — no display needed
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix


# ── Helpers ───────────────────────────────────────────────────────────────────

def save_json(path: str, obj: Any) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


# ── Confusion matrix ──────────────────────────────────────────────────────────

def plot_confusion_matrix(
    cm: np.ndarray,
    labels: List[str],
    out_path: str,
    title: str,
) -> None:
    fig, ax = plt.subplots(figsize=(max(5, len(labels)), max(4, len(labels) - 1)))

    try:
        import seaborn as sns
        sns.heatmap(
            cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=labels, yticklabels=labels,
            linewidths=0.5, ax=ax,
        )
    except ImportError:
        im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
        plt.colorbar(im, ax=ax)
        ticks = np.arange(len(labels))
        ax.set_xticks(ticks); ax.set_xticklabels(labels, rotation=45, ha="right")
        ax.set_yticks(ticks); ax.set_yticklabels(labels)
        thresh = cm.max() / 2 if cm.size else 0
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, format(int(cm[i, j]), "d"),
                        ha="center", va="center",
                        color="white" if cm[i, j] > thresh else "black")

    ax.set_title(title, fontsize=12, pad=10)
    ax.set_ylabel("True label");  ax.set_xlabel("Predicted label")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ── Models comparison bar chart ───────────────────────────────────────────────

def plot_model_comparison(
    all_metrics: List[Dict[str, Any]],
    out_path: str,
) -> None:
    names   = [m["model"] for m in all_metrics]
    ba      = [m.get("balanced_accuracy", 0) for m in all_metrics]
    f1      = [m.get("f1_macro", 0) for m in all_metrics]

    x = np.arange(len(names))
    w = 0.35

    fig, ax = plt.subplots(figsize=(max(6, len(names) * 1.5), 4))
    bars1 = ax.bar(x - w / 2, ba, w, label="Balanced Acc", color="#4C72B0", alpha=0.85)
    bars2 = ax.bar(x + w / 2, f1, w, label="Macro F1",      color="#DD8452", alpha=0.85)

    ax.set_ylim(0, 1.05)
    ax.set_xticks(x); ax.set_xticklabels(names, rotation=20, ha="right")
    ax.set_ylabel("Score"); ax.set_title("Model Comparison")
    ax.legend(loc="lower right")
    ax.axhline(0.5, color="grey", linestyle="--", linewidth=0.8, alpha=0.5)

    for bar in [*bars1, *bars2]:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + 0.01,
                f"{h:.2f}", ha="center", va="bottom", fontsize=7)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ── Main evaluation entry point ───────────────────────────────────────────────

def evaluate_best(
    training_payload: Dict[str, Any],
    output_dir: str,
) -> Dict[str, Any]:
    """
    Evaluate the best model, produce artefacts, and return a structured payload.
    """
    best        = training_payload["best"]
    all_metrics = training_payload["all_metrics"]
    cv_scores   = training_payload.get("cv_scores")

    y_test = best["y_test"]
    y_pred = best["y_pred"]

    # Confusion matrix
    cm      = confusion_matrix(y_test, y_pred)
    labels  = sorted([str(x) for x in y_test.dropna().unique().tolist()])
    cm_path = os.path.join(output_dir, "confusion_matrix.png")
    plot_confusion_matrix(cm, labels, cm_path, f"Confusion Matrix — {best['name']}")

    # Model comparison chart
    cmp_path = os.path.join(output_dir, "model_comparison.png")
    plot_model_comparison(all_metrics, cmp_path)

    # Classification report (string + dict)
    cls_report_str  = classification_report(y_test, y_pred, zero_division=0)
    cls_report_dict = classification_report(y_test, y_pred, zero_division=0, output_dict=True)

    return {
        "best_metrics":            best["metrics"],
        "all_metrics":             all_metrics,
        "confusion_matrix_path":   cm_path,
        "model_comparison_path":   cmp_path,
        "classification_report":   cls_report_str,
        "classification_report_dict": cls_report_dict,
        "cv_scores":               cv_scores,
    }


# ── Markdown report ───────────────────────────────────────────────────────────

def write_markdown_report(
    out_path: str,
    ctx: Any,
    fingerprint: str,
    dataset_profile: Dict[str, Any],
    plan: List[str],
    eval_payload: Dict[str, Any],
    reflection: Dict[str, Any],
) -> None:
    best = eval_payload["best_metrics"]
    cv   = eval_payload.get("cv_scores") or {}

    def _short_list(xs: List[str], n: int = 12) -> str:
        return ", ".join(xs[:n]) + (" …" if len(xs) > n else "")

    numeric     = dataset_profile.get("feature_types", {}).get("numeric", [])
    categorical = dataset_profile.get("feature_types", {}).get("categorical", [])
    notes       = dataset_profile.get("notes", [])
    diag        = reflection.get("diagnostics", {})

    # Status badge
    sev    = reflection.get("severity", 0)
    badge  = {0: "✅ OK", 1: "⚠️ Minor Issues", 2: "🔶 Needs Attention", 3: "🔴 Critical"}
    status = badge.get(sev, "⚠️")

    # All-models table rows
    metric_rows = "\n".join(
        f"| {m['model']} | {m['accuracy']:.3f} | {m['balanced_accuracy']:.3f} "
        f"| {m['f1_macro']:.3f} | {m.get('train_time_s', '—')} |"
        for m in eval_payload["all_metrics"]
    )

    # Issues and suggestions
    issues_md  = "\n".join(f"- {i}" for i in reflection.get("issues", [])) or "- None"
    suggest_md = "\n".join(f"- {s}" for s in reflection.get("suggestions", [])) or "- None"

    # CV section
    cv_md = ""
    if cv:
        cv_md = f"""
## Cross-Validation ({cv.get('model', '')})
| Split | Balanced Acc | Macro F1 |
|-------|-------------|---------|
| Train | {cv.get('train_balanced_accuracy', '—'):.3f} | {cv.get('train_f1_macro', '—'):.3f} |
| Test  | {cv.get('test_balanced_accuracy', '—'):.3f} | {cv.get('test_f1_macro', '—'):.3f} |

CV Std (balanced acc): **{cv.get('cv_std_balanced_accuracy', '—')}**
"""

    # Top correlations
    top_corr = dataset_profile.get("top_correlations", [])
    corr_md  = ""
    if top_corr:
        corr_rows = "\n".join(
            f"| {p['feature_a']} | {p['feature_b']} | {p['pearson_r']} |"
            for p in top_corr
        )
        corr_md = f"""
### Top Correlated Features
| Feature A | Feature B | Pearson r |
|-----------|-----------|-----------|
{corr_rows}
"""

    md = f"""# Agentic Data Scientist — Run Report

**Status:** {status}
**Run ID:** `{ctx.run_id}`
**Started (UTC):** {ctx.started_at}
**Dataset:** `{ctx.data_path}`
**Target:** `{ctx.target}`
**Fingerprint:** `{fingerprint}`

---

## Dataset Profile

| Property | Value |
|----------|-------|
| Rows | **{dataset_profile["shape"]["rows"]:,}** |
| Columns | **{dataset_profile["shape"]["cols"]}** |
| Task type | **{"Classification" if dataset_profile.get("is_classification") else "Regression"}** |
| Imbalance ratio | **{dataset_profile.get("imbalance_ratio", "—")}** |
| Duplicates | **{dataset_profile.get("duplicate_rows", {}).get("count", 0)}** |

**Numeric features ({len(numeric)}):** {_short_list(numeric)}
**Categorical features ({len(categorical)}):** {_short_list(categorical)}

**Notes**
{chr(10).join(f"- {n}" for n in notes) if notes else "- (none)"}
{corr_md}

---

## Execution Plan

{chr(10).join(f"{i + 1}. `{t}`" for i, t in enumerate(plan))}

---

## Results

### Best Model: `{best.get("model")}`

| Metric | Score |
|--------|-------|
| Accuracy | **{best.get("accuracy"):.3f}** |
| Balanced Accuracy | **{best.get("balanced_accuracy"):.3f}** |
| Macro F1 | **{best.get("f1_macro"):.3f}** |
| Macro Precision | **{best.get("precision_macro"):.3f}** |
| Macro Recall | **{best.get("recall_macro"):.3f}** |
| Train time | {best.get("train_time_s", "—")} s |

### All Candidates

| Model | Accuracy | Balanced Acc | Macro F1 | Train time (s) |
|-------|----------|--------------|----------|----------------|
{metric_rows}
{cv_md}

---

## Reflection

**Severity:** {status}

### Issues
{issues_md}

### Suggestions
{suggest_md}

### Diagnostics
```json
{json.dumps(diag, indent=2)}
```

---

## Artefacts

- Confusion matrix: `{eval_payload.get("confusion_matrix_path")}`
- Model comparison: `{eval_payload.get("model_comparison_path")}`

"""

    with open(out_path, "w", encoding="utf-8") as f:
        f.write(md)
