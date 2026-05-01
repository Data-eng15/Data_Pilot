#!/usr/bin/env python3
"""
generate_demo.py — capture a demo run as SVG screenshots for GitHub README.

Produces:
  assets/demo_terminal.svg   — full Rich terminal session
  assets/demo_summary.svg    — just the final summary panel
  assets/confusion_matrix.png  (copied from run output)
  assets/model_comparison.png  (copied from run output)
"""

import shutil
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from rich import box
from rich.align import Align
from rich.console import Console
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
)
from rich.rule import Rule
from rich.table import Table
from rich.text import Text

ASSETS = Path("assets")
ASSETS.mkdir(exist_ok=True)

_SEV_STYLE = {0: "green", 1: "yellow", 2: "dark_orange", 3: "red"}
_SEV_LABEL = {0: "✅ OK", 1: "⚠️ Minor Issues", 2: "🔶 Needs Attention", 3: "🔴 Critical"}

# ── Recording console (width fixed for a clean SVG) ───────────────────────────
C = Console(record=True, width=105, force_terminal=True, highlight=False)


def _progress():
    return Progress(
        SpinnerColumn(),
        TextColumn("[bold cyan]{task.description}"),
        BarColumn(bar_width=30),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        console=C,
        transient=False,
    )


# ── Run the actual pipeline and collect artefacts ─────────────────────────────

def run_pipeline():
    from datapilot import DataPilot

    results: Dict[str, Any] = {}

    def on_event(event: str, data: Dict):
        results[event] = data

    agent = DataPilot(verbose=False, on_event=on_event)
    out_dir = agent.run(
        data_path="data/example_dataset.csv",
        target="auto",
        output_root="/tmp/datapilot_demo",
        seed=42,
        test_size=0.2,
        max_replans=0,
        use_cv=True,
    )
    results["_output_dir"] = out_dir
    return results


# ── Render the captured events into the recording console ─────────────────────

def render(results: Dict[str, Any]):
    banner = "[bold cyan]◆ DataPilot[/bold cyan] [dim]— Agentic ML Pipeline[/dim]"
    C.print()
    C.print(Align(banner, align="center"))
    C.print()

    # ── Phase 1 ───────────────────────────────────────────────────────────────
    d = results.get("load_done", {})
    C.print(Rule("[bold]Phase 1 — Loading Dataset[/bold]", style="cyan"))
    C.print(f"  [dim]→[/dim] [cyan]data/example_dataset.csv[/cyan]")
    C.print(
        f"  [green]✓[/green] Loaded [bold]{d.get('rows', 20):,}[/bold] rows × "
        f"[bold]{d.get('cols', 6)}[/bold] columns"
    )

    # ── Phase 2 ───────────────────────────────────────────────────────────────
    p = results.get("profile_done", {}).get("profile", {})
    C.print(Rule("[bold]Phase 2 — Dataset Profile[/bold]", style="cyan"))

    t = Table(box=box.SIMPLE_HEAD, show_header=True, header_style="bold magenta")
    t.add_column("Property",  style="dim", width=22)
    t.add_column("Value",     style="bold white")
    t.add_row("Rows",          f"{p.get('shape', {}).get('rows', 20):,}")
    t.add_row("Columns",       str(p.get("shape", {}).get("cols", 6)))
    t.add_row("Numeric feats", str(len(p.get("feature_types", {}).get("numeric", []))))
    t.add_row("Categorical",   str(len(p.get("feature_types", {}).get("categorical", []))))
    t.add_row("Imbalance",     str(p.get("imbalance_ratio", 1.0)))
    t.add_row("Duplicates",    str(p.get("duplicate_rows", {}).get("count", 0)))
    C.print(Align(t, align="left"))

    for note in p.get("notes", []):
        C.print(f"  [dim]ℹ[/dim] {note}")

    top_corr = p.get("top_correlations", [])
    if top_corr:
        C.print("  [bold]High feature correlations:[/bold]")
        for pair in top_corr[:3]:
            C.print(
                f"    [yellow]{pair['feature_a']}[/yellow] ↔ "
                f"[yellow]{pair['feature_b']}[/yellow]  r={pair['pearson_r']}"
            )

    # ── Phase 3 ───────────────────────────────────────────────────────────────
    plan = results.get("plan_done", {}).get("plan", [])
    rationale = results.get("plan_done", {}).get("rationale", [])
    C.print(Rule("[bold]Phase 3 — Execution Plan[/bold]", style="cyan"))
    t2 = Table(box=box.SIMPLE, show_header=False, padding=(0, 1))
    t2.add_column("#",    style="dim", width=4)
    t2.add_column("Step", style="bold white")
    for i, step in enumerate(plan, 1):
        t2.add_row(str(i), step)
    C.print(Align(t2, align="left"))
    if rationale:
        C.print("  [bold]Reasoning:[/bold]")
        for r in rationale:
            C.print(f"    [dim]•[/dim] {r}")

    # ── Phase 4 ───────────────────────────────────────────────────────────────
    all_metrics = results.get("eval_done", {}).get("all_metrics", [])
    C.print(Rule("[bold]Phase 4 — Model Training[/bold]", style="cyan"))
    for m in all_metrics:
        C.print(
            f"  [green]✓[/green] [bold]{m['model']:<24}[/bold] "
            f"bal_acc=[cyan]{m['balanced_accuracy']:.3f}[/cyan]  "
            f"f1=[cyan]{m['f1_macro']:.3f}[/cyan]  "
            f"[dim]{m.get('train_time_s', '?')}s[/dim]"
        )

    # ── Phase 5 ───────────────────────────────────────────────────────────────
    best_name = results.get("eval_done", {}).get("best_metrics", {}).get("model", "")
    cv        = results.get("eval_done", {}).get("cv_scores") or {}

    C.print(Rule("[bold]Phase 5 — Evaluation[/bold]", style="cyan"))
    t3 = Table(
        title="All Candidates", box=box.SIMPLE_HEAD,
        header_style="bold magenta", padding=(0, 1),
    )
    t3.add_column("Model",        style="bold white", min_width=22)
    t3.add_column("Accuracy",     justify="right")
    t3.add_column("Balanced Acc", justify="right")
    t3.add_column("Macro F1",     justify="right")
    t3.add_column("Train (s)",    justify="right", style="dim")
    for m in all_metrics:
        is_best = m["model"] == best_name
        prefix  = "★ " if is_best else "  "
        style   = "bold green" if is_best else "white"
        t3.add_row(
            f"{prefix}{m['model']}",
            f"{m['accuracy']:.3f}",
            f"{m['balanced_accuracy']:.3f}",
            f"{m['f1_macro']:.3f}",
            str(m.get("train_time_s", "—")),
            style=style,
        )
    C.print(Align(t3, align="left"))

    if cv:
        C.print(
            f"  [bold]5-fold CV ({cv.get('model', '')}):[/bold]  "
            f"train_ba=[cyan]{cv.get('train_balanced_accuracy', 0):.3f}[/cyan]  "
            f"test_ba=[cyan]{cv.get('test_balanced_accuracy', 0):.3f}[/cyan]  "
            f"std=[dim]{cv.get('cv_std_balanced_accuracy', 0):.3f}[/dim]"
        )

    # ── Phase 6 ───────────────────────────────────────────────────────────────
    ref = results.get("reflect_done", {}).get("reflection", {})
    sev = ref.get("severity", 0)
    C.print(Rule("[bold]Phase 6 — Reflection[/bold]", style="cyan"))
    C.print(
        f"  Status: [{_SEV_STYLE[sev]}]{_SEV_LABEL[sev]}[/{_SEV_STYLE[sev]}]  "
        f"Best: [bold]{ref.get('best_model', '?')}[/bold]"
    )
    for issue in ref.get("issues", []):
        C.print(f"    [red]✗[/red] {issue}")
    for sug in ref.get("suggestions", []):
        C.print(f"    [yellow]→[/yellow] {sug}")

    # ── Final summary ─────────────────────────────────────────────────────────
    b     = results.get("eval_done", {}).get("best_metrics", {})
    out   = results.get("_output_dir", "outputs/<run_id>")

    C.print()
    C.print(Rule(style="green"))
    summary = (
        f"[bold]Best model:[/bold]  [bold cyan]{b.get('model', '?')}[/bold cyan]\n"
        f"[bold]Accuracy:[/bold]    {b.get('accuracy', 0):.3f}\n"
        f"[bold]Balanced acc:[/bold] {b.get('balanced_accuracy', 0):.3f}\n"
        f"[bold]Macro F1:[/bold]    {b.get('f1_macro', 0):.3f}\n\n"
        f"[bold]Status:[/bold]      [{_SEV_STYLE[sev]}]{_SEV_LABEL[sev]}[/{_SEV_STYLE[sev]}]\n\n"
        f"[bold]Output:[/bold]  [dim]{out}[/dim]"
    )
    C.print(Panel(summary, title="[bold green]Run Complete[/bold green]",
                  border_style="green", padding=(1, 3)))


if __name__ == "__main__":
    print("Running pipeline …", flush=True)
    results = run_pipeline()

    print("Rendering demo …", flush=True)
    render(results)

    # Export full terminal SVG
    svg_path = ASSETS / "demo_terminal.svg"
    C.save_svg(str(svg_path), title="DataPilot — Agentic ML Pipeline")
    print(f"Saved: {svg_path}")

    # Copy PNG charts into assets/
    out_dir = Path(results["_output_dir"])
    for fname in ("confusion_matrix.png", "model_comparison.png"):
        src = out_dir / fname
        dst = ASSETS / fname
        if src.exists():
            shutil.copy(src, dst)
            print(f"Copied: {dst}")

    print("Done. Check assets/")
