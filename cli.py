#!/usr/bin/env python3
"""
cli.py — Rich terminal UI for the Agentic Data Scientist.

Usage
─────
    python cli.py --data data/demo.csv --target label
    python cli.py --data data/demo.csv --target auto --cv --max_replans 2
    python cli.py  # interactive prompts when flags are omitted
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from rich import box
from rich.align import Align
from rich.columns import Columns
from rich.console import Console
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
)
from rich.prompt import Confirm, Prompt
from rich.rule import Rule
from rich.table import Table
from rich.text import Text

# ── Banner ────────────────────────────────────────────────────────────────────

BANNER = """\
[bold cyan]
   ___                    _   _      ____        _          ___      _           _   _     _
  / _ \\ __ _  ___ _ __ | |_(_) ___|  _ \\  __ _| |_ __ _  / __| ___(_) ___ _ __ | |_(_)___| |_
 / /_)/ / _` |/ _ \\ '_ \\| __| |/ __| | | |/ _` | __/ _` | \\___ \\/ __| |/ _ \\ '_ \\| __| / __| __|
/ ___/ | (_| |  __/ | | | |_| | (__| |_| | (_| | || (_| |  ___) \\__ \\ |  __/ | | | |_| \\__ \\ |_
\\/      \\__,_|\\___|_| |_|\\__|_|\\___|____/ \\__,_|\\__\\__,_| |____/|___/_|\\___|_| |_|\\__|_|___/\\__|
[/bold cyan]"""

COMPACT_BANNER = "[bold cyan]◆ AGENTIC DATA SCIENTIST[/bold cyan]"

console = Console()


# ── Severity colours ──────────────────────────────────────────────────────────

_SEV_STYLE = {0: "green", 1: "yellow", 2: "dark_orange", 3: "red"}
_SEV_LABEL = {0: "✅ OK", 1: "⚠️ Minor Issues", 2: "🔶 Needs Attention", 3: "🔴 Critical"}


# ── CLI class ─────────────────────────────────────────────────────────────────

class DataScientistCLI:
    """Wraps the AgenticDataScientist and renders Rich output at each event."""

    def __init__(self) -> None:
        self._progress:    Optional[Progress] = None
        self._train_task:  Optional[Any]      = None
        self._phase_task:  Optional[Any]      = None
        self._start_time:  float              = 0.0
        self._profile:     Dict[str, Any]     = {}
        self._plan:        List[str]          = []
        self._rationale:   List[str]          = []
        self._all_metrics: List[Dict]         = []
        self._best:        Dict[str, Any]     = {}
        self._cv:          Optional[Dict]     = None
        self._reflection:  Dict[str, Any]     = {}
        self._output_dir:  str                = ""
        self._n_models:    int                = 0
        self._done_models: int                = 0

    # ── Event handler ─────────────────────────────────────────────────────────

    def handle_event(self, event: str, data: Dict[str, Any]) -> None:
        handlers = {
            "load_start":   self._on_load_start,
            "load_done":    self._on_load_done,
            "profile_done": self._on_profile_done,
            "plan_done":    self._on_plan_done,
            "train_start":  self._on_train_start,
            "model_train":  self._on_model_train,
            "model_done":   self._on_model_done,
            "eval_done":    self._on_eval_done,
            "reflect_done": self._on_reflect_done,
            "replan":       self._on_replan,
            "run_done":     self._on_run_done,
        }
        if event in handlers:
            handlers[event](data)

    # ── Individual event handlers ─────────────────────────────────────────────

    def _on_load_start(self, data: Dict) -> None:
        console.print(Rule("[bold]Phase 1 — Loading Dataset[/bold]", style="cyan"))
        console.print(f"  [dim]→[/dim] [cyan]{data['path']}[/cyan]")

    def _on_load_done(self, data: Dict) -> None:
        console.print(
            f"  [green]✓[/green] Loaded [bold]{data['rows']:,}[/bold] rows × "
            f"[bold]{data['cols']}[/bold] columns"
        )

    def _on_profile_done(self, data: Dict) -> None:
        self._profile = data["profile"]
        fp   = data["fingerprint"]
        prev = data.get("prev")

        console.print(Rule("[bold]Phase 2 — Dataset Profile[/bold]", style="cyan"))

        # Dataset stats table
        t = Table(box=box.SIMPLE_HEAD, show_header=True, header_style="bold magenta")
        t.add_column("Property",  style="dim", width=22)
        t.add_column("Value",     style="bold white")

        p = self._profile
        t.add_row("Rows",          f"{p['shape']['rows']:,}")
        t.add_row("Columns",       str(p["shape"]["cols"]))
        t.add_row("Numeric feats", str(len(p["feature_types"]["numeric"])))
        t.add_row("Categorical",   str(len(p["feature_types"]["categorical"])))
        t.add_row("Imbalance",     str(p.get("imbalance_ratio", "—")))
        t.add_row("Duplicates",    f"{p.get('duplicate_rows', {}).get('count', 0)}")
        t.add_row("Fingerprint",   f"[dim]{fp}[/dim]")

        if data["memory_hit"] and prev:
            t.add_row(
                "Memory hit",
                f"[green]✓[/green] prev best = [bold]{prev.get('best_model', '?')}[/bold]",
            )

        console.print(Align(t, align="left"))

        # Missing data warning
        miss_max = max(p.get("missing_pct", {}).values(), default=0.0)
        if miss_max > 10:
            console.print(
                f"  [yellow]⚠[/yellow] Max missing: [bold]{miss_max:.1f}%[/bold]"
            )

        # Notes
        for note in p.get("notes", []):
            console.print(f"  [dim]ℹ[/dim] {note}")

        # Top correlations
        top_corr = p.get("top_correlations", [])
        if top_corr:
            console.print()
            console.print("  [bold]High feature correlations:[/bold]")
            for pair in top_corr:
                console.print(
                    f"    [yellow]{pair['feature_a']}[/yellow] ↔ "
                    f"[yellow]{pair['feature_b']}[/yellow]  r={pair['pearson_r']}"
                )

    def _on_plan_done(self, data: Dict) -> None:
        self._plan      = data["plan"]
        self._rationale = data.get("rationale", [])

        console.print(Rule("[bold]Phase 3 — Execution Plan[/bold]", style="cyan"))

        t = Table(box=box.SIMPLE, show_header=False, padding=(0, 1))
        t.add_column("#",    style="dim",         width=4)
        t.add_column("Step", style="bold white")

        for i, step in enumerate(self._plan, 1):
            t.add_row(str(i), step)

        console.print(Align(t, align="left"))

        if self._rationale:
            console.print()
            console.print("  [bold]Reasoning:[/bold]")
            for r in self._rationale:
                console.print(f"    [dim]•[/dim] {r}")

    def _on_train_start(self, data: Dict) -> None:
        self._n_models    = data["n_models"]
        self._done_models = 0

        console.print()
        console.print(Rule("[bold]Phase 4 — Model Training[/bold]", style="cyan"))

        self._progress = Progress(
            SpinnerColumn(),
            TextColumn("[bold cyan]{task.description}"),
            BarColumn(bar_width=30),
            MofNCompleteColumn(),
            TimeElapsedColumn(),
            console=console,
            transient=False,
        )
        self._train_task = self._progress.add_task(
            "Training models…", total=self._n_models
        )
        self._progress.start()

    def _on_model_train(self, data: Dict) -> None:
        if self._progress and self._train_task is not None:
            self._progress.update(
                self._train_task,
                description=f"Training: [bold]{data['name']}[/bold]",
            )

    def _on_model_done(self, data: Dict) -> None:
        self._done_models += 1
        if self._progress and self._train_task is not None:
            self._progress.update(self._train_task, advance=1)
        m = data["metrics"]
        if self._progress:
            self._progress.console.print(
                f"  [green]✓[/green] [bold]{data['name']:<24}[/bold] "
                f"bal_acc=[cyan]{m['balanced_accuracy']:.3f}[/cyan]  "
                f"f1=[cyan]{m['f1_macro']:.3f}[/cyan]  "
                f"[dim]{m.get('train_time_s', '?')}s[/dim]"
            )

    def _on_eval_done(self, data: Dict) -> None:
        if self._progress:
            self._progress.stop()
            self._progress = None

        self._all_metrics = data["all_metrics"]
        self._best        = data["best_metrics"]
        self._cv          = data.get("cv_scores")

        console.print()
        console.print(Rule("[bold]Phase 5 — Evaluation[/bold]", style="cyan"))

        # Models comparison table
        t = Table(
            title="All Candidates",
            box=box.SIMPLE_HEAD, show_header=True,
            header_style="bold magenta", padding=(0, 1),
        )
        t.add_column("Model",        style="bold white",  min_width=20)
        t.add_column("Accuracy",     justify="right")
        t.add_column("Balanced Acc", justify="right")
        t.add_column("Macro F1",     justify="right")
        t.add_column("Train (s)",    justify="right", style="dim")

        best_name = self._best.get("model", "")
        for m in self._all_metrics:
            is_best = m["model"] == best_name
            style   = "bold green" if is_best else "white"
            prefix  = "★ " if is_best else "  "
            t.add_row(
                f"{prefix}{m['model']}",
                f"{m['accuracy']:.3f}",
                f"{m['balanced_accuracy']:.3f}",
                f"{m['f1_macro']:.3f}",
                str(m.get("train_time_s", "—")),
                style=style,
            )

        console.print(Align(t, align="left"))

        # CV block
        if self._cv:
            cv = self._cv
            console.print()
            console.print(
                f"  [bold]5-fold CV ({cv.get('model', '')}):[/bold]  "
                f"train_ba=[cyan]{cv.get('train_balanced_accuracy', 0):.3f}[/cyan]  "
                f"test_ba=[cyan]{cv.get('test_balanced_accuracy', 0):.3f}[/cyan]  "
                f"std=[dim]{cv.get('cv_std_balanced_accuracy', 0):.3f}[/dim]"
            )

    def _on_reflect_done(self, data: Dict) -> None:
        self._reflection = data["reflection"]
        r   = self._reflection
        sev = r.get("severity", 0)

        console.print()
        console.print(Rule("[bold]Phase 6 — Reflection[/bold]", style="cyan"))
        console.print(
            f"  Status: [{_SEV_STYLE[sev]}]{_SEV_LABEL[sev]}[/{_SEV_STYLE[sev]}]  "
            f"Best model: [bold]{r.get('best_model', '?')}[/bold]"
        )

        if r.get("issues"):
            console.print()
            console.print("  [bold red]Issues:[/bold red]")
            for issue in r["issues"]:
                console.print(f"    [red]✗[/red] {issue}")

        if r.get("suggestions"):
            console.print()
            console.print("  [bold yellow]Suggestions:[/bold yellow]")
            for sug in r["suggestions"]:
                console.print(f"    [yellow]→[/yellow] {sug}")

        diag = r.get("diagnostics", {})
        if diag:
            console.print()
            diag_parts = [
                f"baseline_lift=[cyan]{diag.get('baseline_lift', '—')}[/cyan]",
                f"model_spread=[cyan]{diag.get('model_spread', '—')}[/cyan]",
            ]
            if "overfit_gap" in diag:
                diag_parts.append(f"overfit_gap=[yellow]{diag['overfit_gap']}[/yellow]")
            console.print("  [dim]Diagnostics: " + "  ".join(diag_parts) + "[/dim]")

        if r.get("replan_recommended"):
            console.print()
            console.print(
                "  [bold yellow]↺[/bold yellow] Replan recommended — "
                "will attempt another cycle…"
            )

    def _on_replan(self, data: Dict) -> None:
        a, m = data["attempt"], data["max_replans"]
        console.print()
        console.print(
            Rule(
                f"[bold yellow]Replan Cycle {a}/{m}[/bold yellow]",
                style="yellow",
            )
        )

    def _on_run_done(self, data: Dict) -> None:
        self._output_dir = data["output_dir"]
        elapsed = time.perf_counter() - self._start_time

        console.print()
        console.print(Rule(style="green"))

        # Final summary panel
        b = self._best
        sev   = self._reflection.get("severity", 0)
        style = _SEV_STYLE[sev]

        summary = (
            f"[bold]Best model:[/bold]  [bold cyan]{b.get('model', '?')}[/bold cyan]\n"
            f"[bold]Accuracy:[/bold]    {b.get('accuracy', 0):.3f}\n"
            f"[bold]Balanced acc:[/bold] {b.get('balanced_accuracy', 0):.3f}\n"
            f"[bold]Macro F1:[/bold]    {b.get('f1_macro', 0):.3f}\n\n"
            f"[bold]Status:[/bold]      [{style}]{_SEV_LABEL[sev]}[/{style}]\n"
            f"[bold]Wall time:[/bold]   {elapsed:.1f}s\n\n"
            f"[bold]Output dir:[/bold]\n[dim]{self._output_dir}[/dim]"
        )

        console.print(
            Panel(
                summary,
                title="[bold green]Run Complete[/bold green]",
                border_style="green",
                padding=(1, 3),
            )
        )

        # Artefact list
        out = Path(self._output_dir)
        artefacts = sorted(out.iterdir()) if out.exists() else []
        if artefacts:
            console.print()
            console.print("  [bold]Artefacts:[/bold]")
            for f in artefacts:
                size = f.stat().st_size
                size_str = f"{size // 1024} KB" if size >= 1024 else f"{size} B"
                console.print(f"    [dim]•[/dim] {f.name:<35} [dim]{size_str}[/dim]")

    # ── Main entry point ──────────────────────────────────────────────────────

    def run(self, args: argparse.Namespace) -> int:
        """Execute the pipeline and return an exit code."""
        # Banner
        try:
            console.print(BANNER)
        except Exception:
            console.print(COMPACT_BANNER)
        console.print()

        # Resolve data path interactively if not supplied
        data_path = args.data
        if not data_path:
            data_path = Prompt.ask(
                "[bold]Path to CSV dataset[/bold]",
                default="data/demo.csv",
            )

        if not os.path.isfile(data_path):
            console.print(f"[bold red]✗ File not found:[/bold red] {data_path}")
            return 1

        target = args.target
        if not target:
            target = Prompt.ask("[bold]Target column[/bold] (or 'auto')", default="auto")

        # Config panel
        cfg_table = Table(box=box.SIMPLE, show_header=False, padding=(0, 1))
        cfg_table.add_column("Key",   style="dim",        width=16)
        cfg_table.add_column("Value", style="bold white")
        cfg_table.add_row("Dataset",     data_path)
        cfg_table.add_row("Target",      target)
        cfg_table.add_row("Seed",        str(args.seed))
        cfg_table.add_row("Test split",  f"{args.test_size * 100:.0f}%")
        cfg_table.add_row("Max replans", str(args.max_replans))
        cfg_table.add_row("CV enabled",  "yes" if args.cv else "no")
        cfg_table.add_row("Output root", args.output_root)

        console.print(
            Panel(
                Align(cfg_table, align="left"),
                title="[bold]Run Configuration[/bold]",
                border_style="cyan",
                padding=(0, 1),
            )
        )
        console.print()

        # Lazily import here to keep startup fast
        from datapilot import DataPilot

        agent = DataPilot(
            verbose=False,
            on_event=self.handle_event,
        )

        self._start_time = time.perf_counter()

        try:
            agent.run(
                data_path=data_path,
                target=target,
                output_root=args.output_root,
                seed=args.seed,
                test_size=args.test_size,
                max_replans=args.max_replans,
                use_cv=args.cv,
            )
        except KeyboardInterrupt:
            console.print("\n[yellow]Interrupted.[/yellow]")
            if self._progress:
                self._progress.stop()
            return 130
        except Exception as exc:
            if self._progress:
                self._progress.stop()
            console.print(f"\n[bold red]Error:[/bold red] {exc}")
            if args.debug:
                import traceback
                console.print_exception(show_locals=True)
            return 1

        return 0


# ── Argument parsing ──────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="cli.py",
        description="Agentic Data Scientist — terminal UI",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--data",        help="Path to CSV dataset (omit for interactive prompt)")
    p.add_argument("--target",      help="Target column name or 'auto'")
    p.add_argument("--output_root", default="outputs",    help="Root output directory")
    p.add_argument("--seed",        type=int,   default=42,  help="Random seed")
    p.add_argument("--test_size",   type=float, default=0.2, help="Test split fraction (0–1)")
    p.add_argument("--max_replans", type=int,   default=1,   help="Max replan cycles")
    p.add_argument("--cv",          action="store_true",     help="Run 5-fold CV on best model")
    p.add_argument("--debug",       action="store_true",     help="Show full tracebacks on error")
    return p


def main() -> None:
    args = build_parser().parse_args()
    cli  = DataScientistCLI()
    sys.exit(cli.run(args))


if __name__ == "__main__":
    main()
