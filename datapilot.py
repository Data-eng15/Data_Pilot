"""
DataPilot — Autonomous ML Pipeline Orchestrator.

Handles dataset loading, profiling, planning, training, evaluation, reflection,
and optional re-planning cycles.  Designed for classification tasks.

Event system
────────────
Pass an `on_event(event: str, data: dict)` callable to the constructor.
Events fired (with their data keys):

  load_start       path
  load_done        rows, cols
  profile_done     profile, fingerprint, memory_hit (bool)
  plan_done        plan, rationale
  train_start      n_models
  model_train      name, idx, total
  model_done       name, metrics
  eval_done        best_metrics, all_metrics
  reflect_done     reflection
  replan           attempt, max_replans
  run_done         output_dir, run_id
"""

from __future__ import annotations

import os
import uuid
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional

import pandas as pd

from agents.memory import JSONMemory
from agents.planner import create_plan, plan_rationale
from agents.reflector import reflect, should_replan, apply_replan_strategy
from tools.data_profiler import dataset_fingerprint, infer_target_column, profile_dataset
from tools.evaluation import evaluate_best, save_json, write_markdown_report
from tools.modelling import build_preprocessor, select_models, train_models

EventCallback = Optional[Callable[[str, Dict[str, Any]], None]]


@dataclass
class RunContext:
    run_id:      str
    started_at:  str
    data_path:   str
    target:      str
    output_dir:  str
    seed:        int
    test_size:   float
    max_replans: int
    use_cv:      bool


def _now_iso() -> str:
    return datetime.utcnow().replace(microsecond=0).isoformat() + "Z"


class DataPilot:
    """
    Offline Agentic Data Scientist (classification-focused).

    Parameters
    ----------
    memory_path : path to the persistent JSON memory file
    verbose     : print logs to stdout (independent of on_event)
    on_event    : optional callback for structured progress events
    """

    def __init__(
        self,
        memory_path: str = "agent_memory.json",
        verbose: bool = True,
        on_event: EventCallback = None,
    ) -> None:
        self.verbose   = verbose
        self.memory    = JSONMemory(memory_path)
        self.on_event  = on_event
        self.ctx:   Optional[RunContext]   = None
        self.state: Dict[str, Any]         = {}

    # ── Logging ───────────────────────────────────────────────────────────────

    def log(self, msg: str) -> None:
        if self.verbose:
            print(f"[AgenticDataScientist] {msg}")

    def _fire(self, event: str, data: Dict[str, Any]) -> None:
        if self.on_event:
            try:
                self.on_event(event, data)
            except Exception:
                pass  # never let a callback crash the pipeline

    # ── Data loading ──────────────────────────────────────────────────────────

    def load_data(self, path: str) -> pd.DataFrame:
        self._fire("load_start", {"path": path})
        self.log(f"Loading dataset: {path}")
        df = pd.read_csv(path)
        self.log(f"Loaded {df.shape[0]:,} rows × {df.shape[1]} cols")
        self._fire("load_done", {"rows": df.shape[0], "cols": df.shape[1]})
        return df

    # ── Main orchestration ────────────────────────────────────────────────────

    def run(
        self,
        data_path:   str,
        target:      str,
        output_root: str   = "outputs",
        seed:        int   = 42,
        test_size:   float = 0.2,
        max_replans: int   = 1,
        use_cv:      bool  = False,
    ) -> str:
        """
        Run the full pipeline and return the output directory path.

        Parameters
        ----------
        data_path   : path to the CSV dataset
        target      : target column name or 'auto' to infer
        output_root : root directory for run artefacts
        seed        : random seed for reproducibility
        test_size   : fraction of data held out for testing
        max_replans : max number of re-planning cycles
        use_cv      : run 5-fold CV on best model (slower but more reliable)
        """
        run_id     = datetime.utcnow().strftime("%Y%m%d_%H%M%S") + "_" + str(uuid.uuid4())[:8]
        output_dir = os.path.join(output_root, run_id)
        os.makedirs(output_dir, exist_ok=True)

        self.ctx = RunContext(
            run_id=run_id, started_at=_now_iso(),
            data_path=data_path, target=target,
            output_dir=output_dir, seed=seed,
            test_size=test_size, max_replans=max_replans,
            use_cv=use_cv,
        )
        self.state = {"replan_count": 0}

        # ── Load ──────────────────────────────────────────────────────────────
        df = self.load_data(data_path)

        # ── Auto-infer target ─────────────────────────────────────────────────
        if target.strip().lower() == "auto":
            inferred = infer_target_column(df)
            if not inferred:
                raise ValueError(
                    "Could not infer target column. Please provide --target <name>."
                )
            self.ctx.target = inferred
            self.log(f"Inferred target: {inferred}")

        # ── Profile ───────────────────────────────────────────────────────────
        profile = profile_dataset(df, self.ctx.target)
        fp      = dataset_fingerprint(df, self.ctx.target)
        prev    = self.memory.get_dataset_record(fp)
        memory_hit = prev is not None

        if memory_hit:
            self.log(f"Memory hit: previously best={prev.get('best_model')} for fp={fp}")

        self._fire("profile_done", {
            "profile":     profile,
            "fingerprint": fp,
            "memory_hit":  memory_hit,
            "prev":        prev,
        })

        # ── Plan ──────────────────────────────────────────────────────────────
        plan      = create_plan(profile, memory_hint=prev)
        rationale = plan_rationale(profile, memory_hint=prev)
        self.log(f"Plan: {plan}")
        self._fire("plan_done", {"plan": plan, "rationale": rationale})

        # ── Training loop ─────────────────────────────────────────────────────
        while True:
            preprocessor = build_preprocessor(profile)
            candidates   = select_models(profile, seed=self.ctx.seed)
            self.log(f"Candidates: {[n for n, _ in candidates]}")

            self._fire("train_start", {"n_models": len(candidates)})

            def _model_callback(name: str, idx: int, total: int) -> None:
                self.log(f"Training ({idx + 1}/{total}): {name}")
                self._fire("model_train", {"name": name, "idx": idx, "total": total})

            results = train_models(
                df=df,
                target=self.ctx.target,
                preprocessor=preprocessor,
                candidates=candidates,
                seed=self.ctx.seed,
                test_size=self.ctx.test_size,
                output_dir=self.ctx.output_dir,
                verbose=False,              # let callback handle output
                use_cv=self.ctx.use_cv,
                on_model_train=_model_callback,
            )

            for r in results["results"]:
                self._fire("model_done", {"name": r["name"], "metrics": r["metrics"]})

            # ── Evaluate ──────────────────────────────────────────────────────
            eval_payload = evaluate_best(results, output_dir=self.ctx.output_dir)
            self._fire("eval_done", {
                "best_metrics": eval_payload["best_metrics"],
                "all_metrics":  eval_payload["all_metrics"],
                "cv_scores":    eval_payload.get("cv_scores"),
            })

            # ── Reflect ───────────────────────────────────────────────────────
            reflection = reflect(
                dataset_profile=profile,
                evaluation=eval_payload["best_metrics"],
                all_metrics=eval_payload["all_metrics"],
                cv_scores=eval_payload.get("cv_scores"),
                classification_report_str=eval_payload.get("classification_report"),
            )
            self._fire("reflect_done", {"reflection": reflection})

            # ── Persist artefacts ─────────────────────────────────────────────
            save_json(os.path.join(self.ctx.output_dir, "eda_summary.json"),  profile)
            save_json(os.path.join(self.ctx.output_dir, "plan.json"),         {"plan": plan, "rationale": rationale})
            save_json(os.path.join(self.ctx.output_dir, "metrics.json"),      eval_payload)
            save_json(os.path.join(self.ctx.output_dir, "reflection.json"),   reflection)

            write_markdown_report(
                out_path=os.path.join(self.ctx.output_dir, "report.md"),
                ctx=self.ctx,
                fingerprint=fp,
                dataset_profile=profile,
                plan=plan,
                eval_payload=eval_payload,
                reflection=reflection,
            )

            # ── Update memory ─────────────────────────────────────────────────
            self.memory.upsert_dataset_record(fp, {
                "last_seen":   _now_iso(),
                "target":      self.ctx.target,
                "shape":       profile["shape"],
                "best_model":  eval_payload["best_metrics"]["model"],
                "best_metrics": eval_payload["best_metrics"],
            })

            # ── Replan decision ───────────────────────────────────────────────
            if not should_replan(reflection):
                break

            if self.state["replan_count"] >= self.ctx.max_replans:
                self.log("Replan suggested but max_replans reached. Stopping.")
                break

            self.state["replan_count"] += 1
            self.log(f"Replanning attempt #{self.state['replan_count']}…")
            self._fire("replan", {
                "attempt":     self.state["replan_count"],
                "max_replans": self.ctx.max_replans,
            })
            plan, profile = apply_replan_strategy(plan, profile, reflection)

        self.log(f"Done. Outputs saved to: {self.ctx.output_dir}")
        self._fire("run_done", {"output_dir": self.ctx.output_dir, "run_id": run_id})
        return self.ctx.output_dir


# Backward-compatible alias
AgenticDataScientist = DataPilot
