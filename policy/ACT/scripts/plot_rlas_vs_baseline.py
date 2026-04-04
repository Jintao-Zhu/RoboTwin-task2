#!/usr/bin/env python3
"""
Generate RLAS vs Baseline bar charts and aggregate statistics for docs.

Outputs (default):
  policy/ACT/docs/figures/RLAS_vs_Baseline_bar_10000.png
  policy/ACT/docs/figures/RLAS_vs_Baseline_bar_30000.png
  policy/ACT/docs/figures/RLAS_vs_Baseline_bar_60000.png
  policy/ACT/docs/figures/RLAS_vs_Baseline_bar_120000.png
  policy/ACT/docs/figures/RLAS_vs_Baseline_delta_bar.png
  policy/ACT/docs/figures/rlas_aggregated_stats.json
  policy/ACT/docs/figures/rlas_main_table.md
  policy/ACT/docs/figures/rlas_all_params_table.md
  policy/ACT/docs/figures/rlas_empty_runs_table.md
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


TARGET_TASKS = ["beat_block_hammer", "open_laptop", "stack_blocks_two"]
MAIN_BUDGETS = [10000, 30000, 60000, 120000]
FULL_BUDGETS = [5000, 10000, 30000, 60000, 120000]


@dataclass
class Record:
    task: str
    method: str
    tag: str
    run: str
    budget: int
    success: float
    file: str


def parse_success_percent(value: str) -> float | None:
    if not value:
        return None
    value = value.strip()
    if not value:
        return None
    if value.endswith("%"):
        value = value[:-1]
    try:
        return float(value)
    except ValueError:
        return None


def read_csv_rows(path: str) -> list[dict[str, str]]:
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def gather_baseline_records(log_root: Path) -> list[Record]:
    records: list[Record] = []
    pattern = str(log_root / "*" / "sweeps" / "*" / "*" / "summary.csv")
    for summary in glob.glob(pattern):
        rel = Path(summary).resolve().relative_to(log_root.resolve())
        parts = rel.parts
        # <task>/sweeps/<dataset>/<run>/summary.csv
        if len(parts) < 5:
            continue
        task = parts[0]
        run = parts[3]
        if task not in TARGET_TASKS:
            continue
        rows = read_csv_rows(summary)
        if not rows:
            continue
        for row in rows:
            succ = parse_success_percent(row.get("success_percent", ""))
            if succ is None:
                continue
            budget = int(row["budget"])
            records.append(
                Record(
                    task=task,
                    method="baseline",
                    tag="baseline",
                    run=run,
                    budget=budget,
                    success=succ,
                    file=summary,
                )
            )
    return records


def gather_ati_records(log_root: Path) -> list[Record]:
    records: list[Record] = []
    pattern = str(log_root / "*" / "sweeps_plan" / "*" / "anchorS1_noneK4" / "*" / "summary.csv")
    for summary in glob.glob(pattern):
        rel = Path(summary).resolve().relative_to(log_root.resolve())
        parts = rel.parts
        # <task>/sweeps_plan/<dataset>/anchorS1_noneK4/<run>/summary.csv
        if len(parts) < 6:
            continue
        task = parts[0]
        tag = parts[3]
        run = parts[4]
        if task not in TARGET_TASKS:
            continue
        rows = read_csv_rows(summary)
        if not rows:
            continue
        for row in rows:
            succ = parse_success_percent(row.get("success_percent", ""))
            if succ is None:
                continue
            budget = int(row["budget"])
            records.append(
                Record(
                    task=task,
                    method="ati_none",
                    tag=tag,
                    run=run,
                    budget=budget,
                    success=succ,
                    file=summary,
                )
            )
    return records


def gather_rlas_records_and_empty(log_root: Path) -> tuple[list[Record], list[dict[str, Any]]]:
    records: list[Record] = []
    empty_runs: list[dict[str, Any]] = []
    pattern = str(log_root / "*" / "sweeps_rlas" / "*" / "*" / "*" / "summary.csv")
    for summary in glob.glob(pattern):
        rel = Path(summary).resolve().relative_to(log_root.resolve())
        parts = rel.parts
        # <task>/sweeps_rlas/<dataset>/<param>/<run>/summary.csv
        if len(parts) < 6:
            continue
        task = parts[0]
        param = parts[3]
        run = parts[4]
        if task not in TARGET_TASKS:
            continue
        rows = read_csv_rows(summary)
        if not rows:
            empty_runs.append(
                {
                    "task": task,
                    "param": param,
                    "run": run,
                    "summary_csv": summary,
                }
            )
            continue
        for row in rows:
            succ = parse_success_percent(row.get("success_percent", ""))
            if succ is None:
                continue
            budget = int(row["budget"])
            records.append(
                Record(
                    task=task,
                    method="rlas",
                    tag=param,
                    run=run,
                    budget=budget,
                    success=succ,
                    file=summary,
                )
            )
    empty_runs.sort(key=lambda x: (x["task"], x["param"], x["run"]))
    return records, empty_runs


def best_per_budget(records: list[Record], by_tag: bool = False) -> dict[tuple[str, int] | tuple[str, str, int], Record]:
    out: dict[tuple[str, int] | tuple[str, str, int], Record] = {}
    for r in records:
        key = (r.task, r.tag, r.budget) if by_tag else (r.task, r.budget)
        old = out.get(key)
        if old is None or r.success > old.success:
            out[key] = r
    return out


def latest_per_budget(records: list[Record], by_tag: bool = False) -> dict[tuple[str, int] | tuple[str, str, int], Record]:
    out: dict[tuple[str, int] | tuple[str, str, int], Record] = {}
    for r in records:
        key = (r.task, r.tag, r.budget) if by_tag else (r.task, r.budget)
        old = out.get(key)
        if old is None or r.run > old.run:
            out[key] = r
    return out


def to_curve(values: dict[int, float], budgets: list[int]) -> list[float | None]:
    curve: list[float | None] = []
    for b in budgets:
        curve.append(values.get(b))
    return curve


def curve_text(curve: list[float | None], budgets: list[int]) -> str:
    items = []
    for b, v in zip(budgets, curve):
        if v is None:
            items.append(f"{b}:NA")
        else:
            items.append(f"{b}:{v:.1f}")
    return " ".join(items)


def save_grouped_bar(
    out_path: Path,
    title: str,
    x_labels: list[str],
    left_values: list[float],
    right_values: list[float],
    left_label: str,
    right_label: str,
) -> None:
    x = np.arange(len(x_labels))
    width = 0.34
    fig, ax = plt.subplots(figsize=(10, 6))
    bars1 = ax.bar(x - width / 2, left_values, width, label=left_label, color="#4C72B0")
    bars2 = ax.bar(x + width / 2, right_values, width, label=right_label, color="#DD8452")

    ax.set_ylabel("Success Rate (%)")
    ax.set_ylim(0, 100)
    ax.set_xticks(x)
    ax.set_xticklabels(x_labels)
    ax.set_title(title)
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.35)

    def annotate(bars: Any) -> None:
        for bar in bars:
            h = bar.get_height()
            ax.annotate(
                f"{h:.1f}",
                xy=(bar.get_x() + bar.get_width() / 2, h),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=9,
            )

    annotate(bars1)
    annotate(bars2)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def save_delta_bar(out_path: Path, deltas: dict[int, list[float]]) -> None:
    budgets = sorted(deltas.keys())
    x = np.arange(len(TARGET_TASKS))
    width = 0.18
    fig, ax = plt.subplots(figsize=(11, 6))
    palette = {
        10000: "#55A868",
        30000: "#C44E52",
        60000: "#8172B3",
        120000: "#64B5CD",
    }
    for idx, budget in enumerate(budgets):
        shift = (idx - (len(budgets) - 1) / 2) * width
        bars = ax.bar(
            x + shift,
            deltas[budget],
            width=width,
            label=f"budget={budget}",
            color=palette.get(budget, None),
        )
        for bar in bars:
            h = bar.get_height()
            ax.annotate(
                f"{h:+.1f}",
                xy=(bar.get_x() + bar.get_width() / 2, h),
                xytext=(0, 3 if h >= 0 else -12),
                textcoords="offset points",
                ha="center",
                va="bottom" if h >= 0 else "top",
                fontsize=8,
            )
    ax.axhline(0, color="black", linewidth=1.0)
    ax.set_xticks(x)
    ax.set_xticklabels(TARGET_TASKS)
    ax.set_ylabel("RLAS-best - Baseline (%)")
    ax.set_title("Delta: RLAS-best vs Baseline (Best-over-runs)")
    ax.legend(ncols=2)
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--repo_root",
        type=Path,
        default=Path(__file__).resolve().parents[2],
        help="Repo root (RoboTwin_distillation).",
    )
    parser.add_argument(
        "--out_dir",
        type=Path,
        default=None,
        help="Output figure dir. Default: policy/ACT/docs/figures",
    )
    args = parser.parse_args()

    repo_root: Path = args.repo_root.resolve()
    log_root = repo_root / "policy" / "ACT" / "logs"
    out_dir = args.out_dir or (repo_root / "policy" / "ACT" / "docs" / "figures")
    out_dir.mkdir(parents=True, exist_ok=True)

    baseline_records = gather_baseline_records(log_root)
    ati_records = gather_ati_records(log_root)
    rlas_records, empty_runs = gather_rlas_records_and_empty(log_root)

    baseline_best = best_per_budget(baseline_records, by_tag=False)
    ati_latest = latest_per_budget(ati_records, by_tag=False)
    rlas_best_of_params = best_per_budget(rlas_records, by_tag=False)
    rlas_latest_by_param = latest_per_budget(rlas_records, by_tag=True)
    rlas_best_by_param = best_per_budget(rlas_records, by_tag=True)

    # Main comparison table and figures.
    main_rows: list[dict[str, Any]] = []
    delta_map: dict[int, list[float]] = {b: [] for b in MAIN_BUDGETS}

    for budget in MAIN_BUDGETS:
        task_labels = []
        baseline_vals = []
        rlas_vals = []
        for task in TARGET_TASKS:
            base_rec = baseline_best.get((task, budget))
            rlas_rec = rlas_best_of_params.get((task, budget))
            baseline_success = base_rec.success if base_rec else math.nan
            rlas_success = rlas_rec.success if rlas_rec else math.nan
            delta = rlas_success - baseline_success

            task_labels.append(task)
            baseline_vals.append(baseline_success)
            rlas_vals.append(rlas_success)
            delta_map[budget].append(delta)

            main_rows.append(
                {
                    "task": task,
                    "budget": budget,
                    "baseline_success": None if math.isnan(baseline_success) else round(baseline_success, 3),
                    "baseline_run": None if base_rec is None else base_rec.run,
                    "rlas_success": None if math.isnan(rlas_success) else round(rlas_success, 3),
                    "rlas_param": None if rlas_rec is None else rlas_rec.tag,
                    "rlas_run": None if rlas_rec is None else rlas_rec.run,
                    "delta": None if (math.isnan(baseline_success) or math.isnan(rlas_success)) else round(delta, 3),
                }
            )

        save_grouped_bar(
            out_path=out_dir / f"RLAS_vs_Baseline_bar_{budget}.png",
            title=f"RLAS-best vs Baseline @ budget={budget}",
            x_labels=task_labels,
            left_values=baseline_vals,
            right_values=rlas_vals,
            left_label="Baseline (best-over-runs)",
            right_label="RLAS (best-over-params)",
        )

    save_delta_bar(out_dir / "RLAS_vs_Baseline_delta_bar.png", delta_map)

    # Build all-params summary rows.
    # task + param -> curves for latest and best; and run counters.
    non_empty_run_counter: dict[tuple[str, str], set[str]] = {}
    for r in rlas_records:
        non_empty_run_counter.setdefault((r.task, r.tag), set()).add(r.run)

    empty_run_counter: dict[tuple[str, str], set[str]] = {}
    for e in empty_runs:
        empty_run_counter.setdefault((e["task"], e["param"]), set()).add(e["run"])

    param_keys = sorted({(r.task, r.tag) for r in rlas_records})
    all_params_rows: list[dict[str, Any]] = []
    for task, param in param_keys:
        latest_values: dict[int, float] = {}
        best_values: dict[int, float] = {}
        latest_runs: dict[int, str] = {}
        best_runs: dict[int, str] = {}
        for budget in FULL_BUDGETS:
            rec_latest = rlas_latest_by_param.get((task, param, budget))
            rec_best = rlas_best_by_param.get((task, param, budget))
            if rec_latest:
                latest_values[budget] = rec_latest.success
                latest_runs[budget] = rec_latest.run
            if rec_best:
                best_values[budget] = rec_best.success
                best_runs[budget] = rec_best.run
        latest_curve = to_curve(latest_values, FULL_BUDGETS)
        best_curve = to_curve(best_values, FULL_BUDGETS)
        all_params_rows.append(
            {
                "task": task,
                "param": param,
                "non_empty_runs": len(non_empty_run_counter.get((task, param), set())),
                "empty_runs": len(empty_run_counter.get((task, param), set())),
                "latest_curve": latest_curve,
                "best_curve": best_curve,
                "latest_curve_text": curve_text(latest_curve, FULL_BUDGETS),
                "best_curve_text": curve_text(best_curve, FULL_BUDGETS),
                "latest_runs": latest_runs,
                "best_runs": best_runs,
            }
        )

    summary_payload = {
        "target_tasks": TARGET_TASKS,
        "main_budgets": MAIN_BUDGETS,
        "full_budgets": FULL_BUDGETS,
        "counts": {
            "rlas_summary_total": len(glob.glob(str(log_root / "*" / "sweeps_rlas" / "*" / "*" / "*" / "summary.csv"))),
            "rlas_non_empty": len({(r.task, r.tag, r.run) for r in rlas_records}),
            "rlas_empty": len(empty_runs),
        },
        "main_rows": main_rows,
        "all_params_rows": all_params_rows,
        "empty_runs": empty_runs,
    }

    (out_dir / "rlas_aggregated_stats.json").write_text(
        json.dumps(summary_payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    # Emit helper markdown snippets for docs.
    # 1) Main table
    main_md = []
    main_md.append("| Task | Budget | Baseline(%) | RLAS-best(%) | Delta | RLAS Param |")
    main_md.append("|---|---:|---:|---:|---:|---|")
    for row in sorted(main_rows, key=lambda x: (x["task"], x["budget"])):
        base = "NA" if row["baseline_success"] is None else f"{row['baseline_success']:.1f}"
        rlas = "NA" if row["rlas_success"] is None else f"{row['rlas_success']:.1f}"
        delta = "NA" if row["delta"] is None else f"{row['delta']:+.1f}"
        param = row["rlas_param"] or "NA"
        main_md.append(
            f"| {row['task']} | {row['budget']} | {base} | {rlas} | {delta} | {param} |"
        )
    (out_dir / "rlas_main_table.md").write_text("\n".join(main_md) + "\n", encoding="utf-8")

    # 2) All params table
    all_md = []
    all_md.append("| Task | Param | Non-empty Runs | Empty Runs | Latest Curve(5k/10k/30k/60k/120k) | Best Curve(5k/10k/30k/60k/120k) |")
    all_md.append("|---|---|---:|---:|---|---|")
    for row in all_params_rows:
        all_md.append(
            f"| {row['task']} | {row['param']} | {row['non_empty_runs']} | {row['empty_runs']} | "
            f"{row['latest_curve_text']} | {row['best_curve_text']} |"
        )
    (out_dir / "rlas_all_params_table.md").write_text("\n".join(all_md) + "\n", encoding="utf-8")

    # 3) Empty runs table
    empty_md = []
    empty_md.append("| Task | Param | Run | summary.csv |")
    empty_md.append("|---|---|---|---|")
    for row in empty_runs:
        empty_md.append(
            f"| {row['task']} | {row['param']} | {row['run']} | `{row['summary_csv']}` |"
        )
    (out_dir / "rlas_empty_runs_table.md").write_text("\n".join(empty_md) + "\n", encoding="utf-8")

    print(f"[OK] figures and stats written to: {out_dir}")
    print(f"[OK] main rows: {len(main_rows)}, all-param rows: {len(all_params_rows)}, empty runs: {len(empty_runs)}")


if __name__ == "__main__":
    main()
