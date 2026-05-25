import argparse
import json
import math
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


@dataclass(frozen=True)
class MethodSpec:
    key: str
    label: str
    color: str
    marker: str
    pattern: re.Pattern


@dataclass(frozen=True)
class ResultPoint:
    step: int
    success_rate: float
    successes: int | None
    episodes: int | None
    metrics_path: Path


def build_method_specs(task_config: str) -> list[MethodSpec]:
    escaped_config = re.escape(task_config)
    return [
        MethodSpec(
            key="baseline",
            label="Baseline",
            color="#0072B2",
            marker="o",
            pattern=re.compile(rf"^{escaped_config}-u(?P<step>\d+)$"),
        ),
        MethodSpec(
            key="rlas_group",
            label="RLAS-Group",
            color="#D55E00",
            marker="s",
            pattern=re.compile(rf"^{escaped_config}-rlas[-_]group-u(?P<step>\d+)$"),
        ),
    ]


def parse_timestamp_dir(path: Path) -> datetime:
    try:
        return datetime.strptime(path.name, "%Y-%m-%d %H:%M:%S")
    except ValueError:
        return datetime.fromtimestamp(path.stat().st_mtime)


def latest_metrics_file(run_dir: Path) -> Path | None:
    metrics_files = list(run_dir.glob("*/_metrics.json"))
    if (run_dir / "_metrics.json").exists():
        metrics_files.append(run_dir / "_metrics.json")
    if not metrics_files:
        return None
    return max(metrics_files, key=lambda path: parse_timestamp_dir(path.parent))


def read_success_rate(metrics_path: Path) -> tuple[float, int | None, int | None]:
    with metrics_path.open("r", encoding="utf-8") as f:
        metrics = json.load(f)

    success_rate = float(metrics["success_rate"])
    if success_rate <= 1.0:
        success_rate *= 100.0

    successes = metrics.get("successes")
    episodes = metrics.get("episodes")
    return success_rate, successes, episodes


def collect_results(result_dir: Path) -> dict[str, list[ResultPoint]]:
    task_config = result_dir.name
    specs = build_method_specs(task_config)
    results = {spec.key: [] for spec in specs}

    for run_dir in sorted(path for path in result_dir.iterdir() if path.is_dir()):
        for spec in specs:
            match = spec.pattern.match(run_dir.name)
            if not match:
                continue

            metrics_path = latest_metrics_file(run_dir)
            if metrics_path is None:
                print(f"Skip {run_dir}: _metrics.json not found")
                continue

            success_rate, successes, episodes = read_success_rate(metrics_path)
            results[spec.key].append(
                ResultPoint(
                    step=int(match.group("step")),
                    success_rate=success_rate,
                    successes=successes,
                    episodes=episodes,
                    metrics_path=metrics_path,
                )
            )

    return {key: sorted(points, key=lambda point: point.step) for key, points in results.items()}


def format_step(step: int) -> str:
    if step >= 1000 and step % 1000 == 0:
        return f"{step // 1000}k"
    return str(step)


def print_summary(results: dict[str, list[ResultPoint]], specs: list[MethodSpec]) -> None:
    print("Extracted success rates:")
    for spec in specs:
        points = results.get(spec.key, [])
        if not points:
            print(f"  {spec.label}: no matching results")
            continue
        values = ", ".join(f"{format_step(point.step)}={point.success_rate:.1f}%" for point in points)
        print(f"  {spec.label}: {values}")


def plot_results(
    result_dir: Path,
    results: dict[str, list[ResultPoint]],
    output: Path,
    dpi: int,
    show_inset: bool,
    annotate: bool,
) -> None:
    specs = build_method_specs(result_dir.name)
    all_steps = sorted({point.step for points in results.values() for point in points})
    if not all_steps:
        raise RuntimeError(f"No baseline/RLAS-Group metrics found under {result_dir}")

    step_to_x = {step: idx for idx, step in enumerate(all_steps)}
    x_labels = [format_step(step) for step in all_steps]

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "axes.labelsize": 16,
            "axes.titlesize": 17,
            "xtick.labelsize": 13,
            "ytick.labelsize": 13,
            "legend.fontsize": 13,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    fig, ax = plt.subplots(figsize=(8.6, 5.2))
    max_y = 0.0
    min_y = 100.0

    for spec in specs:
        points = results.get(spec.key, [])
        if not points:
            continue

        x_values = [step_to_x[point.step] for point in points]
        y_values = [point.success_rate for point in points]
        max_y = max(max_y, max(y_values))
        min_y = min(min_y, min(y_values))

        ax.plot(
            x_values,
            y_values,
            marker=spec.marker,
            markersize=7.5,
            linewidth=2.8,
            color=spec.color,
            label=spec.label,
        )

        if annotate:
            offset = -4.0 if spec.key == "baseline" else 3.0
            vertical_alignment = "top" if offset < 0 else "bottom"
            for x_value, y_value in zip(x_values, y_values):
                ax.text(
                    x_value,
                    y_value + offset,
                    f"{y_value:.0f}",
                    color=spec.color,
                    ha="center",
                    va=vertical_alignment,
                    fontsize=11,
                    fontweight="bold",
                )

    ax.set_title("Open Laptop: RLAS-Group vs Baseline")
    ax.set_xlabel("Training Steps")
    ax.set_ylabel("Success Rate (%)")
    ax.set_xticks(range(len(all_steps)))
    ax.set_xticklabels(x_labels)
    ax.set_ylim(0, 100)
    ax.set_xlim(-0.35, len(all_steps) - 0.65)
    ax.grid(True, color="#B0B0B0", alpha=0.28, linewidth=1.0)
    ax.legend(loc="lower right", frameon=True, framealpha=0.92)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    if show_inset and max_y > min_y:
        zoom_min = max(0, math.floor((min_y - 6) / 5) * 5)
        zoom_max = min(100, math.ceil((max_y + 6) / 5) * 5)
        rect = Rectangle(
            (-0.18, zoom_min),
            len(all_steps) - 0.64,
            zoom_max - zoom_min,
            fill=False,
            linestyle=(0, (4, 3)),
            linewidth=1.2,
            edgecolor="#666666",
        )
        ax.add_patch(rect)

        inset = ax.inset_axes([0.16, 0.52, 0.42, 0.38])
        for spec in specs:
            points = results.get(spec.key, [])
            if not points:
                continue
            inset.plot(
                [step_to_x[point.step] for point in points],
                [point.success_rate for point in points],
                marker=spec.marker,
                markersize=5,
                linewidth=2.0,
                color=spec.color,
            )
        inset.set_xlim(-0.25, len(all_steps) - 0.75)
        inset.set_ylim(zoom_min, zoom_max)
        inset.set_xticks(range(len(all_steps)))
        inset.set_xticklabels(x_labels, fontsize=9)
        inset.tick_params(axis="y", labelsize=9)
        inset.grid(True, color="#B0B0B0", alpha=0.24, linewidth=0.8)
        for spine in inset.spines.values():
            spine.set_edgecolor("#666666")
            spine.set_linewidth(1.0)

    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=dpi, bbox_inches="tight")

    if output.suffix.lower() != ".pdf":
        fig.savefig(output.with_suffix(".pdf"), bbox_inches="tight")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot baseline vs RLAS-Group success rates from RoboTwin eval metrics."
    )
    parser.add_argument(
        "--result-dir",
        type=Path,
        default=Path("eval_result/open_laptop/DP/demo_clean"),
        help="Directory containing demo_clean-u* and demo_clean-rlas-group-u* eval results.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output figure path. Defaults to RESULT_DIR/rlas_group_vs_baseline_success.png.",
    )
    parser.add_argument("--dpi", type=int, default=300, help="PNG output DPI.")
    parser.add_argument("--no-inset", action="store_true", help="Disable the zoomed inset.")
    parser.add_argument("--no-annotate", action="store_true", help="Disable point value labels.")
    args = parser.parse_args()

    result_dir = args.result_dir.resolve()
    output = args.output
    if output is None:
        output = result_dir / "rlas_group_vs_baseline_success.png"
    else:
        output = output.resolve()

    results = collect_results(result_dir)
    specs = build_method_specs(result_dir.name)
    print_summary(results, specs)
    plot_results(
        result_dir=result_dir,
        results=results,
        output=output,
        dpi=args.dpi,
        show_inset=not args.no_inset,
        annotate=not args.no_annotate,
    )
    print(f"Saved figure: {output}")
    if output.suffix.lower() != ".pdf":
        print(f"Saved figure: {output.with_suffix('.pdf')}")


if __name__ == "__main__":
    main()
