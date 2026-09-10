#!/usr/bin/env python3
"""Create a presentation-ready three-method 1CLL comparison bar chart.

The chart consumes the seed-aligned ``comparison_summary.csv`` produced by
``experiments/1cll/compare_methods.py``. It writes a vector SVG with grouped
bars for mean-of-K and max-of-K, 95% t-confidence whiskers, a compact numeric
table, and a CSV containing the chart statistics.

Example:

    python scripts/plot_all_methods.py \
        --summary outputs/1cll/k10_n100/runs/RUN_ID/comparison_summary.csv \
        --output-dir outputs/analysis/RUN_ID

The implementation uses only the Python standard library, so it does not
require matplotlib or a GUI backend on the lab machine.
"""

from __future__ import annotations

import argparse
import csv
import html
import math
from pathlib import Path
from typing import Iterable


METHODS = (
    ("best_k_of_n", "Best K-of-N", "#667085"),
    ("o3", "O3", "#1769AA"),
    ("random_pfode", "Random PF-ODE", "#D97706"),
)
METRICS = ("mean_of_K", "max_of_K")
T_CRITICAL_95 = {
    1: 0.0,
    2: 12.706,
    3: 4.303,
    4: 3.182,
    5: 2.776,
    6: 2.571,
    7: 2.447,
    8: 2.365,
    9: 2.306,
    10: 2.262,
    11: 2.201,
    12: 2.179,
    13: 2.160,
    14: 2.145,
    15: 2.131,
    16: 2.120,
    17: 2.110,
    18: 2.101,
    19: 2.093,
    20: 2.086,
}


def mean(values: list[float]) -> float:
    return sum(values) / len(values)


def sample_sd(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    center = mean(values)
    return math.sqrt(sum((value - center) ** 2 for value in values) / (len(values) - 1))


def ci95(values: list[float]) -> float:
    critical = T_CRITICAL_95.get(len(values), 1.96)
    return critical * sample_sd(values) / math.sqrt(len(values))


def read_summary(path: Path) -> tuple[tuple[int, int], list[dict[str, object]], list[str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"No rows in comparison summary: {path}")

    fieldnames = set(rows[0])
    methods = [
        method
        for method, _label, _color in METHODS
        if f"{method}_mean_of_K" in fieldnames and f"{method}_max_of_K" in fieldnames
    ]
    expected = [method for method, _label, _color in METHODS]
    missing = [method for method in expected if method not in methods]
    if missing:
        raise ValueError(
            f"Summary is missing methods {missing}; expected columns for {expected}: {path}"
        )

    n_values = {int(row["N"]) for row in rows}
    k_values = {int(row["K"]) for row in rows}
    if len(n_values) != 1 or len(k_values) != 1:
        raise ValueError("A presentation chart requires one N/K budget")
    n, k = n_values.pop(), k_values.pop()

    parsed: list[dict[str, object]] = []
    for row in rows:
        parsed_row: dict[str, object] = {"seed": int(row["seed"])}
        for method in methods:
            for metric in METRICS:
                parsed_row[f"{method}_{metric}"] = float(row[f"{method}_{metric}"])
        parsed.append(parsed_row)
    return (n, k), parsed, methods


def build_stats(
    budget: tuple[int, int], rows: list[dict[str, object]], methods: list[str]
) -> list[dict[str, object]]:
    n, k = budget
    output: list[dict[str, object]] = []
    for method in methods:
        metric_values = {
            metric: [float(row[f"{method}_{metric}"]) for row in rows]
            for metric in METRICS
        }
        means = {metric: mean(values) for metric, values in metric_values.items()}
        output.append(
            {
                "method": method,
                "N": n,
                "K": k,
                "replicates": len(rows),
                "mean_of_K": means["mean_of_K"],
                "mean_of_K_sd": sample_sd(metric_values["mean_of_K"]),
                "mean_of_K_ci95": ci95(metric_values["mean_of_K"]),
                "mean_of_K_best": max(metric_values["mean_of_K"]),
                "max_of_K": means["max_of_K"],
                "max_of_K_sd": sample_sd(metric_values["max_of_K"]),
                "max_of_K_ci95": ci95(metric_values["max_of_K"]),
                "max_of_K_best": max(metric_values["max_of_K"]),
            }
        )
    return output


def write_stats(path: Path, stats: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(stats[0])
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(stats)


def esc(value: object) -> str:
    return html.escape(str(value), quote=True)


def text(
    x: float,
    y: float,
    value: str,
    *,
    size: int = 13,
    anchor: str = "start",
    weight: str = "400",
    fill: str = "#344054",
) -> str:
    return (
        f'<text x="{x:.1f}" y="{y:.1f}" font-family="Arial, Helvetica, sans-serif" '
        f'font-size="{size}px" font-weight="{weight}" text-anchor="{anchor}" '
        f'fill="{fill}">{esc(value)}</text>'
    )


def scale(value: float, low: float, high: float, pixel_low: float, pixel_high: float) -> float:
    if high == low:
        return (pixel_low + pixel_high) / 2
    return pixel_low + (value - low) * (pixel_high - pixel_low) / (high - low)


def draw_panel(
    out: list[str],
    *,
    x: float,
    y: float,
    width: float,
    height: float,
    title: str,
    metric: str,
    stats: list[dict[str, object]],
) -> None:
    left, right = x + 72, x + width - 28
    top, bottom = y + 58, y + height - 60
    y_low, y_high = 0.0, 1.0

    out.append(
        f'<rect x="{x:.1f}" y="{y:.1f}" width="{width:.1f}" height="{height:.1f}" '
        'rx="12" fill="#ffffff" stroke="#d0d5dd"/>'
    )
    out.append(text(x + 24, y + 35, title, size=17, weight="700", fill="#172033"))
    out.append(text(x + width - 24, y + 35, "higher is better", size=11, anchor="end", fill="#667085"))

    for tick in range(6):
        value = tick / 5
        py = scale(value, y_low, y_high, bottom, top)
        out.append(
            f'<line x1="{left:.1f}" y1="{py:.1f}" x2="{right:.1f}" y2="{py:.1f}" '
            f'stroke="{"#d0d5dd" if tick == 0 else "#eaecf0"}" stroke-width="{"1.3" if tick == 0 else "1"}"/>'
        )
        out.append(text(left - 12, py + 4, f"{value:.1f}", size=11, anchor="end", fill="#667085"))

    out.append(
        f'<line x1="{left:.1f}" y1="{top:.1f}" x2="{left:.1f}" y2="{bottom:.1f}" stroke="#98a2b3"/>'
    )
    out.append(
        f'<line x1="{left:.1f}" y1="{bottom:.1f}" x2="{right:.1f}" y2="{bottom:.1f}" stroke="#98a2b3"/>'
    )
    out.append(
        f'<g transform="translate({x + 22:.1f},{(top + bottom) / 2:.1f}) rotate(-90)">'
        + text(0, 0, "TM-score", size=12, anchor="middle")
        + "</g>"
    )

    bar_width = 82.0
    group_width = (right - left) / len(stats)
    for index, item in enumerate(stats):
        method = str(item["method"])
        label = next(label for key, label, _color in METHODS if key == method)
        color = next(color for key, _label, color in METHODS if key == method)
        value = float(item[metric])
        ci = float(item[f"{metric}_ci95"])
        center = left + group_width * (index + 0.5)
        bar_left = center - bar_width / 2
        bar_top = scale(value, y_low, y_high, bottom, top)
        error_low = scale(max(y_low, value - ci), y_low, y_high, bottom, top)
        error_high = scale(min(y_high, value + ci), y_low, y_high, bottom, top)
        out.append(
            f'<rect x="{bar_left:.1f}" y="{bar_top:.1f}" width="{bar_width:.1f}" '
            f'height="{bottom - bar_top:.1f}" rx="5" fill="{color}"/>'
        )
        out.append(
            f'<line x1="{center:.1f}" y1="{error_low:.1f}" x2="{center:.1f}" y2="{error_high:.1f}" '
            'stroke="#1d2939" stroke-width="2.5"/>'
        )
        out.append(
            f'<line x1="{center - 10:.1f}" y1="{error_low:.1f}" x2="{center + 10:.1f}" y2="{error_low:.1f}" '
            'stroke="#1d2939" stroke-width="2.5"/>'
        )
        out.append(
            f'<line x1="{center - 10:.1f}" y1="{error_high:.1f}" x2="{center + 10:.1f}" y2="{error_high:.1f}" '
            'stroke="#1d2939" stroke-width="2.5"/>'
        )
        # Place the value inside the bar so it cannot collide with the CI cap
        # when a confidence interval is large.
        out.append(text(center, bar_top + 22, f"{value:.3f}", size=13, anchor="middle", weight="700", fill="#ffffff"))
        out.append(text(center, bottom + 24, label, size=12, anchor="middle", weight="600", fill="#344054"))
        out.append(text(center, bottom + 42, f"n={item['replicates']}", size=11, anchor="middle", fill="#667085"))


def write_svg(
    path: Path,
    *,
    budget: tuple[int, int],
    stats: list[dict[str, object]],
    run_label: str,
) -> None:
    width, height = 1400, 900
    n, k = budget
    out = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#f8fafc"/>',
        text(56, 58, "1CLL oracle-budget comparison", size=28, weight="700", fill="#101828"),
        text(56, 89, f"N={n}, K={k}  •  {len(stats) and stats[0]['replicates']} paired replicates  •  {run_label}", size=14, fill="#667085"),
    ]

    draw_panel(out, x=44, y=122, width=640, height=425, title="Mean-of-K TM-score", metric="mean_of_K", stats=stats)
    draw_panel(out, x=716, y=122, width=640, height=425, title="Max-of-K TM-score", metric="max_of_K", stats=stats)

    table_x, table_y, table_w, row_h = 44, 590, 1312, 42
    out.append(
        f'<rect x="{table_x:.1f}" y="{table_y:.1f}" width="{table_w:.1f}" height="{row_h * (len(stats) + 1):.1f}" '
        'rx="10" fill="#ffffff" stroke="#d0d5dd"/>'
    )
    columns = [
        (table_x + 24, "Method", "start"),
        (table_x + 350, "Mean-of-K", "end"),
        (table_x + 520, "95% CI", "end"),
        (table_x + 820, "Max-of-K", "end"),
        (table_x + 990, "95% CI", "end"),
        (table_x + 1275, "Mean Δ vs baseline", "end"),
    ]
    for x, label, anchor in columns:
        out.append(text(x, table_y + 27, label, size=11, anchor=anchor, weight="700", fill="#667085"))
    out.append(f'<line x1="{table_x + 20:.1f}" y1="{table_y + row_h:.1f}" x2="{table_x + table_w - 20:.1f}" y2="{table_y + row_h:.1f}" stroke="#eaecf0"/>')

    baseline = next(item for item in stats if item["method"] == "best_k_of_n")
    baseline_mean = float(baseline["mean_of_K"])
    for index, item in enumerate(stats):
        row_top = table_y + row_h * (index + 1)
        if index % 2 == 1:
            out.append(f'<rect x="{table_x + 1:.1f}" y="{row_top:.1f}" width="{table_w - 2:.1f}" height="{row_h:.1f}" fill="#f8fafc"/>')
        method = str(item["method"])
        label = next(label for key, label, _color in METHODS if key == method)
        color = next(color for key, _label, color in METHODS if key == method)
        delta = float(item["mean_of_K"]) - baseline_mean
        out.append(text(columns[0][0], row_top + 27, label, size=12, weight="600", fill=color))
        out.append(text(columns[1][0], row_top + 27, f"{float(item['mean_of_K']):.3f}", size=12, anchor="end", fill="#172033"))
        out.append(text(columns[2][0], row_top + 27, f"± {float(item['mean_of_K_ci95']):.3f}", size=12, anchor="end", fill="#667085"))
        out.append(text(columns[3][0], row_top + 27, f"{float(item['max_of_K']):.3f}", size=12, anchor="end", fill="#172033"))
        out.append(text(columns[4][0], row_top + 27, f"± {float(item['max_of_K_ci95']):.3f}", size=12, anchor="end", fill="#667085"))
        out.append(text(columns[5][0], row_top + 27, "baseline" if method == "best_k_of_n" else f"{delta:+.3f}", size=12, anchor="end", weight="600", fill="#667085" if method == "best_k_of_n" else "#1769AA"))

    out.extend(
        [
            text(56, 790, "Chart notes", size=12, weight="700", fill="#344054"),
            text(56, 816, "Bars show across-seed means; whiskers show 95% t-confidence intervals. All methods use the same paired replicate seeds.", size=12, fill="#667085"),
            text(56, 840, "Primary metric: mean-of-K, the mean TM-score among the K selected structures. Higher TM-score is better.", size=12, fill="#667085"),
            text(56, 870, "Generated from the seed-aligned comparison summary; this is a descriptive comparison, not a significance test.", size=11, fill="#98A2B3"),
            '</svg>',
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(out), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, required=True, help="Seed-aligned comparison_summary.csv")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/analysis"))
    parser.add_argument("--basename", default="all_methods_comparison")
    parser.add_argument("--run-label", default=None, help="Subtitle label, such as a run ID")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    budget, rows, methods = read_summary(args.summary)
    stats = build_stats(budget, rows, methods)
    run_label = args.run_label or args.summary.parent.name
    svg_path = args.output_dir / f"{args.basename}.svg"
    stats_path = args.output_dir / f"{args.basename}_stats.csv"
    write_svg(svg_path, budget=budget, stats=stats, run_label=run_label)
    write_stats(stats_path, stats)
    print(f"Wrote {svg_path}")
    print(f"Wrote {stats_path}")
    for item in stats:
        print(
            f"{item['method']}: mean_of_K={float(item['mean_of_K']):.4f} "
            f"± {float(item['mean_of_K_ci95']):.4f}; "
            f"max_of_K={float(item['max_of_K']):.4f} ± {float(item['max_of_K_ci95']):.4f}"
        )


if __name__ == "__main__":
    main()
