#!/usr/bin/env python3
"""Plot paired O3 versus Best K-of-N results across experiment budgets.

The script discovers aggregate.csv files below outputs/1cll, selects the O3
and Best K-of-N file pair with the largest number of shared seeds for each
budget, and writes:

  outputs/analysis/o3_vs_best_k_of_n.svg
  outputs/analysis/o3_vs_best_k_of_n_paired_differences.csv
  outputs/analysis/o3_vs_best_k_of_n_summary.csv

The primary metric is max_of_K. Differences are paired as O3 - Best K-of-N,
so positive values are gains and negative values are losses. The SVG is
generated with only Python's standard library and can be opened directly in a
browser, image viewer, or vector graphics editor.
"""

from __future__ import annotations

import argparse
import csv
import html
import math
from pathlib import Path
from typing import Iterable


METHOD_O3 = "O3"
METHOD_BASELINE = "Best K-of-N"
METRICS = ("max_of_K", "mean_of_K")


def seed_column(fieldnames: Iterable[str]) -> str:
    fields = set(fieldnames)
    for candidate in ("seed", "run_seed"):
        if candidate in fields:
            return candidate
    raise ValueError("aggregate.csv has neither 'seed' nor 'run_seed'")


def method_for(path: Path) -> str | None:
    parts = set(path.parts)
    if "best_k_of_n" in parts:
        return METHOD_BASELINE
    if "o3" in parts:
        return METHOD_O3
    return None


def read_aggregate(path: Path) -> tuple[tuple[int, int], dict[str, dict[str, float]]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"No rows in {path}")
    column = seed_column(rows[0].keys())
    n = int(rows[0]["N"])
    k = int(rows[0]["K"])
    records: dict[str, dict[str, float]] = {}
    for row in rows:
        records[row[column]] = {metric: float(row[metric]) for metric in METRICS}
    return (n, k), records


def choose_pairs(root: Path) -> list[dict[str, object]]:
    candidates: dict[tuple[tuple[int, int], str], list[tuple[Path, dict[str, dict[str, float]]]]] = {}
    for path in sorted(root.glob("**/aggregate.csv")):
        method = method_for(path)
        if method is None:
            continue
        budget, records = read_aggregate(path)
        candidates.setdefault((budget, method), []).append((path, records))

    budgets = sorted({budget for budget, _method in candidates})
    selected: list[dict[str, object]] = []
    for budget in budgets:
        o3_files = candidates.get((budget, METHOD_O3), [])
        baseline_files = candidates.get((budget, METHOD_BASELINE), [])
        if not o3_files or not baseline_files:
            continue
        pair_options = []
        for o3_path, o3_records in o3_files:
            for baseline_path, baseline_records in baseline_files:
                shared = sorted(set(o3_records) & set(baseline_records))
                pair_options.append((len(shared), o3_path, o3_records, baseline_path, baseline_records, shared))
        _count, o3_path, o3_records, baseline_path, baseline_records, shared = max(
            pair_options,
            key=lambda item: (item[0], str(item[1]), str(item[3])),
        )
        if not shared:
            continue
        selected.append(
            {
                "N": budget[0],
                "K": budget[1],
                "o3_path": o3_path,
                "baseline_path": baseline_path,
                "o3": o3_records,
                "baseline": baseline_records,
                "seeds": shared,
            }
        )
    return selected


def paired_rows(selections: list[dict[str, object]]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for selection in selections:
        n = int(selection["N"])
        k = int(selection["K"])
        o3 = selection["o3"]
        baseline = selection["baseline"]
        for seed in selection["seeds"]:
            row: dict[str, object] = {"N": n, "K": k, "seed": seed}
            for metric in METRICS:
                o3_score = float(o3[seed][metric])
                baseline_score = float(baseline[seed][metric])
                row[f"o3_{metric}"] = o3_score
                row[f"best_k_of_n_{metric}"] = baseline_score
                row[f"delta_{metric}"] = o3_score - baseline_score
            rows.append(row)
    return rows


def mean(values: list[float]) -> float:
    return sum(values) / len(values)


def sample_sd(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    center = mean(values)
    return math.sqrt(sum((value - center) ** 2 for value in values) / (len(values) - 1))


def summary(values: list[float]) -> dict[str, float]:
    center = mean(values)
    sd = sample_sd(values)
    # t=2.776 for n=5 and t=3.182 for n=3; this small-sample correction is
    # preferable to presenting a normal-theory interval as if these were large
    # samples. For n=1 the interval is zero and is labeled n=1 in the figure.
    t_critical = {1: 0.0, 2: 12.706, 3: 4.303, 4: 3.182, 5: 2.776}.get(len(values), 1.96)
    half_ci = t_critical * sd / math.sqrt(len(values)) if values else 0.0
    return {"mean": center, "sd": sd, "ci": half_ci, "min": min(values), "max": max(values)}


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "N", "K", "seed",
        "o3_max_of_K", "best_k_of_n_max_of_K", "delta_max_of_K",
        "o3_mean_of_K", "best_k_of_n_mean_of_K", "delta_mean_of_K",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_summary_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "N", "K", "n_pairs", "metric", "mean_delta", "sd_delta",
        "ci95_half_width", "min_delta", "max_delta", "wins", "win_rate",
    ]
    output: list[dict[str, object]] = []
    for n, k in sorted({(int(row["N"]), int(row["K"])) for row in rows}):
        group = [row for row in rows if int(row["N"]) == n and int(row["K"]) == k]
        for metric in METRICS:
            deltas = [float(row[f"delta_{metric}"]) for row in group]
            stats = summary(deltas)
            output.append(
                {
                    "N": n,
                    "K": k,
                    "n_pairs": len(deltas),
                    "metric": metric,
                    "mean_delta": stats["mean"],
                    "sd_delta": stats["sd"],
                    "ci95_half_width": stats["ci"],
                    "min_delta": stats["min"],
                    "max_delta": stats["max"],
                    "wins": sum(delta >= 0 for delta in deltas),
                    "win_rate": sum(delta >= 0 for delta in deltas) / len(deltas),
                }
            )
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(output)


def esc(value: object) -> str:
    return html.escape(str(value), quote=True)


def fmt(value: float) -> str:
    return f"{value:.3f}"


def svg_text(x: float, y: float, text: str, *, size: int = 13, anchor: str = "start", weight: str = "400", fill: str = "#243142") -> str:
    return f'<text x="{x:.1f}" y="{y:.1f}" font-family="Arial, Helvetica, sans-serif" font-size="{size}px" font-weight="{weight}" text-anchor="{anchor}" fill="{fill}">{esc(text)}</text>'


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
    ylabel: str,
    selections: list[dict[str, object]],
    rows: list[dict[str, object]],
    metric: str,
    difference: bool,
) -> None:
    left, right, top, bottom = x + 58, x + width - 20, y + 42, y + height - 42
    values: list[float] = []
    interval_values: list[float] = []
    grouped: list[tuple[dict[str, object], list[dict[str, object]]]] = []
    for selection in selections:
        n, k = int(selection["N"]), int(selection["K"])
        budget_rows = [row for row in rows if int(row["N"]) == n and int(row["K"]) == k]
        grouped.append((selection, budget_rows))
        if difference:
            scores = [float(row[f"delta_{metric}"]) for row in budget_rows]
            values.extend(scores)
            stats = summary(scores)
            interval_values.extend((stats["mean"] - stats["ci"], stats["mean"] + stats["ci"]))
        else:
            for prefix in ("o3", "best_k_of_n"):
                scores = [float(row[f"{prefix}_{metric}"]) for row in budget_rows]
                values.extend(scores)
                stats = summary(scores)
                interval_values.extend((stats["mean"] - stats["ci"], stats["mean"] + stats["ci"]))

    if difference:
        interval_low = min(interval_values) if interval_values else -0.1
        interval_high = max(interval_values) if interval_values else 0.1
        maximum = max(abs(interval_low), abs(interval_high), 0.02)
        padding = max(0.012, maximum * 0.14)
        y_low, y_high = min(-0.02, interval_low - padding), max(0.02, interval_high + padding)
    else:
        interval_high = max(interval_values) if interval_values else 1.0
        # Scores are shown as bars from zero, which makes the bar heights
        # directly interpretable instead of magnifying small differences.
        y_low = 0.0
        y_high = min(1.10, interval_high + max(0.02, interval_high * 0.08))

    out.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{width:.1f}" height="{height:.1f}" rx="8" fill="#ffffff" stroke="#d8dee8"/>')
    out.append(svg_text(x + 16, y + 25, title, size=15, weight="700"))

    ticks = 5
    for i in range(ticks + 1):
        value = y_low + (y_high - y_low) * i / ticks
        py = scale(value, y_low, y_high, bottom, top)
        out.append(f'<line x1="{left:.1f}" y1="{py:.1f}" x2="{right:.1f}" y2="{py:.1f}" stroke="#e7ebf0"/>')
        out.append(svg_text(left - 10, py + 4, fmt(value), size=10, anchor="end", fill="#5b6778"))

    if difference and y_low <= 0 <= y_high:
        zero = scale(0, y_low, y_high, bottom, top)
        out.append(f'<line x1="{left:.1f}" y1="{zero:.1f}" x2="{right:.1f}" y2="{zero:.1f}" stroke="#687587" stroke-width="1.4" stroke-dasharray="5,4"/>')

    out.append(f'<line x1="{left:.1f}" y1="{top:.1f}" x2="{left:.1f}" y2="{bottom:.1f}" stroke="#7c8795"/>')
    out.append(f'<line x1="{left:.1f}" y1="{bottom:.1f}" x2="{right:.1f}" y2="{bottom:.1f}" stroke="#7c8795"/>')
    out.append(f'<g transform="translate({x + 15:.1f},{(top + bottom) / 2:.1f}) rotate(-90)">{svg_text(0, 0, ylabel, size=11, anchor="middle")}</g>')

    colors = {"O3": "#1769aa", "Best K-of-N": "#e07a24"}
    count = max(len(grouped), 1)
    group_width = (right - left) / count
    for index, (selection, budget_rows) in enumerate(grouped):
        center_x = left + group_width * (index + 0.5)
        n, k = int(selection["N"]), int(selection["K"])
        label = f"N={n}\nK={k}\nn={len(budget_rows)}"
        for line_index, line in enumerate(label.splitlines()):
            out.append(svg_text(center_x, bottom + 19 + line_index * 13, line, size=10, anchor="middle", fill="#5b6778"))
        if difference:
            deltas = [float(row[f"delta_{metric}"]) for row in budget_rows]
            stats = summary(deltas)
            mean_y = scale(stats["mean"], y_low, y_high, bottom, top)
            zero_y = scale(0, y_low, y_high, bottom, top)
            ci_low_y = scale(stats["mean"] - stats["ci"], y_low, y_high, bottom, top)
            ci_high_y = scale(stats["mean"] + stats["ci"], y_low, y_high, bottom, top)
            bar_color = "#1769aa" if stats["mean"] >= 0 else "#b33a3a"
            bar_top = min(zero_y, mean_y)
            bar_height = max(1.0, abs(mean_y - zero_y))
            out.append(f'<rect x="{center_x - 20:.1f}" y="{bar_top:.1f}" width="40" height="{bar_height:.1f}" fill="{bar_color}" fill-opacity="0.86"/>')
            out.append(f'<line x1="{center_x:.1f}" y1="{ci_low_y:.1f}" x2="{center_x:.1f}" y2="{ci_high_y:.1f}" stroke="#1f2937" stroke-width="2.2"/>')
            out.append(f'<line x1="{center_x - 7:.1f}" y1="{ci_low_y:.1f}" x2="{center_x + 7:.1f}" y2="{ci_low_y:.1f}" stroke="#1f2937" stroke-width="2"/>')
            out.append(f'<line x1="{center_x - 7:.1f}" y1="{ci_high_y:.1f}" x2="{center_x + 7:.1f}" y2="{ci_high_y:.1f}" stroke="#1f2937" stroke-width="2"/>')
            label_y = mean_y - 10 if stats["mean"] >= 0 else mean_y + 18
            out.append(svg_text(center_x, label_y, f"Δ={stats['mean']:+.3f}", size=10, anchor="middle", weight="700", fill="#344054"))
            out.append(svg_text(center_x, label_y + (13 if stats["mean"] >= 0 else -13), f"{sum(v >= 0 for v in deltas)}/{len(deltas)} wins", size=9, anchor="middle", fill="#526071"))
        else:
            for method, offset in ((METHOD_BASELINE, -10), (METHOD_O3, 10)):
                scores = [float(row[f"{'o3' if method == METHOD_O3 else 'best_k_of_n'}_{metric}"]) for row in budget_rows]
                stats = summary(scores)
                px = center_x + offset
                py = scale(stats["mean"], y_low, y_high, bottom, top)
                ci_low_y = scale(stats["mean"] - stats["ci"], y_low, y_high, bottom, top)
                ci_high_y = scale(stats["mean"] + stats["ci"], y_low, y_high, bottom, top)
                zero_y = scale(0, y_low, y_high, bottom, top)
                bar_top = min(zero_y, py)
                bar_height = max(1.0, abs(py - zero_y))
                out.append(f'<rect x="{px - 12:.1f}" y="{bar_top:.1f}" width="24" height="{bar_height:.1f}" fill="{colors[method]}" fill-opacity="0.86"/>')
                out.append(f'<line x1="{px:.1f}" y1="{ci_low_y:.1f}" x2="{px:.1f}" y2="{ci_high_y:.1f}" stroke="{colors[method]}" stroke-width="2"/>')
                out.append(f'<line x1="{px - 5:.1f}" y1="{ci_low_y:.1f}" x2="{px + 5:.1f}" y2="{ci_low_y:.1f}" stroke="{colors[method]}" stroke-width="2"/>')
                out.append(f'<line x1="{px - 5:.1f}" y1="{ci_high_y:.1f}" x2="{px + 5:.1f}" y2="{ci_high_y:.1f}" stroke="{colors[method]}" stroke-width="2"/>')
                label_y = max(top + 14, py - 11)
                out.append(svg_text(px, label_y, f"μ={stats['mean']:.3f}", size=9, anchor="middle", weight="700", fill=colors[method]))


def write_svg(path: Path, selections: list[dict[str, object]], rows: list[dict[str, object]]) -> None:
    width, height = 1120, 900
    out = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#f7f9fc"/>',
        svg_text(40, 38, "O3 versus Best K-of-N across budgets", size=23, weight="700", fill="#172033"),
        svg_text(40, 62, "Bar height = across-seed mean; whiskers = 95% t-confidence interval. Paired differences are O3 − Best K-of-N.", size=12, fill="#526071"),
    ]
    panel_w, panel_h = 510, 330
    draw_panel(out, x=35, y=85, width=panel_w, height=panel_h, title="A  Grouped mean bars: max-of-K", ylabel="max_of_K (TM-score)", selections=selections, rows=rows, metric="max_of_K", difference=False)
    draw_panel(out, x=575, y=85, width=panel_w, height=panel_h, title="B  Mean difference bars: max-of-K", ylabel="Δ max_of_K", selections=selections, rows=rows, metric="max_of_K", difference=True)
    draw_panel(out, x=35, y=445, width=panel_w, height=panel_h, title="C  Grouped mean bars: mean-of-K", ylabel="mean_of_K (TM-score)", selections=selections, rows=rows, metric="mean_of_K", difference=False)
    draw_panel(out, x=575, y=445, width=panel_w, height=panel_h, title="D  Mean difference bars: mean-of-K", ylabel="Δ mean_of_K", selections=selections, rows=rows, metric="mean_of_K", difference=True)
    out.extend([
        svg_text(40, 850, "Chart key:", size=12, weight="700"),
        '<rect x="112" y="839" width="12" height="12" fill="#1769aa" fill-opacity="0.86"/>',
        svg_text(132, 850, "bar = mean (μ)", size=11),
        '<line x1="330" y1="842" x2="330" y2="851" stroke="#1f2937" stroke-width="2"/><line x1="325" y1="842" x2="335" y2="842" stroke="#1f2937" stroke-width="2"/><line x1="325" y1="851" x2="335" y2="851" stroke="#1f2937" stroke-width="2"/>',
        svg_text(345, 850, "95% t-CI", size=11),
        '<line x1="430" y1="846" x2="475" y2="846" stroke="#687587" stroke-width="1.4" stroke-dasharray="5,4"/>',
        svg_text(487, 850, "zero difference", size=11),
        '<circle cx="700" cy="846" r="5" fill="#1769aa"/><text x="713" y="850" font-family="Arial" font-size="11px" fill="#344054">O3 / positive Δ</text>',
        '<circle cx="835" cy="846" r="5" fill="#e07a24"/><text x="848" y="850" font-family="Arial" font-size="11px" fill="#344054">Best K-of-N</text>',
        '<circle cx="965" cy="846" r="5" fill="#b33a3a"/><text x="978" y="850" font-family="Arial" font-size="11px" fill="#344054">negative Δ</text>',
        svg_text(40, 878, "Interpretation: μ is the across-seed mean; whiskers are 95% t-confidence intervals; positive Δ favors O3. n is the number of paired seeds.", size=11, fill="#526071"),
        '</svg>',
    ])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(out), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("outputs/1cll"), help="Output root to scan (default: outputs/1cll)")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/analysis"), help="Directory for chart and paired CSV")
    args = parser.parse_args()

    selections = choose_pairs(args.root)
    if not selections:
        raise SystemExit(f"No paired O3/Best K-of-N aggregate files found below {args.root}")
    rows = paired_rows(selections)
    svg_path = args.output_dir / "o3_vs_best_k_of_n.svg"
    csv_path = args.output_dir / "o3_vs_best_k_of_n_paired_differences.csv"
    summary_path = args.output_dir / "o3_vs_best_k_of_n_summary.csv"
    write_svg(svg_path, selections, rows)
    write_csv(csv_path, rows)
    write_summary_csv(summary_path, rows)

    print(f"Wrote {svg_path}")
    print(f"Wrote {csv_path}")
    print(f"Wrote {summary_path}")
    for selection in selections:
        print(
            f"N={selection['N']}, K={selection['K']}: {len(selection['seeds'])} paired seeds\n"
            f"  O3: {selection['o3_path']}\n"
            f"  Best K-of-N: {selection['baseline_path']}"
        )


if __name__ == "__main__":
    main()
