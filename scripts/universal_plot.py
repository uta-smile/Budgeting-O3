#!/usr/bin/env python3

import argparse
import math
import re
from pathlib import Path
from plotly.colors import qualitative
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


# ============================================================
# CLI
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Generate professional grouped bar charts comparing "
            "multiple methods across shared metrics in a CSV file."
        )
    )

    parser.add_argument(
        "--input",
        "-i",
        required=True,
        type=Path,
        help="Path to input CSV.",
    )

    parser.add_argument(
        "--output-dir",
        "-o",
        required=True,
        type=Path,
        help="Directory for generated charts.",
    )

    parser.add_argument(
        "--x-col",
        default="seed",
        help="Column used for the x-axis. Default: seed",
    )

    parser.add_argument(
        "--group-by",
        default="target",
        help=(
            "Column used to generate separate figures. "
            "Default: target. Use 'none' for one chart."
        ),
    )

    parser.add_argument(
        "--min-methods",
        type=int,
        default=2,
        help=(
            "Minimum number of methods sharing a metric "
            "for that metric to be plotted. Default: 2"
        ),
    )

    parser.add_argument(
        "--exclude",
        nargs="*",
        default=[],
        help="Additional columns to ignore.",
    )

    parser.add_argument(
        "--metrics",
        nargs="*",
        default=None,
        help=(
            "Optional metric whitelist. Example: "
            "--metrics total_mean mean_of_K max_of_K"
        ),
    )

    parser.add_argument(
        "--methods",
        nargs="*",
        default=None,
        help=(
            "Optional method whitelist. Example: "
            "--methods stochastic pfode diffusion"
        ),
    )

    parser.add_argument(
        "--format",
        choices=["html", "png", "pdf", "svg", "all"],
        default="html",
        help=(
            "Output format. Default: html. "
            "PNG/PDF/SVG require kaleido."
        ),
    )

    parser.add_argument(
        "--cols",
        type=int,
        default=3,
        help="Maximum number of subplot columns. Default: 3",
    )

    parser.add_argument(
        "--title",
        type=str,
        default=None,
        help="Optional custom figure title.",
    )

    parser.add_argument(
        "--width",
        type=int,
        default=None,
        help="Optional figure width in pixels.",
    )

    parser.add_argument(
        "--height",
        type=int,
        default=None,
        help="Optional figure height in pixels.",
    )

    parser.add_argument(
        "--scale",
        type=float,
        default=2.0,
        help="PNG export scale factor. Default: 2.0",
    )

    parser.add_argument(
        "--y-min",
        type=float,
        default=None,
        help="Optional fixed minimum y-axis value.",
    )

    parser.add_argument(
        "--y-max",
        type=float,
        default=None,
        help="Optional fixed maximum y-axis value.",
    )

    parser.add_argument(
        "--no-labels",
        action="store_true",
        help="Disable numeric value labels above bars.",
    )

    parser.add_argument(
        "--precision",
        type=int,
        default=3,
        help="Decimal places shown on bar labels. Default: 3",
    )

    return parser.parse_args()


# ============================================================
# Helpers
# ============================================================

def prettify(text):
    return str(text).replace("_", " ").strip().title()


def safe_filename(text):
    text = str(text)
    text = re.sub(r"[^\w\-.]+", "_", text)
    return text.strip("_")


def get_split_candidates(column):
    """
    Generate every possible METHOD_METRIC split.

    Example:
        flow_matching_total_mean

    candidates:
        flow | matching_total_mean
        flow_matching | total_mean
        flow_matching_total | mean
    """

    parts = column.split("_")
    candidates = []

    for i in range(1, len(parts)):
        method = "_".join(parts[:i])
        metric = "_".join(parts[i:])
        candidates.append((method, metric))

    return candidates


# ============================================================
# Automatic method / metric detection
# ============================================================

def infer_method_metric_columns(
    df,
    metadata_columns,
    min_methods=2,
):
    """
    Returns:

        {
            "total_mean": {
                "stochastic": "stochastic_total_mean",
                "pfode": "pfode_total_mean",
                "diffusion": "diffusion_total_mean",
            },
            ...
        }

    Prefers the longest shared metric suffix.

    This means:
        stochastic_total_mean
        pfode_total_mean

    is interpreted as:

        methods:
            stochastic
            pfode

        metric:
            total_mean

    rather than metric="mean".
    """

    candidate_columns = []

    for column in df.columns:

        if column in metadata_columns:
            continue

        if not pd.api.types.is_numeric_dtype(df[column]):
            continue

        if "_" not in column:
            continue

        candidate_columns.append(column)

    metric_candidates = {}

    for column in candidate_columns:

        for method, metric in get_split_candidates(column):

            if metric not in metric_candidates:
                metric_candidates[metric] = {}

            metric_candidates[metric][method] = column

    # Metrics must be shared by at least N methods
    valid_metrics = {
        metric: methods
        for metric, methods in metric_candidates.items()
        if len(methods) >= min_methods
    }

    if not valid_metrics:
        return {}

    # Prefer longest suffix
    sorted_metrics = sorted(
        valid_metrics,
        key=lambda metric: len(metric.split("_")),
        reverse=True,
    )

    assignments = {}

    for column in candidate_columns:

        for metric in sorted_metrics:

            suffix = "_" + metric

            if column.endswith(suffix):

                method = column[:-len(suffix)]

                if method:
                    assignments[column] = (method, metric)
                    break

    result = {}

    for column, (method, metric) in assignments.items():

        if metric not in result:
            result[metric] = {}

        result[metric][method] = column

    # Final safety filtering
    result = {
        metric: methods
        for metric, methods in result.items()
        if len(methods) >= min_methods
    }

    return result


# ============================================================
# Filtering
# ============================================================

def filter_metric_map(
    metric_map,
    metrics=None,
    methods=None,
    min_methods=2,
):

    result = {}

    for metric, method_map in metric_map.items():

        if metrics is not None and metric not in metrics:
            continue

        filtered_methods = method_map

        if methods is not None:
            filtered_methods = {
                method: column
                for method, column in method_map.items()
                if method in methods
            }

        if len(filtered_methods) >= min_methods:
            result[metric] = filtered_methods

    return result


# ============================================================
# Plot generation
# ============================================================

def create_plot(
    df,
    metric_map,
    x_col,
    group_name,
    custom_title,
    max_cols,
    width,
    height,
    show_labels,
    precision,
    y_min,
    y_max,
):

    metrics = sorted(metric_map.keys())

    if not metrics:
        raise ValueError("No comparable metrics found.")

    n_metrics = len(metrics)

    n_cols = min(max_cols, n_metrics)
    n_rows = math.ceil(n_metrics / n_cols)

    if width is None:
        width = max(650, 480 * n_cols)

    if height is None:
        height = max(450, 390 * n_rows)

    subplot_titles = [
        prettify(metric)
        for metric in metrics
    ]

    fig = make_subplots(
        rows=n_rows,
        cols=n_cols,
        subplot_titles=subplot_titles,
        horizontal_spacing=0.08,
        vertical_spacing=0.15,
    )

    if x_col in df.columns:
        try:
            df = df.sort_values(
                by=x_col,
                kind="stable",
            ).reset_index(drop=True)
        except Exception:
            df = df.reset_index(drop=True)

        x_values = df[x_col].astype(str).tolist()

    else:
        x_values = [
            str(i + 1)
            for i in range(len(df))
        ]

    all_methods = sorted({
        method
        for method_map in metric_map.values()
        for method in method_map.keys()
    })

    # Assign one permanent color to each method
    palette = qualitative.Plotly

    method_colors = {
        method: palette[i % len(palette)]
        for i, method in enumerate(all_methods)
    }

    legend_seen = set()

    for metric_index, metric in enumerate(metrics):

        row = metric_index // n_cols + 1
        col = metric_index % n_cols + 1

        method_map = metric_map[metric]

        for method in all_methods:

            if method not in method_map:
                continue

            column = method_map[method]

            values = pd.to_numeric(
                df[column],
                errors="coerce",
            )

            if show_labels:
                text = [
                    (
                        f"{value:.{precision}f}"
                        if pd.notna(value)
                        else ""
                    )
                    for value in values
                ]
            else:
                text = None

            show_legend = method not in legend_seen

            fig.add_trace(
            go.Bar(
                name=prettify(method),
                x=x_values,
                y=values,

                # Keep method color consistent across every subplot
                marker_color=method_colors[method],

                text=text,
                textposition="outside" if show_labels else None,
                cliponaxis=False,

                showlegend=show_legend,
                legendgroup=method,

                hovertemplate=(
                    "<b>%{x}</b><br>"
                    f"Method: {prettify(method)}<br>"
                    f"{prettify(metric)}: %{{y:.{precision}f}}"
                    "<extra></extra>"
                ),
            ),
            row=row,
            col=col,
        )

            legend_seen.add(method)

        # X-axis title
        fig.update_xaxes(
            title_text=(
                prettify(x_col)
                if x_col in df.columns
                else "Sample"
            ),
            row=row,
            col=col,
        )

        # Y-axis
        axis_args = {
            "title_text": "Metric Value"
            if col == 1
            else None,
            "showgrid": True,
            "gridcolor": "rgba(0,0,0,0.08)",
            "zeroline": False,
        }

        if y_min is not None or y_max is not None:

            lower = y_min
            upper = y_max

            if lower is None:
                lower = 0

            axis_args["range"] = [lower, upper]

        fig.update_yaxes(
            row=row,
            col=col,
            **axis_args,
        )

    # ========================================================
    # Title
    # ========================================================

    if custom_title:

        title = custom_title

        if group_name is not None:
            title += f" — {group_name}"

    elif group_name is not None:

        title = f"Method Comparison — {group_name}"

    else:

        title = "Method Comparison"

    # ========================================================
    # Global styling
    # ========================================================

    fig.update_layout(
        title={
            "text": title,
            "x": 0.5,
            "xanchor": "center",
            "font": {
                "size": 22,
            },
        },
        barmode="group",
        template="plotly_white",
        width=width,
        height=height,
        legend={
            "title": {
                "text": "Method"
            },
            "orientation": "h",
            "yanchor": "bottom",
            "y": 1.02,
            "xanchor": "center",
            "x": 0.5,
        },
        margin={
            "l": 70,
            "r": 40,
            "t": 120,
            "b": 70,
        },
        font={
            "size": 13,
        },
        bargap=0.22,
        bargroupgap=0.05,
    )

    return fig


# ============================================================
# Saving
# ============================================================

def save_figure(
    fig,
    output_dir,
    stem,
    output_format,
    scale,
):

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    saved = []

    formats = (
        ["html", "png", "pdf", "svg"]
        if output_format == "all"
        else [output_format]
    )

    for fmt in formats:

        output_path = output_dir / f"{stem}.{fmt}"

        if fmt == "html":

            fig.write_html(
                output_path,
                include_plotlyjs="cdn",
            )

        else:

            try:
                fig.write_image(
                    output_path,
                    scale=scale if fmt == "png" else 1,
                )

            except Exception as exc:

                raise RuntimeError(
                    f"\nFailed to export {fmt.upper()}.\n"
                    "Static image export requires Kaleido.\n\n"
                    "Install it with:\n"
                    "    pip install kaleido\n\n"
                    f"Original error:\n{exc}"
                )

        saved.append(output_path)

    return saved


# ============================================================
# Main
# ============================================================

def main():

    args = parse_args()

    # --------------------------------------------------------
    # Input validation
    # --------------------------------------------------------

    if not args.input.exists():

        raise FileNotFoundError(
            f"Input CSV not found: {args.input}"
        )

    if args.cols < 1:

        raise ValueError(
            "--cols must be >= 1"
        )

    if args.min_methods < 2:

        raise ValueError(
            "--min-methods should normally be >= 2"
        )

    df = pd.read_csv(args.input)

    if df.empty:

        raise ValueError(
            "Input CSV contains no rows."
        )

    # --------------------------------------------------------
    # Metadata columns
    # --------------------------------------------------------

    metadata_columns = set(args.exclude)

    if args.x_col in df.columns:
        metadata_columns.add(args.x_col)

    group_by = args.group_by

    if group_by.lower() == "none":

        group_by = None

    elif group_by in df.columns:

        metadata_columns.add(group_by)

    else:

        print(
            f"Warning: group-by column '{group_by}' "
            "was not found."
        )

        print(
            "Generating one figure for the entire CSV.\n"
        )

        group_by = None

    # --------------------------------------------------------
    # Infer metrics and methods
    # --------------------------------------------------------

    metric_map = infer_method_metric_columns(
        df=df,
        metadata_columns=metadata_columns,
        min_methods=args.min_methods,
    )

    metric_map = filter_metric_map(
        metric_map=metric_map,
        metrics=args.metrics,
        methods=args.methods,
        min_methods=args.min_methods,
    )

    if not metric_map:

        raise ValueError(
            "\nCould not automatically detect comparable methods.\n\n"
            "Expected column naming similar to:\n\n"
            "    stochastic_total_mean\n"
            "    pfode_total_mean\n"
            "    diffusion_total_mean\n\n"
            "and optionally:\n\n"
            "    stochastic_mean_of_K\n"
            "    pfode_mean_of_K\n"
            "    diffusion_mean_of_K\n\n"
            "At least two method prefixes must share the "
            "same metric suffix."
        )

    # --------------------------------------------------------
    # Show detected schema
    # --------------------------------------------------------

    print("\nDetected method comparisons:")

    for metric in sorted(metric_map):

        print(
            f"\n  Metric: {metric}"
        )

        for method, column in sorted(
            metric_map[metric].items()
        ):

            print(
                f"    {method:<25} -> {column}"
            )

    print()

    # --------------------------------------------------------
    # Generate plots
    # --------------------------------------------------------

    if group_by is None:

        fig = create_plot(
            df=df,
            metric_map=metric_map,
            x_col=args.x_col,
            group_name=None,
            custom_title=args.title,
            max_cols=args.cols,
            width=args.width,
            height=args.height,
            show_labels=not args.no_labels,
            precision=args.precision,
            y_min=args.y_min,
            y_max=args.y_max,
        )

        saved_files = save_figure(
            fig=fig,
            output_dir=args.output_dir,
            stem="method_comparison",
            output_format=args.format,
            scale=args.scale,
        )

        for path in saved_files:
            print(f"Saved: {path}")

    else:

        grouped = df.groupby(
            group_by,
            sort=False,
            dropna=False,
        )

        for group_value, group_df in grouped:

            group_label = (
                f"{prettify(group_by)} {group_value}"
            )

            fig = create_plot(
                df=group_df.copy(),
                metric_map=metric_map,
                x_col=args.x_col,
                group_name=group_label,
                custom_title=args.title,
                max_cols=args.cols,
                width=args.width,
                height=args.height,
                show_labels=not args.no_labels,
                precision=args.precision,
                y_min=args.y_min,
                y_max=args.y_max,
            )

            stem = (
                f"{safe_filename(group_value)}"
                "_method_comparison"
            )

            saved_files = save_figure(
                fig=fig,
                output_dir=args.output_dir,
                stem=stem,
                output_format=args.format,
                scale=args.scale,
            )

            for path in saved_files:
                print(f"Saved: {path}")


if __name__ == "__main__":
    main()