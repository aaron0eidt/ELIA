import os
import re
from typing import List, Optional

import matplotlib

matplotlib.use("Agg", force=True)
from matplotlib.patches import Patch
from matplotlib.ticker import MultipleLocator
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def _parse_score(score_str: str) -> Optional[tuple[int, int]]:
    score_str = score_str.strip()
    if not score_str:
        return None

    match = re.search(r"(\d+)\s*of\s*(\d+)", score_str)
    if not match:
        return None

    obtained, total = int(match.group(1)), int(match.group(2))
    if total == 0:
        return None

    return obtained, total


def load_and_preprocess_faithfulness_data(filepath: str = "user_study/Faithfulness.csv") -> pd.DataFrame:
    """Parses the custom-formatted faithfulness CSV into a tidy dataframe."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"The data file was not found at {filepath}")

    raw = pd.read_csv(filepath, header=None).fillna("")

    records: List[dict] = []
    current_analysis: Optional[str] = None
    current_prompt: Optional[str] = None

    for _, row in raw.iterrows():
        cells = [str(cell).strip() for cell in row.tolist()]
        # Early continue if the row is fully empty.
        if not any(cells):
            continue

        primary = cells[0]
        metric = cells[1] if len(cells) > 1 else ""
        extra_cells = cells[2:]

        # Section header row ("Attribution Analysis, Faithfulness Score")
        if primary and metric.lower().startswith("faithfulness score"):
            current_analysis = primary
            current_prompt = None
            continue

        # Skip rows without an active analysis section.
        if current_analysis is None:
            continue

        # Rows that introduce a new prompt.
        if primary:
            current_prompt = primary.strip('"')

        # Skip rows that don't have a prompt in context or a metric label.
        if not current_prompt or not metric:
            continue

        # Collect all valid score cells.
        for raw_cell in extra_cells:
            if not raw_cell:
                continue

            # Handle sub-metric labels such as "L0-L10: 4 of 4".
            sub_label = None
            score_part = raw_cell

            if ":" in raw_cell:
                left, right = raw_cell.split(":", 1)
                if _parse_score(right):
                    sub_label = left.strip()
                    score_part = right.strip()
                else:
                    # Skip notes or malformed values.
                    continue

            parsed_score = _parse_score(score_part)
            if not parsed_score:
                continue

            obtained, total = parsed_score
            metric_name = metric
            if sub_label:
                metric_name = f"{metric} ({sub_label})"

            records.append(
                {
                    "analysis": current_analysis,
                    "prompt": current_prompt,
                    "metric": metric_name,
                    "obtained": obtained,
                    "total": total,
                    "ratio": obtained / total,
                }
            )

    df = pd.DataFrame(records)
    if df.empty:
        raise ValueError("No data could be parsed from the faithfulness CSV.")

    df["metric_base"] = df["metric"].str.replace(r"\s*\(.*\)$", "", regex=True)

    return df


def _configure_plot_style() -> None:
    sns.set_theme(style="ticks", palette="viridis")
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = "Arial"
    plt.rcParams["axes.labelweight"] = "normal"
    plt.rcParams["axes.titleweight"] = "bold"
    plt.rcParams["figure.titleweight"] = "bold"
    plt.rcParams["savefig.dpi"] = 300
    plt.rcParams["figure.facecolor"] = "white"
    plt.rcParams["axes.facecolor"] = "white"
    plt.rcParams["grid.alpha"] = 0.2
    plt.rcParams["axes.spines.top"] = False
    plt.rcParams["axes.spines.right"] = False


def plot_average_faithfulness_by_metric(
    df: pd.DataFrame,
    output_dir: str = "writing/ELIA__EACL_2026_System_Demonstrations_/figures",
) -> None:
    os.makedirs(output_dir, exist_ok=True)
    _configure_plot_style()

    aggregates = (
        df.groupby(["analysis", "metric_base"])
        .agg(mean_ratio=("ratio", "mean"))
        .reset_index()
    )

    analyses = aggregates["analysis"].unique().tolist()
    desired_order = ["Attribution Analysis", "Function Vectors", "Circuit Tracing"]
    analyses = [a for a in desired_order if a in analyses] + [
        a for a in analyses if a not in desired_order
    ]
    if not analyses:
        return

    metric_order = (
        aggregates[["metric_base"]]
        .drop_duplicates()
        .sort_values("metric_base")
        .squeeze()
        .tolist()
    )
    desired_metric_order = [
        "Occlusion",
        "Saliency",
        "Integrated Gradients",
        "Category Analysis",
        "Overall Placement",
        "Function Type Attribution",
        "Layer Evolution",
        "Circuit Overview",
        "Subnetwork Explorer",
        "Feature Explorer",
    ]
    metric_order = [m for m in desired_metric_order if m in metric_order] + [
        m for m in metric_order if m not in desired_metric_order
    ]
    palette = sns.color_palette("colorblind", n_colors=len(metric_order))
    color_map = dict(zip(metric_order, palette))

    fig_height = max(7, len(analyses) * 3.5)
    fig, axes = plt.subplots(len(analyses), 1, figsize=(10, fig_height), sharex=True)
    if len(analyses) == 1:
        axes = [axes]

    for ax, analysis in zip(axes, analyses):
        subset = aggregates[aggregates["analysis"] == analysis].copy()
        subset["metric_base"] = pd.Categorical(
            subset["metric_base"],
            categories=metric_order,
            ordered=True,
        )
        subset = subset.sort_values("metric_base").reset_index(drop=True)
        analysis_order = [m for m in metric_order if m in subset["metric_base"].unique()]
        sns.barplot(
            data=subset,
            x="mean_ratio",
            y="metric_base",
            hue="metric_base",
            palette=color_map,
            orient="h",
            dodge=False,
            legend=False,
            ax=ax,
            order=analysis_order,
        )

        if ax.legend_:
            ax.legend_.remove()

        ax.set_xlim(0, 1.04)
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.set_title(analysis, fontsize=18, pad=14)
        ax.xaxis.set_major_locator(MultipleLocator(0.1))
        ax.tick_params(axis="x", labelsize=14)
        ax.set_yticklabels([])
        ax.tick_params(axis="y", length=0)
        ax.grid(
            axis="x",
            linestyle="--",
            linewidth=0.7,
            alpha=0.35,
            color="#6b6b6b",
        )
        ax.set_axisbelow(True)

        for patch in ax.patches:
            width = patch.get_width()
            y_mid = patch.get_y() + patch.get_height() / 2
            label_x = min(width + 0.008, 1.038)
            ax.text(
                label_x,
                y_mid,
                f"{width:.2f}",
                va="center",
                ha="left",
                fontsize=14,
                color="#2c2c2c",
                fontweight="medium",
            )

    axes[-1].set_xlabel("Average Faithfulness Ratio", fontsize=16, labelpad=10)

    fig.tight_layout(rect=(0, 0.13, 1, 0.98))

    legend_groups = [
        [m for m in metric_order if m in ["Occlusion", "Saliency", "Integrated Gradients"]],
        [
            m
            for m in metric_order
            if m in ["Category Analysis", "Overall Placement", "Function Type Attribution", "Layer Evolution"]
        ],
        [
            m
            for m in metric_order
            if m.startswith("Circuit Overview")
            or m.startswith("Subnetwork Explorer")
            or m.startswith("Feature Explorer")
        ],
    ]

    legend_positions = [0.18, 0.5, 0.82]

    for group, x_pos in zip(legend_groups, legend_positions):
        handles = [Patch(color=color_map[m], label=m) for m in group if m in color_map]
        labels = [h.get_label() for h in handles]
        if not handles:
            continue
        fig.legend(
            handles,
            labels,
            loc="upper center",
            bbox_to_anchor=(x_pos, 0.14),
            frameon=False,
            fontsize=14,
            ncol=1,
            columnspacing=1.0,
            handlelength=1.2,
        )
    output_path = os.path.join(output_dir, "faithfulness_average_overview.png")
    fig.savefig(output_path)
    plt.close(fig)


def plot_faithfulness_heatmaps(
    df: pd.DataFrame,
    output_dir: str = "writing/ELIA__EACL_2026_System_Demonstrations_/figures",
) -> None:
    os.makedirs(output_dir, exist_ok=True)
    _configure_plot_style()

    cmap = sns.color_palette("viridis", as_cmap=True)

    for analysis, data in df.groupby("analysis"):
        pivot = (
            data.pivot_table(
                index="prompt",
                columns="metric",
                values="ratio",
                aggfunc="mean",
            )
            .sort_index()
        )

        plt.figure(figsize=(max(8, pivot.shape[1] * 1.5), max(6, pivot.shape[0] * 0.6)))
        sns.heatmap(
            pivot,
            annot=True,
            fmt=".2f",
            cmap=cmap,
            vmin=0,
            vmax=1,
            cbar_kws={"label": "Faithfulness Ratio"},
        )
        plt.title(f"{analysis} Faithfulness Heatmap", fontsize=18)
        plt.xlabel("Metric", fontsize=14)
        plt.ylabel("Prompt", fontsize=14)
        plt.xticks(rotation=30, ha="right")
        plt.yticks(rotation=0)
        plt.tight_layout()
        filename = f"faithfulness_heatmap_{analysis.lower().replace(' ', '_')}.png"
        plt.savefig(os.path.join(output_dir, filename))
        plt.close()


def plot_faithfulness_distribution(
    df: pd.DataFrame,
    output_dir: str = "writing/ELIA__EACL_2026_System_Demonstrations_/figures",
) -> None:
    os.makedirs(output_dir, exist_ok=True)
    _configure_plot_style()

    plt.figure(figsize=(10, 6))
    sns.boxplot(
        data=df,
        x="analysis",
        y="ratio",
        hue="metric_base",
        palette="colorblind",
        fliersize=0,
    )
    plt.ylim(0, 1)
    plt.ylabel("Faithfulness Ratio", fontsize=16)
    plt.xlabel("Analysis", fontsize=16)
    plt.title("Faithfulness Distribution by Analysis and Metric", fontsize=18)
    plt.xticks(rotation=15)
    plt.legend(title="Metric", fontsize=12)
    plt.yticks([i / 10 for i in range(0, 11)])
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "faithfulness_distribution.png"))
    plt.close()


if __name__ == "__main__":
    try:
        data = load_and_preprocess_faithfulness_data("user_study/Faithfulness.csv")
        plot_average_faithfulness_by_metric(data)
        plot_faithfulness_heatmaps(data)
        plot_faithfulness_distribution(data)
        print("Faithfulness plots generated successfully.")
    except Exception as exc:
        print(f"An error occurred while generating faithfulness plots: {exc}")


