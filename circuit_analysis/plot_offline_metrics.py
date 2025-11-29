#!/usr/bin/env python3
"""
Visualize the aggregate metrics produced by offline_circuit_metrics.py
both overall and per prompt.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Any
from textwrap import fill

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

DEFAULT_RESULTS = Path(__file__).parent / "results" / "offline_circuit_metrics.json"
DEFAULT_CPR_CMD = Path(__file__).parent / "results" / "cpr_cmd_results.json"
# Save directly to the paper figures directory
DEFAULT_FIG = Path(__file__).parent.parent / "writing" / "ELIA__EACL_2026_System_Demonstrations_" / "figures" / "offline_circuit_metrics_combined.png"


def _load_payload(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if "aggregate_summary" not in data or "per_prompt" not in data:
        raise ValueError(f"Expected 'aggregate_summary' and 'per_prompt' in {path}")
    return data

def _configure_plot_style() -> None:
    sns.set_theme(style="ticks", palette="colorblind")
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


def _load_cpr_cmd(path: Path) -> Dict[str, Any]:
    """Load CPR/CMD results if available."""
    if not path.exists():
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data
    except Exception as e:
        print(f"Warning: Could not load CPR/CMD results from {path}: {e}")
        return None

def plot_combined(summary: Dict[str, Any], per_prompt: Dict[str, Any], output_path: Path, cpr_cmd_data: Dict[str, Any] = None):
    _configure_plot_style()
    
    # Prepare data
    labels = [r"$\mathbf{Aggregate}$"]
    targeted_vals = [summary["targeted"]["avg_abs_probability_change"]]
    random_vals = [summary["random_baseline"]["avg_abs_probability_change"]]
    path_vals = [summary["path"]["avg_abs_probability_change"]]
    random_path_vals = [summary["random_path_baseline"]["avg_abs_probability_change"]]
    
    # Prepare CPR data if available
    cpr_vals = []
    if cpr_cmd_data:
        # Get average CPR for aggregate
        results = cpr_cmd_data.get("results", [])
        if results:
            avg_cpr = cpr_cmd_data.get("average_CPR", 0.0)
            cpr_vals.append(avg_cpr)
        
        # Map prompts to CPR values
        prompt_to_cpr = {}
        for result in results:
            prompt_text = result.get("prompt", "")
            prompt_to_cpr[prompt_text] = result.get("CPR", 0.0)

    for key, data in per_prompt.items():
        # Clean up prompt label for display (first 5 words or so)
        prompt_text = data.get("prompt", key)
        labels.append(prompt_text)
        
        stats = data.get("summary_statistics", {})
        targeted_vals.append(stats.get("targeted", {}).get("avg_abs_probability_change", 0.0))
        random_vals.append(stats.get("random_baseline", {}).get("avg_abs_probability_change", 0.0))
        path_vals.append(stats.get("path", {}).get("avg_abs_probability_change", 0.0))
        random_path_vals.append(stats.get("random_path_baseline", {}).get("avg_abs_probability_change", 0.0))
        
        # Add CPR for this prompt if available
        if cpr_cmd_data and prompt_text in prompt_to_cpr:
            cpr_vals.append(prompt_to_cpr[prompt_text])
        elif cpr_cmd_data:
            # If CPR data exists but this prompt isn't in it, add zero
            cpr_vals.append(0.0)

    x = np.arange(len(labels))
    width = 0.2

    # Use a aspect ratio that fits well in a paper (e.g. wide enough for column)
    fig, ax = plt.subplots(figsize=(10, 6), constrained_layout=True)
    
    # Create second y-axis for CPR if data is available
    ax2 = None
    if cpr_cmd_data and cpr_vals:
        ax2 = ax.twinx()

    # Color palette - using specific indices from colorblind to ensure contrast
    # 0: Blue, 1: Orange, 2: Green, 3: Red, 4: Purple, etc.
    palette = sns.color_palette("colorblind")
    c_target = palette[0]  # Blue
    c_random = palette[7]  # Grey-ish or distinct
    c_path = palette[2]    # Green
    c_path_rnd = palette[3] # Red

    # Plot bars
    features_targeted = ax.bar(x - width * 1.5, targeted_vals, width, label="Targeted Features", color=c_target)
    features_random = ax.bar(x - width/2, random_vals, width, label="Random Features", color=c_random, alpha=0.7)
    paths_targeted = ax.bar(x + width/2, path_vals, width, label="Traced Circuits", color=c_path)
    paths_random = ax.bar(x + width * 1.5, random_path_vals, width, label="Random Path Baseline", color=c_path_rnd, alpha=0.7)

    # Add value labels on top of bars (only if they are significant enough to not clutter)
    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            # Threshold logic: Only skip if truly tiny (effectively zero)
            if height > 0.01: 
                ax.annotate(
                    f"{height:.2f}",
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha="center",
                    va="bottom",
                    fontsize=14,
                    fontweight="normal",
                    color="black"
                )

    autolabel(features_targeted)
    autolabel(features_random)
    autolabel(paths_targeted)
    autolabel(paths_random)
    
    # Plot CPR on second axis if available
    if ax2 and cpr_vals:
        # Plot as line with markers
        line1 = ax2.plot(x, cpr_vals, marker='o', linestyle='--', linewidth=2, 
                        markersize=8, color='purple', label='CPR', zorder=5)
        ax2.set_ylabel("CPR", fontsize=16, fontweight="normal", color='black')
        ax2.tick_params(axis='y', labelcolor='black', labelsize=14)
        ax2.set_ylim(0, 1.1)  # CPR is in [0,1]
        
        # Add value labels for CPR (below the markers)
        for i, cpr_val in enumerate(cpr_vals):
            if cpr_val > 0.01:
                ax2.annotate(f'{cpr_val:.2f}', xy=(i, cpr_val), xytext=(-20, -5),
                           textcoords='offset points', fontsize=11, color='purple',
                           fontweight='bold', ha='center')
        
        # Add CPR to legend
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, loc="upper left", ncol=3, 
                 frameon=True, framealpha=0.9, edgecolor="white", fontsize=12)
    else:
        # Original legend if no CPR
        ax.legend(loc="upper left", ncol=2, frameon=True, framealpha=0.9, edgecolor="white", fontsize=14)

    ax.set_ylabel("Avg. |Probability Change| (|Δp|)", fontsize=16, fontweight="normal")
    ax.set_xticks(x)
    
    # Wrap labels nicely (but preserve LaTeX formatting for Aggregate)
    wrapped_labels = []
    for label in labels:
        if r"$\mathbf{Aggregate}$" in label:
            wrapped_labels.append(label)
        else:
            wrapped_labels.append(fill(label, 20))
    ax.set_xticklabels(wrapped_labels, rotation=0, ha="center", fontsize=14)
    
    # Add subtle grid
    ax.grid(axis='y', linestyle='--', alpha=0.3)
    
    # Adjust y-limit to give some headroom for labels
    y_max = max(max(targeted_vals), max(path_vals), max(random_vals), max(random_path_vals))
    ax.set_ylim(0, y_max * 1.30)
    
    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Plot offline attribution metrics.")
    parser.add_argument(
        "--input",
        type=str,
        default=str(DEFAULT_RESULTS),
        help="Path to offline_circuit_metrics.json"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(DEFAULT_FIG),
        help="Path to save the per-prompt figure (PNG)."
    )
    parser.add_argument(
        "--cpr-cmd",
        type=str,
        default=str(DEFAULT_CPR_CMD),
        help="Path to CPR/CMD results JSON file (optional)."
    )
    args = parser.parse_args()

    if not Path(args.input).exists():
        print(f"Error: Input file {args.input} not found. Please run offline_circuit_metrics.py first.")
        return

    payload = _load_payload(Path(args.input))
    summary = payload["aggregate_summary"]
    per_prompt = payload["per_prompt"]
    
    # Load CPR/CMD data if available
    cpr_cmd_data = _load_cpr_cmd(Path(args.cpr_cmd))

    plot_combined(summary, per_prompt, Path(args.output), cpr_cmd_data)
    print(f"Saved combined plot to {args.output}")


if __name__ == "__main__":
    main()
