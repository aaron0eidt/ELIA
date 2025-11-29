#!/usr/bin/env python3
"""
Run attribution-graph ablation experiments outside the Streamlit UI.

This script executes the same targeted/random/path perturbations as the
interactive tool and emits aggregate metrics so we can verify that the
visual plots actually reflect causal differences.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Any

# Ensure we can import the pipeline when this script is executed directly.
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from circuit_analysis.attribution_graphs_olmo import (  # noqa: E402
    AttributionGraphsPipeline,
    AttributionGraphConfig,
    ANALYSIS_PROMPTS,
)

RESULTS_DIR = SCRIPT_DIR / "results"
DEFAULT_OUTPUT = RESULTS_DIR / "offline_circuit_metrics.json"


def _load_prompts(args: argparse.Namespace) -> List[str]:
    if args.prompts_file:
        path = Path(args.prompts_file)
        if not path.exists():
            raise FileNotFoundError(f"Prompts file not found: {path}")
        with open(path, "r", encoding="utf-8") as f:
            prompts = [line.strip() for line in f if line.strip()]
        if not prompts:
            raise ValueError(f"No prompts found in {path}")
        return prompts

    if args.prompt_text:
        return [args.prompt_text]

    if args.prompt_index is not None:
        idx = args.prompt_index
        if not (0 <= idx < len(ANALYSIS_PROMPTS)):
            raise ValueError(f"--prompt-index must be between 0 and {len(ANALYSIS_PROMPTS)-1}")
        return [ANALYSIS_PROMPTS[idx]]

    if args.use_all:
        return ANALYSIS_PROMPTS

    # Default: run the canonical prompt set.
    return ANALYSIS_PROMPTS


def _format_summary(label: str, summary: Dict[str, Any]) -> str:
    return (
        f"{label:<20} "
        f"count={summary.get('count', 0):3d}  "
        f"avg|Δp|={summary.get('avg_abs_probability_change', 0.0):.4f}  "
        f"flip_rate={summary.get('flip_rate', 0.0):.2%}  "
        f"avg|Δlogit|={summary.get('avg_abs_logit_change', 0.0):.4f}"
    )


def _top_experiments(experiments: List[Dict[str, Any]], top_n: int = 5) -> List[Dict[str, Any]]:
    sorted_exps = sorted(
        experiments,
        key=lambda exp: abs(exp.get("probability_change", 0.0)),
        reverse=True,
    )
    summary = []
    for exp in sorted_exps[:top_n]:
        summary.append(
            {
                "label": exp.get("feature_name") or exp.get("path_description") or "feature_set",
                "probability_change": exp.get("probability_change", 0.0),
                "logit_change": exp.get("logit_change", 0.0),
                "flip": bool(exp.get("ablation_flips_top_prediction")),
            }
        )
    return summary


def _sanitize_analysis(analysis: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "prompt": analysis.get("prompt"),
        "summary_statistics": analysis.get("summary_statistics", {}),
        "counts": {
            "targeted": len(analysis.get("perturbation_experiments", []) or []),
            "random": len(analysis.get("random_baseline_experiments", []) or []),
            "path": len(analysis.get("path_ablation_experiments", []) or []),
            "random_path": len(analysis.get("random_path_baseline_experiments", []) or []),
        },
        "top_targeted": _top_experiments(analysis.get("perturbation_experiments", []) or []),
        "top_paths": _top_experiments(analysis.get("path_ablation_experiments", []) or []),
    }


def main():
    parser = argparse.ArgumentParser(description="Run offline attribution-graph ablation metrics.")
    parser.add_argument("--prompt-index", type=int, help="Run a single prompt by index from ANALYSIS_PROMPTS.")
    parser.add_argument("--prompt-text", type=str, help="Run a single custom prompt.")
    parser.add_argument("--prompts-file", type=str, help="Path to a text file with one prompt per line.")
    parser.add_argument("--use-all", action="store_true", help="Run all predefined analysis prompts.")
    parser.add_argument(
        "--feature-top-k",
        type=int,
        default=12,
        help="Number of top features per layer to analyze for targeted ablations.",
    )
    parser.add_argument(
        "--ablation-features-per-layer",
        type=int,
        default=4,
        help="Limit of targeted feature ablations per layer.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(DEFAULT_OUTPUT),
        help="Where to store the JSON summary of the offline metrics.",
    )
    args = parser.parse_args()

    prompts = _load_prompts(args)
    
    # Use consistent config matching the trained CLT
    # Use the constructed graph (pruned), not the full universe
    config = AttributionGraphConfig(
        n_features_per_layer=512,           # Match trained CLT
        sparsity_lambda=1e-3,              # Match training
        graph_feature_activation_threshold=0.01,
        graph_edge_weight_threshold=0.003,
        graph_max_features_per_layer=40,
        graph_max_edges_per_node=20,
        ablation_features_per_layer=args.ablation_features_per_layer,
        # Use default pruning_threshold (0.8) to use the constructed graph, not full universe
    )

    pipeline = AttributionGraphsPipeline(config)
    print(f"Running offline faithfulness experiments for {len(prompts)} prompt(s)...")
    batch_payload = pipeline.analyze_prompts_batch(prompts)

    aggregate_summary = batch_payload.get("aggregate_summary", {})
    print("\n=== Aggregate Metrics ===")
    print(_format_summary("Targeted", aggregate_summary.get("targeted", {})))
    print(_format_summary("Random baseline", aggregate_summary.get("random_baseline", {})))
    print(_format_summary("Path", aggregate_summary.get("path", {})))
    print(_format_summary("Random path", aggregate_summary.get("random_path_baseline", {})))
    print(
        f"\nTargeted − Random |Δp| = "
        f"{aggregate_summary.get('target_minus_random_abs_probability_change', 0.0):.4f}"
    )
    print(
        f"Path − Random path |Δp| = "
        f"{aggregate_summary.get('path_minus_random_abs_probability_change', 0.0):.4f}"
    )
    print(
        f"Targeted − Random flip rate = "
        f"{aggregate_summary.get('target_flip_rate_minus_random', 0.0):.4f}"
    )
    print(
        f"Path − Random path flip rate = "
        f"{aggregate_summary.get('path_flip_rate_minus_random', 0.0):.4f}"
    )

    sanitized_per_prompt = {
        key: _sanitize_analysis(analysis) for key, analysis in batch_payload.get("analyses", {}).items()
    }

    output_payload = {
        "prompts_ran": prompts,
        "aggregate_summary": aggregate_summary,
        "per_prompt": sanitized_per_prompt,
        "config": config.__dict__,
    }

    os.makedirs(Path(args.output).parent, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(output_payload, f, indent=2)

    print(f"\nSaved offline metrics to {args.output}")


if __name__ == "__main__":
    main()

