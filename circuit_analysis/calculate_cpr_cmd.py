
import numpy as np
import networkx as nx
import argparse
import json
import os
import sys
import logging
from typing import List, Tuple
from pathlib import Path
import math

# Ensure we can import the pipeline
sys.path.append(str(Path(__file__).resolve().parent.parent))

from circuit_analysis.attribution_graphs_olmo import (
    AttributionGraphsPipeline,
    AttributionGraphConfig,
    ANALYSIS_PROMPTS,
    AttributionGraph
)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def compute_cpr(k_values: List[float], f_values: List[float]) -> float:
    """
    Compute CPR (Integrated Circuit Performance Ratio) using the trapezoidal rule.
    CPR = Integral of f(C_k) dk
    """
    cpr = 0.0
    for i in range(len(k_values) - 1):
        cpr += 0.5 * (f_values[i] + f_values[i+1]) * (k_values[i+1] - k_values[i])
    return cpr

def compute_cmd(k_values: List[float], f_values: List[float]) -> float:
    """
    Compute CMD (Integrated Circuit-Model Distance) using the trapezoidal rule.
    CMD = Integral of |1 - f(C_k)| dk
    """
    cmd = 0.0
    for i in range(len(k_values) - 1):
        y0 = abs(1.0 - f_values[i])
        y1 = abs(1.0 - f_values[i+1])
        cmd += 0.5 * (y0 + y1) * (k_values[i+1] - k_values[i])
    return cmd

def get_active_features_from_graph(graph: nx.DiGraph) -> List[Tuple[int, int]]:
    """
    Extracts the list of feature nodes (as layer_idx, feature_idx tuples) from the graph.
    """
    features = []
    for node in graph.nodes():
        if node.startswith("feat_"):
            parts = node.split('_')
            try:
                # Format: feat_L{layer}_T{token}_F{feature}
                layer_idx = int(parts[1][1:])
                feature_idx = int(parts[3][1:])
                # We only care about unique (layer, feature) pairs for ablation
                features.append((layer_idx, feature_idx))
            except (IndexError, ValueError):
                continue
    return list(set(features))

def calculate_graph_importance(attribution_graph_obj: AttributionGraph, graph: nx.DiGraph) -> List[Tuple[str, float]]:
    """
    Calculates the importance of each feature node in the graph based on edge weights.
    Returns a list of (node_id, importance_score) sorted by importance descending.
    """
    node_importance = {}
    
    # Identify feature nodes
    feature_nodes = [n for n in graph.nodes() if attribution_graph_obj.node_types.get(n) == "feature"]
    
    # Calculate importance as sum of absolute weights of connected edges
    for node in feature_nodes:
        importance = 0.0
        # Outgoing edges
        for _, target in graph.out_edges(node):
            weight = attribution_graph_obj.edge_weights.get((node, target), 0.0)
            importance += abs(weight)
        # Incoming edges? MIB usually focuses on "importance" for the task.
        # Using sum of absolute edge weights is a standard proxy.
        # attribution_graphs_olmo.py prune_graph uses sum of all connected edge weights (in and out).
        for source, _ in graph.in_edges(node):
            weight = attribution_graph_obj.edge_weights.get((source, node), 0.0)
            importance += abs(weight)
            
        node_importance[node] = importance
        
    return sorted(node_importance.items(), key=lambda x: x[1], reverse=True)

def get_edges_count(graph: nx.DiGraph, nodes: List[str]) -> int:
    """
    Returns the number of edges in the subgraph induced by the given nodes 
    (plus edges to output/embedding if we consider them part of the circuit context).
    However, strictly following "fraction of total edges":
    We should count edges where BOTH source and target are in the kept set (including embeddings/output).
    """
    # Assuming embeddings and output are always "kept" or don't count towards the quota 
    # if we only ablate features.
    # But for the metric k = |C|/|N|, we need a consistent definition.
    # Let's define |C| as the number of edges in the subgraph induced by (Selected Features + Embeddings + Output).
    
    nodes_set = set(nodes)
    count = 0
    for u, v in graph.edges():
        if u in nodes_set and v in nodes_set:
            count += 1
    return count

def run_cpr_cmd_analysis(pipeline: AttributionGraphsPipeline, prompt_idx: int):
    """
    Compute CPR and CMD for a given prompt, using:
    
    - Universe: all feature nodes present in the attribution graph
    - Metric m: logit(target) only (no foil)
    - Interventions: ablation of feature sets with intervention_strength=1.0
    """
    prompt = ANALYSIS_PROMPTS[prompt_idx]
    logger.info(f"Analyzing prompt {prompt_idx}: '{prompt}'")
    
    # Build/prune the attribution graph for this prompt
    pipeline.analyze_prompt(prompt)
    full_graph = pipeline.attribution_graph.graph
    
    # Baseline: run once to get logits & feature activations
    baseline_data = pipeline.perturbation_experiments._prepare_inputs(prompt, top_k=1)
    target_token_id = baseline_data['baseline_top_tokens'].indices[0].item()
    baseline_logits = baseline_data['baseline_last_token_logits']
    m_N = baseline_logits[target_token_id].item()
    
    logger.info(
        f"Baseline m(N) = {m_N:.4f} "
        f"(Token: {pipeline.tokenizer.decode([target_token_id])})"
    )
    
    # Universe: all feature nodes in the graph
    universe_features = get_active_features_from_graph(full_graph)
    logger.info(f"Graph Universe size: {len(universe_features)} features")
    
    if not universe_features:
        logger.warning("No features found in graph. Skipping.")
        return None
    
    # Empty circuit: ablate all universe features
    empty_res = pipeline.perturbation_experiments.feature_set_ablation_experiment(
        prompt,
        feature_set=universe_features,
        intervention_strength=1.0,
        target_token_id=target_token_id
    )
    m_empty = empty_res["ablated_logit"]
    logger.info(f"Empty m(Ø) = {m_empty:.4f}")

    if not math.isfinite(m_empty):
        logger.warning(
            f"m_empty is non-finite ({m_empty}) for prompt {prompt_idx}; "
            "skipping CPR/CMD for this prompt."
        )
        return None
    
    # Node importance within the graph
    sorted_nodes = calculate_graph_importance(pipeline.attribution_graph, full_graph)
    
    total_edges = full_graph.number_of_edges()
    k_grid = [0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0]
    f_values = []
    actual_k_values = []
    
    # Embeddings/output are always kept
    always_kept_nodes = [n for n in full_graph.nodes() if not n.startswith("feat_")]
    
    logger.info("Computing faithfulness curve...")
    
    for k in k_grid:
        target_edge_count = int(k * total_edges)
        
        current_circuit_nodes = list(always_kept_nodes)
        current_feature_tuples = []
        
        for node, _ in sorted_nodes:
            current_edge_count = get_edges_count(full_graph, current_circuit_nodes)
            if current_edge_count >= target_edge_count and len(current_feature_tuples) > 0:
                break
            
            current_circuit_nodes.append(node)
            parts = node.split("_")
            l = int(parts[1][1:])
            f = int(parts[3][1:])
            current_feature_tuples.append((l, f))
        
        actual_edges = get_edges_count(full_graph, current_circuit_nodes)
        actual_k = actual_edges / total_edges if total_edges > 0 else 0.0
        actual_k_values.append(actual_k)
        
        # Complement = universe \ current features
        current_set = set(current_feature_tuples)
        complement_set = [ft for ft in universe_features if ft not in current_set]
        
        if not complement_set:
            m_Ck = m_N
        else:
            res = pipeline.perturbation_experiments.feature_set_ablation_experiment(
                prompt,
                feature_set=complement_set,
                intervention_strength=1.0,
                target_token_id=target_token_id
            )
            m_Ck = res["ablated_logit"]
        
        if not math.isfinite(m_Ck):
            logger.warning(
                f"Non-finite m_Ck={m_Ck} for k={k:.4f} on prompt {prompt_idx}; "
                "skipping this k point."
            )
            continue

        if abs(m_N - m_empty) < 1e-6:
            f_k = 0.0
        else:
            raw_f = (m_Ck - m_empty) / (m_N - m_empty)
            f_k = max(0.0, min(1.0, raw_f))
        
        f_values.append(f_k)
    
    if not actual_k_values or not f_values:
        logger.warning(f"No valid k-points for prompt {prompt_idx}; skipping.")
        return None
    
    pairs = sorted(zip(actual_k_values, f_values), key=lambda x: x[0])
    sorted_k = [p[0] for p in pairs]
    sorted_f = [p[1] for p in pairs]
    
    if sorted_k[0] > 0.0:
        sorted_k.insert(0, 0.0)
        sorted_f.insert(0, 0.0)
    if sorted_k[-1] < 1.0:
        last_f = sorted_f[-1]
        sorted_k.append(1.0)
        sorted_f.append(last_f)
        
    cpr = compute_cpr(sorted_k, sorted_f)
    cmd = compute_cmd(sorted_k, sorted_f)
    
    logger.info(f"Result: CPR={cpr:.4f}, CMD={cmd:.4f}")
    
    return {
        "prompt": prompt,
        "target_token": pipeline.tokenizer.decode([target_token_id]),
        "m_N": m_N,
        "m_empty": m_empty,
        "curve_k": sorted_k,
        "curve_f": sorted_f,
        "CPR": cpr,
        "CMD": cmd
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=str, default="circuit_analysis/results/cpr_cmd_results.json")
    args = parser.parse_args()
    
    # Initialize Pipeline
    config = AttributionGraphConfig(
        model_path="models/OLMo-2-1124-7B", # Adjust relative path if needed
        n_features_per_layer=512,  # Back to 512 due to memory constraints
        # We want a fairly rich graph to start with, so we can prune it down
        graph_feature_activation_threshold=0.01,
        graph_edge_weight_threshold=0.003, # Lower threshold for more edges (prev: 0.005)
        graph_max_features_per_layer=40, # Increased from 24 (prev: 100 was too slow)
        graph_max_edges_per_node=20,     # Increased from 12 (prev: 50 was too slow)
        # intervention_strength defaults to 5.0 in AttributionGraphConfig, which was working better
        intervention_strength=1.0,
    )
    
    # Check model path
    if not os.path.exists(config.model_path):
        # Try absolute python3 circuit_analysis/calculate_cpr_cmd.pypath or relative to script
        root_path = Path(__file__).resolve().parent.parent
        possible_path = root_path / "models" / "OLMo-2-1124-7B"
        if possible_path.exists():
            config.model_path = str(possible_path)
        else:
            # Try the one in current dir?
            pass
            
    pipeline = AttributionGraphsPipeline(config)
    
    # Load CLT
    clt_path = "circuit_analysis/models/clt_model.pth"
    if not os.path.exists(clt_path):
        # Try full path
        clt_path = str(Path(__file__).resolve().parent / "models" / "clt_model.pth")
        
    if os.path.exists(clt_path):
        pipeline.load_clt(clt_path)
    else:
        logger.error(f"CLT model not found at {clt_path}. Please train it first.")
        return

    results = []
    for i in range(len(ANALYSIS_PROMPTS)):
        try:
            res = run_cpr_cmd_analysis(pipeline, i)
            if res:
                results.append(res)
        except Exception as e:
            logger.error(f"Failed prompt {i}: {e}", exc_info=True)
            
    # Average CPR/CMD
    if results:
        avg_cpr = np.mean([r['CPR'] for r in results])
        avg_cmd = np.mean([r['CMD'] for r in results])
    else:
        avg_cpr = 0.0
        avg_cmd = 0.0
    
    final_output = {
        "results": results,
        "average_CPR": avg_cpr,
        "average_CMD": avg_cmd
    }
    
    # Save
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, 'w') as f:
        json.dump(final_output, f, indent=2)
        
    print(f"\n\nFinal Average CPR: {avg_cpr:.4f}")
    print(f"Final Average CMD: {avg_cmd:.4f}")
    print(f"Results saved to {args.output}")

if __name__ == "__main__":
    main()

