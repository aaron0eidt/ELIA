#!/usr/bin/env python3
# This script generates attribution graphs for the OLMo2 7B model.

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Any, Set
import json
import logging
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM
from collections import defaultdict
import networkx as nx
from dataclasses import dataclass
from tqdm import tqdm
import pickle
import requests
import time
import random
import copy
import os
import argparse

# --- Add this block to fix the import path ---
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))
# ---------------------------------------------

from utilities.utils import init_qwen_api, set_seed

# --- Constants ---
RESULTS_DIR = "circuit_analysis/results"
CLT_SAVE_PATH = "circuit_analysis/models/clt_model.pth"

# Configure logging.
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Set the device for training.
if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
    logger.info("Using MPS (Metal Performance Shaders) for GPU acceleration")
elif torch.cuda.is_available():
    DEVICE = torch.device("cuda")
    logger.info("Using CUDA for GPU acceleration")
else:
    DEVICE = torch.device("cpu")
    logger.info("Using CPU")

@dataclass
class AttributionGraphConfig:
    # Configuration for building the attribution graph.
    model_path: str = "./models/OLMo-2-1124-7B"
    max_seq_length: int = 512 
    n_features_per_layer: int = 512   # Number of features in each CLT layer
    sparsity_lambda: float = 1e-3     # Updated for L1 sparsity
    reconstruction_loss_weight: float = 1.0
    batch_size: int = 8
    learning_rate: float = 1e-4
    training_steps: int = 1000
    device: str = str(DEVICE)
    pruning_threshold: float = 0.8  # For graph pruning
    intervention_strength: float = 5.0  # For perturbation experiments
    qwen_api_config: Optional[Dict[str, str]] = None
    max_ablation_experiments: Optional[int] = None
    ablation_top_k_tokens: int = 5
    ablation_features_per_layer: Optional[int] = 2
    summary_max_layers: Optional[int] = None
    summary_features_per_layer: Optional[int] = 2
    random_baseline_trials: int = 5
    random_baseline_features: int = 1
    random_baseline_seed: int = 1234
    path_ablation_top_k: int = 3
    random_path_baseline_trials: int = 5
    graph_max_features_per_layer: int = 40
    graph_feature_activation_threshold: float = 0.01
    graph_edge_weight_threshold: float = 0.0
    graph_max_edges_per_node: int = 12

class JumpReLU(nn.Module):
    # The JumpReLU activation function.
    
    def __init__(self, threshold: float = 0.0):
        super().__init__()
        self.threshold = threshold
    
    def forward(self, x):
        return F.relu(x - self.threshold)

class CrossLayerTranscoder(nn.Module):
    # The Cross-Layer Transcoder (CLT) model.
    
    def __init__(self, model_config: Dict, clt_config: AttributionGraphConfig):
        super().__init__()
        self.config = clt_config
        self.model_config = model_config
        self.n_layers = model_config['num_hidden_layers']
        self.hidden_size = model_config['hidden_size']
        self.n_features = clt_config.n_features_per_layer
        
        # Encoder weights for each layer.
        self.encoders = nn.ModuleList([
            nn.Linear(self.hidden_size, self.n_features, bias=False)
            for _ in range(self.n_layers)
        ])
        
        # Decoder weights for cross-layer connections.
        self.decoders = nn.ModuleDict()
        for source_layer in range(self.n_layers):
            for target_layer in range(source_layer, self.n_layers):
                key = f"{source_layer}_to_{target_layer}"
                self.decoders[key] = nn.Linear(self.n_features, self.hidden_size, bias=False)
        
        # The activation function.
        self.activation = JumpReLU(threshold=0.0)
        
        # Initialize the weights.
        self._init_weights()
    
    def _init_weights(self):
        # Initializes the weights with small random values.
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.01)
    
    def encode(self, layer_idx: int, residual_activations: torch.Tensor) -> torch.Tensor:
        # Encodes residual stream activations to feature activations.
        return self.activation(self.encoders[layer_idx](residual_activations))
    
    def decode(self, source_layer: int, target_layer: int, feature_activations: torch.Tensor) -> torch.Tensor:
        # Decodes feature activations to the MLP output space.
        key = f"{source_layer}_to_{target_layer}"
        return self.decoders[key](feature_activations)
    
    def forward(self, residual_activations: List[torch.Tensor]) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        # The forward pass of the CLT.
        feature_activations = []
        reconstructed_mlp_outputs = []
        
        # Encode features for each layer.
        for layer_idx, residual in enumerate(residual_activations):
            features = self.encode(layer_idx, residual)
            feature_activations.append(features)
        
        # Reconstruct MLP outputs with cross-layer connections.
        for target_layer in range(self.n_layers):
            reconstruction = torch.zeros_like(residual_activations[target_layer])
            
            # Sum contributions from all previous layers.
            for source_layer in range(target_layer + 1):
                decoded = self.decode(source_layer, target_layer, feature_activations[source_layer])
                reconstruction += decoded
            
            reconstructed_mlp_outputs.append(reconstruction)
        
        return feature_activations, reconstructed_mlp_outputs

class FeatureVisualizer:
    # A class to visualize and interpret individual features.
    
    def __init__(self, tokenizer, cache_dir: Optional[Path] = None):
        self.tokenizer = tokenizer
        self.feature_interpretations: Dict[str, str] = {}
        self.cache_dir = cache_dir
        if self.cache_dir is not None:
            self.cache_dir = Path(self.cache_dir)
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            self._load_cache()
    
    def _cache_file(self) -> Optional[Path]:
        if self.cache_dir is None:
            return None
        return self.cache_dir / "feature_interpretations.json"
    
    def _load_cache(self):
        cache_file = self._cache_file()
        if cache_file is None or not cache_file.exists():
            return
        try:
            with open(cache_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                if isinstance(data, dict):
                    self.feature_interpretations.update({str(k): str(v) for k, v in data.items()})
        except Exception as e:
            logger.warning(f"Failed to load feature interpretation cache: {e}")
    
    def _save_cache(self):
        cache_file = self._cache_file()
        if cache_file is None:
            return
        try:
            with open(cache_file, 'w', encoding='utf-8') as f:
                json.dump(self.feature_interpretations, f, indent=2)
        except Exception as e:
            logger.warning(f"Failed to save feature interpretation cache: {e}")
    
    def visualize_feature(self, feature_idx: int, layer_idx: int, 
                         activations: torch.Tensor, input_tokens: List[str],
                         top_k: int = 10) -> Dict:
        # Creates a visualization for a single feature.
        feature_acts = activations[:, feature_idx].detach().cpu().numpy()
        
        # Find the top activating positions.
        top_positions = np.argsort(feature_acts)[-top_k:][::-1]
        
        visualization = {
            'feature_idx': feature_idx,
            'layer_idx': layer_idx,
            'max_activation': float(feature_acts.max()),
            'mean_activation': float(feature_acts.mean()),
            'sparsity': float((feature_acts > 0.1).mean()),
            'top_activations': []
        }
        
        for pos in top_positions:
            if pos < len(input_tokens):
                visualization['top_activations'].append({
                    'token': input_tokens[pos],
                    'position': int(pos),
                    'activation': float(feature_acts[pos])
                })
        
        return visualization
    
    def interpret_feature(self, feature_idx: int, layer_idx: int,
                          visualization_data: Dict,
                          qwen_api_config: Optional[Dict[str, str]] = None) -> str:
        # Interprets a feature based on its top activating tokens.
        top_tokens = [item['token'] for item in visualization_data['top_activations']]
        
        cache_key = f"L{layer_idx}_F{feature_idx}"
        
        if cache_key in self.feature_interpretations:
            return self.feature_interpretations[cache_key]
        
        # Use the Qwen API if it is configured.
        if qwen_api_config and qwen_api_config.get('api_key'):
            feature_name = cache_key
            interpretation = get_feature_interpretation_with_qwen(
                qwen_api_config, top_tokens, feature_name, layer_idx
            )
        else:
            # Use a simple heuristic as a fallback.
            if len(set(top_tokens)) == 1 and top_tokens:
                interpretation = f"Specific token: '{top_tokens[0]}'"
            elif top_tokens and all(token.isalpha() for token in top_tokens):
                interpretation = "Word/alphabetic tokens"
            elif top_tokens and all(token.isdigit() for token in top_tokens):
                interpretation = "Numeric tokens"
            elif top_tokens and all(token in '.,!?;:' for token in top_tokens):
                interpretation = "Punctuation"
            else:
                interpretation = "Mixed/polysemantic feature"
        
        self.feature_interpretations[cache_key] = interpretation
        self._save_cache()
        return interpretation

class AttributionGraph:
    # A class to construct and analyze attribution graphs.
    
    def __init__(self, clt: CrossLayerTranscoder, tokenizer, config: AttributionGraphConfig):
        self.clt = clt
        self.tokenizer = tokenizer
        self.config = config
        self.graph = nx.DiGraph()
        self.node_types = {}  # Track node types (feature, embedding, error, output)
        self.edge_weights = {}
        self.feature_metadata: Dict[str, Dict[str, Any]] = {}
        
    def compute_virtual_weights(self, source_layer: int, target_layer: int,
                               source_feature: int, target_feature: int) -> float:
        # Computes the virtual weight between two features.
        if target_layer <= source_layer:
            return 0.0
        
        # Get the encoder and decoder weights.
        encoder_weight = self.clt.encoders[target_layer].weight[target_feature]  # [hidden_size]
        
        total_weight = 0.0
        for intermediate_layer in range(source_layer, target_layer):
            decoder_key = f"{source_layer}_to_{intermediate_layer}"
            if decoder_key in self.clt.decoders:
                decoder_weight = self.clt.decoders[decoder_key].weight[:, source_feature]  # [hidden_size]
                # The virtual weight is inner product
                virtual_weight = torch.dot(decoder_weight, encoder_weight).item()
                total_weight += virtual_weight
        
        return total_weight
    
    def construct_graph(self, input_tokens: List[str], 
                       feature_activations: List[torch.Tensor],
                       target_token_idx: int = -1) -> nx.DiGraph:
        # Constructs the attribution graph for a prompt.
        self.graph.clear()
        self.node_types.clear()
        self.edge_weights.clear()
        
        seq_len = len(input_tokens)
        n_layers = len(feature_activations)
        
        # Add embedding nodes for the input tokens.
        for i, token in enumerate(input_tokens):
            node_id = f"emb_{i}_{token}"
            self.graph.add_node(node_id)
            self.node_types[node_id] = "embedding"
        
        # Add nodes for the features.
        active_features = {}  # Track which features are significantly active
        max_features_per_layer = self.config.graph_max_features_per_layer or 20  # Limit features per layer to prevent explosion
        activation_threshold = self.config.graph_feature_activation_threshold
        edge_weight_threshold = self.config.graph_edge_weight_threshold
        max_edges_per_node_cfg = self.config.graph_max_edges_per_node or 5
        
        for layer_idx, features in enumerate(feature_activations):
            # features shape: [batch_size, seq_len, n_features]
            batch_size, seq_len_layer, n_features = features.shape
            
            # Get the top activating features for this layer.
            layer_activations = features[0].mean(dim=0)  # Average across sequence
            top_features = torch.topk(layer_activations, 
                                    k=min(max_features_per_layer, n_features)).indices
            
            for token_pos in range(min(seq_len, seq_len_layer)):
                for feat_idx in top_features:
                    activation = features[0, token_pos, feat_idx.item()].item()
                    if activation > activation_threshold:
                        node_id = f"feat_L{layer_idx}_T{token_pos}_F{feat_idx.item()}"
                        self.graph.add_node(node_id)
                        self.node_types[node_id] = "feature"
                        active_features[node_id] = {
                            'layer': layer_idx,
                            'token_pos': token_pos,
                            'feature_idx': feat_idx.item(),
                            'activation': activation
                        }
                        self.feature_metadata[node_id] = {
                            'layer': layer_idx,
                            'token_position': token_pos,
                            'feature_index': feat_idx.item(),
                            'activation': activation,
                            'input_token': input_tokens[token_pos] if token_pos < len(input_tokens) else None
                        }
        
        # Add an output node for the target token.
        output_node = f"output_{target_token_idx}"
        self.graph.add_node(output_node)
        self.node_types[output_node] = "output"
        
        # Add edges based on virtual weights and activations.
        feature_nodes = [node for node, type_ in self.node_types.items() if type_ == "feature"]
        print(f"  Building attribution graph: {len(feature_nodes)} feature nodes, {len(self.graph.nodes())} total nodes")
        
        # Limit the number of edges to compute.
        max_edges_per_node = max(max_edges_per_node_cfg, 1)  # Limit connections per node
        
        for i, source_node in enumerate(feature_nodes):
            if i % 50 == 0:  # Progress indicator
                print(f"  Processing node {i+1}/{len(feature_nodes)}")
                
            edges_added = 0
            source_info = active_features[source_node]
            source_activation = source_info['activation']
            
            # Add edges to other features.
            for target_node in feature_nodes:
                if source_node == target_node or edges_added >= max_edges_per_node:
                    continue
                    
                target_info = active_features[target_node]
                
                # Only add edges that go forward in the network.
                if (target_info['layer'] > source_info['layer'] or 
                    (target_info['layer'] == source_info['layer'] and 
                     target_info['token_pos'] > source_info['token_pos'])):
                    
                    virtual_weight = self.compute_virtual_weights(
                        source_info['layer'], target_info['layer'],
                        source_info['feature_idx'], target_info['feature_idx']
                    )
                    
                    if abs(virtual_weight) > edge_weight_threshold:
                        edge_weight = source_activation * virtual_weight
                        self.graph.add_edge(source_node, target_node, weight=edge_weight)
                        self.edge_weights[(source_node, target_node)] = edge_weight
                        edges_added += 1
            
            # Add edges to the output node.
            layer_position = source_info['layer']
            # Allow contributions from all layers, with smaller weights for early layers.
            layer_scale = 0.1 if layer_position >= n_layers - 2 else max(0.05, 0.1 * (layer_position + 1) / n_layers)
            output_weight = source_activation * layer_scale
            if abs(output_weight) > 0:
                self.graph.add_edge(source_node, output_node, weight=output_weight)
                self.edge_weights[(source_node, output_node)] = output_weight
        
        # Add edges from embeddings to early features.
        for emb_node in [node for node, type_ in self.node_types.items() if type_ == "embedding"]:
            token_idx = int(emb_node.split('_')[1])
            for feat_node in feature_nodes:
                feat_info = active_features[feat_node]
                if feat_info['layer'] == 0 and feat_info['token_pos'] == token_idx:
                    # Direct connection from an embedding to a first-layer feature.
                    weight = feat_info['activation'] * 0.5  # Simplified
                    self.graph.add_edge(emb_node, feat_node, weight=weight)
                    self.edge_weights[(emb_node, feat_node)] = weight
        
        return self.graph
    
    def prune_graph(self, threshold: float = 0.8) -> nx.DiGraph:
        # Prunes the graph to keep only the most important nodes.
        # Calculate node importance based on edge weights.
        node_importance = defaultdict(float)
        
        for (source, target), weight in self.edge_weights.items():
            node_importance[source] += abs(weight)
            node_importance[target] += abs(weight)
        
        # Keep the top nodes by importance.
        sorted_nodes = sorted(node_importance.items(), key=lambda x: x[1], reverse=True)
        n_keep = int(len(sorted_nodes) * threshold)
        important_nodes = set([node for node, _ in sorted_nodes[:n_keep]])
        
        # Always keep the output and embedding nodes.
        for node, type_ in self.node_types.items():
            if type_ in ["output", "embedding"]:
                important_nodes.add(node)
        
        # Create the pruned graph.
        pruned_graph = self.graph.subgraph(important_nodes).copy()
        
        return pruned_graph
    
    def visualize_graph(self, graph: nx.DiGraph = None, save_path: str = None):
        # Visualizes the attribution graph.
        if graph is None:
            graph = self.graph
        
        plt.figure(figsize=(12, 8))
        
        # Create a layout for the graph.
        pos = nx.spring_layout(graph, k=1, iterations=50)
        
        # Color the nodes by type.
        node_colors = []
        for node in graph.nodes():
            node_type = self.node_types.get(node, "unknown")
            if node_type == "embedding":
                node_colors.append('lightblue')
            elif node_type == "feature":
                node_colors.append('lightgreen')
            elif node_type == "output":
                node_colors.append('orange')
            else:
                node_colors.append('gray')
        
        # Draw the nodes.
        nx.draw_networkx_nodes(graph, pos, node_color=node_colors, 
                              node_size=300, alpha=0.8)
        
        # Draw the edges with thickness based on weight.
        edges = graph.edges()
        edge_weights = [abs(self.edge_weights.get((u, v), 0.1)) for u, v in edges]
        max_weight = max(edge_weights) if edge_weights else 1
        edge_widths = [w / max_weight * 3 for w in edge_weights]
        
        nx.draw_networkx_edges(graph, pos, width=edge_widths, alpha=0.6,
                              edge_color='gray', arrows=True)
        
        # Draw the labels.
        nx.draw_networkx_labels(graph, pos, font_size=8)
        
        plt.title("Attribution Graph")
        plt.axis('off')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

class PerturbationExperiments:
    # Conducts perturbation experiments to validate hypotheses.
    
    def __init__(self, model, clt: CrossLayerTranscoder, tokenizer):
        self.model = model
        self.clt = clt
        self.tokenizer = tokenizer
        self._transformer_blocks: Optional[List[nn.Module]] = None
    
    def _get_transformer_blocks(self) -> List[nn.Module]:
        if self._transformer_blocks is not None:
            return self._transformer_blocks
        
        n_layers = getattr(self.model.config, "num_hidden_layers", None)
        if n_layers is None:
            raise ValueError("Model config does not expose num_hidden_layers; cannot resolve transformer blocks.")
        
        candidate_lists: List[Tuple[str, nn.ModuleList]] = []
        for name, module in self.model.named_modules():
            if isinstance(module, nn.ModuleList) and len(module) == n_layers:
                candidate_lists.append((name, module))
        
        if not candidate_lists:
            raise ValueError("Unable to locate transformer block ModuleList in model.")
        
        # Prefer names that look like transformer blocks.
        def _score(name: str) -> Tuple[int, str]:
            preferred_suffixes = ("layers", "blocks", "h")
            for idx, suffix in enumerate(preferred_suffixes):
                if name.endswith(suffix):
                    return (idx, name)
            return (len(preferred_suffixes), name)
        
        selected_name, selected_list = sorted(candidate_lists, key=lambda item: _score(item[0]))[0]
        self._transformer_blocks = list(selected_list)
        logger.debug(f"Resolved transformer blocks from ModuleList '{selected_name}'.")
        return self._transformer_blocks
    
    def _format_top_tokens(self, top_tokens: torch.return_types.topk) -> List[Tuple[str, float]]:
        return [
            (self.tokenizer.decode([idx]), prob.item())
            for idx, prob in zip(top_tokens.indices, top_tokens.values)
        ]
    
    def _prepare_inputs(self, input_text: str, top_k: int) -> Dict[str, Any]:
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
        
        device = next(self.model.parameters()).device
        inputs = self.tokenizer(
            input_text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512
        )
        if inputs["input_ids"].size(0) != 1:
            raise ValueError("Perturbation experiments currently support only batch size 1.")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        with torch.no_grad():
            baseline_outputs = self.model(**inputs, output_hidden_states=True, return_dict=True)
        
        baseline_logits = baseline_outputs.logits[0]
        target_position = baseline_logits.size(0) - 1
        baseline_last_token_logits = baseline_logits[target_position]
        baseline_probs = F.softmax(baseline_last_token_logits, dim=-1)
        baseline_top_tokens = torch.topk(baseline_probs, k=top_k)
        
        hidden_states: List[torch.Tensor] = list(baseline_outputs.hidden_states[1:])
        with torch.no_grad():
            feature_activations, _ = self.clt(hidden_states)
        
        return {
            'inputs': inputs,
            'baseline_outputs': baseline_outputs,
            'baseline_logits': baseline_logits,
            'baseline_last_token_logits': baseline_last_token_logits,
            'baseline_probs': baseline_probs,
            'baseline_top_tokens': baseline_top_tokens,
            'target_position': target_position,
            'hidden_states': hidden_states,
            'feature_activations': feature_activations,
            'default_target_token_id': baseline_top_tokens.indices[0].item()
        }
    
    def _compute_feature_contributions(
        self,
        feature_activations: List[torch.Tensor],
        feature_set: List[Tuple[int, int]]
    ) -> Dict[int, torch.Tensor]:
        contributions: Dict[int, torch.Tensor] = {}
        with torch.no_grad():
            for layer_idx, feature_idx in feature_set:
                if layer_idx >= len(feature_activations):
                    continue
                features = feature_activations[layer_idx]
                if feature_idx >= features.size(-1):
                    continue
                feature_values = features[:, :, feature_idx].detach()
                
                for dest_layer in range(layer_idx, self.clt.n_layers):
                    decoder_key = f"{layer_idx}_to_{dest_layer}"
                    if decoder_key not in self.clt.decoders:
                        continue
                    decoder = self.clt.decoders[decoder_key]
                    weight_column = decoder.weight[:, feature_idx]
                    contrib = torch.einsum('bs,h->bsh', feature_values, weight_column).detach()
                    if dest_layer in contributions:
                        contributions[dest_layer] += contrib
                    else:
                        contributions[dest_layer] = contrib
        return contributions
    
    def _run_with_hooks(
        self,
        inputs: Dict[str, torch.Tensor],
        contributions: Dict[int, torch.Tensor],
        intervention_strength: float
    ):
        blocks = self._get_transformer_blocks()
        handles: List[Any] = []
        
        def _make_hook(cached_contrib: torch.Tensor):
            def hook(module, module_input, module_output):
                if isinstance(module_output, torch.Tensor):
                    target_tensor = module_output
                elif isinstance(module_output, (tuple, list)):
                    target_tensor = module_output[0]
                elif hasattr(module_output, "last_hidden_state"):
                    target_tensor = module_output.last_hidden_state
                else:
                    raise TypeError(
                        f"Unsupported module output type '{type(module_output)}' for perturbation hook."
                    )
                
                tensor_contrib = cached_contrib.to(target_tensor.device).to(target_tensor.dtype)
                scaled = intervention_strength * tensor_contrib
                
                if isinstance(module_output, torch.Tensor):
                    return module_output - scaled
                elif isinstance(module_output, tuple):
                    modified = module_output[0] - scaled
                    return (modified,) + tuple(module_output[1:])
                elif isinstance(module_output, list):
                    modified = [module_output[0] - scaled, *module_output[1:]]
                    return modified
                else:
                    module_output.last_hidden_state = module_output.last_hidden_state - scaled
                    return module_output
            return hook
        
        try:
            for dest_layer, contrib in contributions.items():
                if dest_layer >= len(blocks):
                    continue
                handles.append(blocks[dest_layer].register_forward_hook(_make_hook(contrib)))
            
            with torch.no_grad():
                outputs = self.model(**inputs, output_hidden_states=True, return_dict=True)
        finally:
            for handle in handles:
                handle.remove()
        
        return outputs
    
    def feature_set_ablation_experiment(
        self,
        input_text: str,
        feature_set: List[Tuple[int, int]],
        intervention_strength: float = 5.0,
        target_token_id: Optional[int] = None,
        top_k: int = 5,
        ablation_label: str = "feature_set",
        extra_metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        try:
            baseline_data = self._prepare_inputs(input_text, top_k)
            if target_token_id is None:
                target_token_id = baseline_data['default_target_token_id']
            
            feature_set_normalized = [
                (int(layer_idx), int(feature_idx)) for layer_idx, feature_idx in feature_set
            ]
            contributions = self._compute_feature_contributions(
                baseline_data['feature_activations'],
                feature_set_normalized
            )
            
            baseline_probs = baseline_data['baseline_probs']
            baseline_top_tokens = baseline_data['baseline_top_tokens']
            baseline_last_token_logits = baseline_data['baseline_last_token_logits']
            target_position = baseline_data['target_position']
            hidden_states = baseline_data['hidden_states']
            
            baseline_prob = baseline_probs[target_token_id].item()
            baseline_logit = baseline_last_token_logits[target_token_id].item()
            baseline_summary = {
                'baseline_top_tokens': self._format_top_tokens(baseline_top_tokens),
                'baseline_probability': baseline_prob,
                'baseline_logit': baseline_logit
            }
            
            if not contributions:
                result = {
                    **baseline_summary,
                    'ablated_top_tokens': baseline_summary['baseline_top_tokens'],
                    'ablated_probability': baseline_prob,
                    'ablated_logit': baseline_logit,
                    'probability_change': 0.0,
                    'logit_change': 0.0,
                    'kl_divergence': 0.0,
                    'entropy_change': 0.0,
                    'hidden_state_delta_norm': 0.0,
                    'hidden_state_relative_change': 0.0,
                    'ablation_flips_top_prediction': False,
                    'feature_set': [
                        {'layer': layer_idx, 'feature': feature_idx}
                        for layer_idx, feature_idx in feature_set_normalized
                    ],
                    'feature_set_size': len(feature_set_normalized),
                    'intervention_strength': intervention_strength,
                    'target_token_id': target_token_id,
                    'target_token': self.tokenizer.decode([target_token_id]),
                    'contributing_layers': [],
                    'ablation_applied': False,
                    'ablation_type': ablation_label,
                    'warning': 'no_contributions_found'
                }
                if extra_metadata:
                    result.update(extra_metadata)
                return result
            
            ablated_outputs = self._run_with_hooks(
                baseline_data['inputs'],
                contributions,
                intervention_strength
            )
            
            ablated_logits = ablated_outputs.logits[0, target_position]
            ablated_probs = F.softmax(ablated_logits, dim=-1)
            ablated_top_tokens = torch.topk(ablated_probs, k=top_k)
            
            ablated_prob = ablated_probs[target_token_id].item()
            ablated_logit = ablated_logits[target_token_id].item()
            
            epsilon = 1e-9
            kl_divergence = torch.sum(
                baseline_probs * (torch.log(baseline_probs + epsilon) - torch.log(ablated_probs + epsilon))
            ).item()
            if not np.isfinite(kl_divergence):
                kl_divergence = 0.0
                
            entropy_baseline = -(baseline_probs * torch.log(baseline_probs + epsilon)).sum().item()
            entropy_ablated = -(ablated_probs * torch.log(ablated_probs + epsilon)).sum().item()
            entropy_change = entropy_ablated - entropy_baseline
            if not np.isfinite(entropy_change):
                entropy_change = 0.0
            
            baseline_hidden = hidden_states[-1][:, target_position, :]
            ablated_hidden = ablated_outputs.hidden_states[-1][:, target_position, :]
            hidden_delta_norm = torch.norm(baseline_hidden - ablated_hidden, dim=-1).item()
            hidden_baseline_norm = torch.norm(baseline_hidden, dim=-1).item()
            hidden_relative_change = hidden_delta_norm / (hidden_baseline_norm + 1e-9)
            
            result = {
                **baseline_summary,
                'ablated_top_tokens': self._format_top_tokens(ablated_top_tokens),
                'ablated_probability': ablated_prob,
                'ablated_logit': ablated_logit,
                'probability_change': baseline_prob - ablated_prob,
                'logit_change': baseline_logit - ablated_logit,
                'kl_divergence': kl_divergence,
                'entropy_change': entropy_change,
                'hidden_state_delta_norm': hidden_delta_norm,
                'hidden_state_relative_change': hidden_relative_change,
                'ablation_flips_top_prediction': bool(
                    baseline_top_tokens.indices[0].item() != ablated_top_tokens.indices[0].item()
                ),
                'feature_set': [
                    {'layer': layer_idx, 'feature': feature_idx}
                    for layer_idx, feature_idx in feature_set_normalized
                ],
                'feature_set_size': len(feature_set_normalized),
                'intervention_strength': intervention_strength,
                'target_token_id': target_token_id,
                'target_token': self.tokenizer.decode([target_token_id]),
                'contributing_layers': sorted(list(contributions.keys())),
                'ablation_applied': True,
                'ablation_type': ablation_label
            }
            if extra_metadata:
                result.update(extra_metadata)
            return result
        
        except Exception as e:
            logger.warning(f"Perturbation experiment failed: {e}")
            return {
                'baseline_top_tokens': [],
                'ablated_top_tokens': [],
                'feature_set': [
                    {'layer': layer_idx, 'feature': feature_idx}
                    for layer_idx, feature_idx in feature_set
                ],
                'feature_set_size': len(feature_set),
                'intervention_strength': intervention_strength,
                'probability_change': 0.0,
                'logit_change': 0.0,
                'kl_divergence': 0.0,
                'entropy_change': 0.0,
                'hidden_state_delta_norm': 0.0,
                'hidden_state_relative_change': 0.0,
                'ablation_flips_top_prediction': False,
                'ablation_applied': False,
                'ablation_type': ablation_label,
                'error': str(e)
            }
    
    def feature_ablation_experiment(
        self,
        input_text: str,
        target_layer: int,
        target_feature: int,
        intervention_strength: float = 5.0,
        target_token_id: Optional[int] = None,
        top_k: int = 5,
    ) -> Dict[str, Any]:
        return self.feature_set_ablation_experiment(
            input_text=input_text,
            feature_set=[(target_layer, target_feature)],
            intervention_strength=intervention_strength,
            target_token_id=target_token_id,
            top_k=top_k,
            ablation_label="targeted_feature"
        )
    
    def random_feature_ablation_experiment(
        self,
        input_text: str,
        num_features: int = 1,
        intervention_strength: float = 5.0,
        target_token_id: Optional[int] = None,
        top_k: int = 5,
        seed: Optional[int] = None
    ) -> Dict[str, Any]:
        rng = random.Random(seed)
        num_features = max(1, int(num_features))
        feature_set: List[Tuple[int, int]] = []
        for _ in range(num_features):
            layer_idx = rng.randrange(self.clt.n_layers)
            feature_idx = rng.randrange(self.clt.n_features)
            feature_set.append((layer_idx, feature_idx))
        
        result = self.feature_set_ablation_experiment(
            input_text=input_text,
            feature_set=feature_set,
            intervention_strength=intervention_strength,
            target_token_id=target_token_id,
            top_k=top_k,
            ablation_label="random_baseline",
            extra_metadata={'random_seed': seed}
        )
        return result

class AttributionGraphsPipeline:
    # The main pipeline for the attribution graph analysis.
    
    def __init__(self, config: AttributionGraphConfig):
        self.config = config
        self.device = torch.device(config.device)
        
        # Load the model and tokenizer.
        logger.info(f"Loading OLMo2 7B model from {config.model_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(config.model_path)
        
        # Configure model loading based on the device.
        if "mps" in config.device:
            # MPS supports float16 but not device_map.
            self.model = AutoModelForCausalLM.from_pretrained(
                config.model_path,
                torch_dtype=torch.float16,
                device_map=None
            ).to(self.device)
        elif "cuda" in config.device:
            self.model = AutoModelForCausalLM.from_pretrained(
                config.model_path,
                torch_dtype=torch.float16,
                device_map="auto"
            )
        else:
            # CPU
            self.model = AutoModelForCausalLM.from_pretrained(
                config.model_path,
                torch_dtype=torch.float32,
                device_map=None
            ).to(self.device)
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Initialize the CLT.
        model_config = self.model.config.to_dict()
        self.clt = CrossLayerTranscoder(model_config, config).to(self.device)
        
        # Initialize the other components.
        # cache_dir = Path(RESULTS_DIR) / "feature_interpretations_cache"
        # Disable persistent caching to ensure interpretations are prompt-specific and not reused from other contexts.
        self.feature_visualizer = FeatureVisualizer(self.tokenizer, cache_dir=None)
        self.attribution_graph = AttributionGraph(self.clt, self.tokenizer, config)
        self.perturbation_experiments = PerturbationExperiments(self.model, self.clt, self.tokenizer)
        
        logger.info("Attribution Graphs Pipeline initialized successfully")
    
    def train_clt(self, training_texts: List[str]) -> Dict:
        # Trains the Cross-Layer Transcoder.
        logger.info("Starting CLT training...")
        
        optimizer = torch.optim.Adam(self.clt.parameters(), lr=self.config.learning_rate)
        
        training_stats = {
            'reconstruction_losses': [],
            'sparsity_losses': [],
            'total_losses': []
        }
        
        for step in tqdm(range(self.config.training_steps), desc="Training CLT"):
            # Sample a batch of texts.
            batch_texts = np.random.choice(training_texts, size=self.config.batch_size)
            
            total_loss = 0.0
            total_recon_loss = 0.0
            total_sparsity_loss = 0.0
            
            for text in batch_texts:
                # Tokenize the text.
                inputs = self.tokenizer(text, return_tensors="pt", max_length=self.config.max_seq_length,
                                      truncation=True, padding=True).to(self.device)
                
                # Get the model activations.
                with torch.no_grad():
                    outputs = self.model(**inputs, output_hidden_states=True)
                    hidden_states = outputs.hidden_states[1:]
                
                # Forward pass through the CLT.
                feature_activations, reconstructed_outputs = self.clt(hidden_states)
                
                # Compute the reconstruction loss.
                recon_loss = 0.0
                for i, (target, pred) in enumerate(zip(hidden_states, reconstructed_outputs)):
                    recon_loss += F.mse_loss(pred, target)
                
                # Compute the sparsity loss.
                sparsity_loss = 0.0
                for features in feature_activations:
                    sparsity_loss += torch.mean(torch.tanh(self.config.sparsity_lambda * features))
                
                # Total loss.
                loss = (self.config.reconstruction_loss_weight * recon_loss + 
                       self.config.sparsity_lambda * sparsity_loss)
                
                total_loss += loss
                total_recon_loss += recon_loss
                total_sparsity_loss += sparsity_loss
            
            # Average the losses.
            total_loss /= self.config.batch_size
            total_recon_loss /= self.config.batch_size
            total_sparsity_loss /= self.config.batch_size
            
            # Backward pass.
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            
            # Log the progress.
            training_stats['total_losses'].append(total_loss.item())
            training_stats['reconstruction_losses'].append(total_recon_loss.item())
            training_stats['sparsity_losses'].append(total_sparsity_loss.item())
            
            if step % 100 == 0:
                logger.info(f"Step {step}: Total Loss = {total_loss.item():.4f}, "
                           f"Recon Loss = {total_recon_loss.item():.4f}, "
                           f"Sparsity Loss = {total_sparsity_loss.item():.4f}")
        
        logger.info("CLT training completed")
        return training_stats
    
    def analyze_prompt(self, prompt: str, target_token_idx: int = -1) -> Dict:
        # Performs a complete analysis for a single prompt.
        logger.info(f"Analyzing prompt: '{prompt[:50]}...'")
        
        # Tokenize the prompt.
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        input_tokens = self.tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
        
        # Get the model activations.
        with torch.no_grad():
            outputs = self.model(**inputs, output_hidden_states=True)
            hidden_states = outputs.hidden_states[1:]
        
        # Forward pass through the CLT.
        feature_activations, reconstructed_outputs = self.clt(hidden_states)
        
        logger.info("  > Starting feature visualization and interpretation...")
        feature_visualizations = {}
        for layer_idx, features in enumerate(feature_activations):
            logger.info(f"  - Processing Layer {layer_idx}...")
            layer_viz = {}
            # Analyze the top features for this layer.
            # features shape: [batch_size, seq_len, n_features]
            feature_importance = torch.mean(features, dim=(0, 1))  # Average over batch and sequence
            top_features = torch.topk(feature_importance, k=min(5, feature_importance.size(0))).indices
            
            for feat_idx in top_features:
                viz = self.feature_visualizer.visualize_feature(
                    feat_idx.item(), layer_idx, features[0], input_tokens
                )
                interpretation = self.feature_visualizer.interpret_feature(
                    feat_idx.item(), layer_idx, viz, self.config.qwen_api_config
                )
                viz['interpretation'] = interpretation
                layer_viz[f"feature_{feat_idx.item()}"] = viz
            
            feature_visualizations[f"layer_{layer_idx}"] = layer_viz
        
        # Construct the attribution graph.
        graph = self.attribution_graph.construct_graph(
            input_tokens, feature_activations, target_token_idx
        )
        
        # Prune the graph.
        pruned_graph = self.attribution_graph.prune_graph(self.config.pruning_threshold)
        
        # Analyze the most important paths.
        important_paths = []
        if len(pruned_graph.nodes()) > 0:
            # Find paths from embeddings to the output.
            embedding_nodes = [node for node, type_ in self.attribution_graph.node_types.items() 
                             if type_ == "embedding" and node in pruned_graph]
            output_nodes = [node for node, type_ in self.attribution_graph.node_types.items() 
                           if type_ == "output" and node in pruned_graph]
            
            for emb_node in embedding_nodes[:3]:  # Top 3 embedding nodes
                for out_node in output_nodes:
                    try:
                        paths = list(nx.all_simple_paths(pruned_graph, emb_node, out_node, cutoff=5))
                        for path in paths[:2]:  # Top 2 paths
                            path_weight = 1.0
                            for i in range(len(path) - 1):
                                edge_weight = self.attribution_graph.edge_weights.get(
                                    (path[i], path[i+1]), 0.0
                                )
                                path_weight *= abs(edge_weight)
                            
                            important_paths.append({
                                'path': path,
                                'weight': path_weight,
                                'description': self._describe_path(path)
                            })
                    except nx.NetworkXNoPath:
                        continue
        
        # Sort paths by importance.
        important_paths.sort(key=lambda x: x['weight'], reverse=True)
        
        # Run targeted perturbation experiments for highlighted features.
        targeted_feature_ablation_results: List[Dict[str, Any]] = []
        max_total_experiments = self.config.max_ablation_experiments
        per_layer_limit = self.config.ablation_features_per_layer
        total_run = 0
        stop_all = False
        for layer_name, layer_features in feature_visualizations.items():
            if stop_all:
                break
            try:
                layer_idx = int(layer_name.split('_')[1])
            except (IndexError, ValueError):
                logger.warning(f"Unable to parse layer index from key '{layer_name}'. Skipping perturbation experiments for this layer.")
                continue
            
            feature_items = list(layer_features.items())
            if per_layer_limit is not None:
                feature_items = feature_items[:per_layer_limit]
            
            for feature_name, feature_payload in feature_items:
                if max_total_experiments is not None and total_run >= max_total_experiments:
                    stop_all = True
                    break
                try:
                    feature_idx = int(feature_name.split('_')[1])
                except (IndexError, ValueError):
                    logger.warning(f"Unable to parse feature index from key '{feature_name}'. Skipping perturbation experiment.")
                    continue
                
                ablation = self.perturbation_experiments.feature_ablation_experiment(
                    prompt,
                    layer_idx,
                    feature_idx,
                    intervention_strength=self.config.intervention_strength,
                    target_token_id=None,
                    top_k=self.config.ablation_top_k_tokens,
                )
                ablation.update({
                    'layer_name': layer_name,
                    'feature_name': feature_name,
                    'feature_interpretation': feature_payload.get('interpretation'),
                    'feature_max_activation': feature_payload.get('max_activation'),
                })
                targeted_feature_ablation_results.append(ablation)
                total_run += 1
        
        # Random baseline perturbations for comparison.
        random_baseline_results: List[Dict[str, Any]] = []
        baseline_trials = self.config.random_baseline_trials
        if baseline_trials and baseline_trials > 0:
            num_features = self.config.random_baseline_features or 1
            for trial_idx in range(baseline_trials):
                seed = None
                if self.config.random_baseline_seed is not None:
                    seed = self.config.random_baseline_seed + trial_idx
                random_result = self.perturbation_experiments.random_feature_ablation_experiment(
                    prompt,
                    num_features=num_features,
                    intervention_strength=self.config.intervention_strength,
                    target_token_id=None,
                    top_k=self.config.ablation_top_k_tokens,
                    seed=seed
                )
                random_result['trial_index'] = trial_idx
                random_baseline_results.append(random_result)
        
        # Path-level ablations for the most important circuits.
        path_ablation_results: List[Dict[str, Any]] = []
        max_paths = self.config.path_ablation_top_k or 0
        extracted_paths: List[Dict[str, Any]] = []
        if max_paths > 0 and important_paths:
            for path_info in important_paths[:max_paths]:
                feature_set = self._extract_feature_set_from_path(path_info.get('path', []))
                if not feature_set:
                    continue
                path_result = self.perturbation_experiments.feature_set_ablation_experiment(
                    prompt,
                    feature_set=feature_set,
                    intervention_strength=self.config.intervention_strength,
                    target_token_id=None,
                    top_k=self.config.ablation_top_k_tokens,
                    ablation_label="path",
                    extra_metadata={
                        'path_nodes': path_info.get('path'),
                        'path_description': path_info.get('description'),
                        'path_weight': path_info.get('weight')
                    }
                )
                path_ablation_results.append(path_result)
                enriched_path_info = path_info.copy()
                enriched_path_info['feature_set'] = feature_set
                extracted_paths.append(enriched_path_info)
        
        random_path_baseline_results: List[Dict[str, Any]] = []
        path_baseline_trials = self.config.random_path_baseline_trials
        if path_baseline_trials and path_baseline_trials > 0 and extracted_paths:
            rng = random.Random(self.config.random_baseline_seed)
            available_nodes = [
                data for data in self.attribution_graph.node_types.items()
                if data[1] == "feature"
            ]
            for trial in range(path_baseline_trials):
                selected_path = extracted_paths[min(trial % len(extracted_paths), len(extracted_paths) - 1)]
                target_length = len(selected_path.get('feature_set', []))
                source_layers = [layer for layer, _ in selected_path.get('feature_set', [])]
                min_layer = min(source_layers) if source_layers else 0
                max_layer = max(source_layers) if source_layers else self.clt.n_layers - 1
                excluded_keys = {
                    (layer, feature)
                    for layer, feature in selected_path.get('feature_set', [])
                }
                random_feature_set: List[Tuple[int, int]] = []
                attempts = 0
                while len(random_feature_set) < target_length and attempts < target_length * 5:
                    attempts += 1
                    if not available_nodes:
                        break
                    node_name, node_type = rng.choice(available_nodes)
                    metadata = self.attribution_graph.feature_metadata.get(node_name)
                    if metadata is None:
                        continue
                    if metadata['layer'] < min_layer or metadata['layer'] > max_layer:
                        continue
                    key = (metadata['layer'], metadata['feature_index'])
                    if key in excluded_keys:
                        continue
                    if key not in random_feature_set:
                        random_feature_set.append(key)
                if not random_feature_set:
                    continue
                if len(random_feature_set) < max(1, target_length):
                    continue
                random_path_result = self.perturbation_experiments.feature_set_ablation_experiment(
                    prompt,
                    feature_set=random_feature_set,
                    intervention_strength=self.config.intervention_strength,
                    target_token_id=None,
                    top_k=self.config.ablation_top_k_tokens,
                    ablation_label="random_path_baseline",
                    extra_metadata={
                        'trial_index': trial,
                        'sampled_feature_set': random_feature_set,
                        'reference_path_weight': selected_path.get('weight')
                    }
                )
                random_path_baseline_results.append(random_path_result)
        
        targeted_summary = self._summarize_ablation_results(targeted_feature_ablation_results)
        random_summary = self._summarize_ablation_results(random_baseline_results)
        path_summary = self._summarize_ablation_results(path_ablation_results)
        random_path_summary = self._summarize_ablation_results(random_path_baseline_results)
        summary_statistics = {
            'targeted': targeted_summary,
            'random_baseline': random_summary,
            'path': path_summary,
            'random_path_baseline': random_path_summary,
            'target_minus_random_abs_probability_change': targeted_summary.get('avg_abs_probability_change', 0.0) - random_summary.get('avg_abs_probability_change', 0.0),
            'target_flip_rate_minus_random': targeted_summary.get('flip_rate', 0.0) - random_summary.get('flip_rate', 0.0),
            'path_minus_random_abs_probability_change': path_summary.get('avg_abs_probability_change', 0.0) - random_path_summary.get('avg_abs_probability_change', 0.0),
            'path_flip_rate_minus_random': path_summary.get('flip_rate', 0.0) - random_path_summary.get('flip_rate', 0.0)
        }
        
        results = {
            'prompt': prompt,
            'input_tokens': input_tokens,
            'feature_visualizations': feature_visualizations,
            'full_graph_stats': {
                'n_nodes': len(graph.nodes()),
                'n_edges': len(graph.edges()),
                'node_types': dict(self.attribution_graph.node_types)
            },
            'pruned_graph_stats': {
                'n_nodes': len(pruned_graph.nodes()),
                'n_edges': len(pruned_graph.edges())
            },
            'important_paths': important_paths[:5],  # Top 5 paths
            'graph': pruned_graph,
            'perturbation_experiments': targeted_feature_ablation_results,
            'random_baseline_experiments': random_baseline_results,
            'path_ablation_experiments': path_ablation_results,
            'random_path_baseline_experiments': random_path_baseline_results,
            'summary_statistics': summary_statistics
        }
        
        return results
    
    def _extract_feature_set_from_path(self, path: List[str]) -> List[Tuple[int, int]]:
        feature_set: List[Tuple[int, int]] = []
        seen: Set[Tuple[int, int]] = set()
        for node in path:
            if not isinstance(node, str):
                continue
            if not node.startswith("feat_"):
                continue
            parts = node.split('_')
            try:
                layer_str = parts[1]  # e.g., "L0"
                feature_str = parts[3]  # e.g., "F123"
                layer_idx = int(layer_str[1:])
                feature_idx = int(feature_str[1:])
            except (IndexError, ValueError):
                continue
            key = (layer_idx, feature_idx)
            if key not in seen:
                seen.add(key)
                feature_set.append(key)
        return feature_set
    
    def _summarize_ablation_results(self, experiments: List[Dict[str, Any]]) -> Dict[str, Any]:
        summary = {
            'count': len(experiments),
            'avg_probability_change': 0.0,
            'avg_abs_probability_change': 0.0,
            'std_probability_change': 0.0,
            'avg_logit_change': 0.0,
            'avg_abs_logit_change': 0.0,
            'std_logit_change': 0.0,
            'avg_kl_divergence': 0.0,
            'avg_entropy_change': 0.0,
            'avg_hidden_state_delta_norm': 0.0,
            'avg_hidden_state_relative_change': 0.0,
            'flip_rate': 0.0,
            'count_flipped': 0
        }
        if not experiments:
            return summary
        
        probability_changes = np.array([exp.get('probability_change', 0.0) for exp in experiments], dtype=float)
        logit_changes = np.array([exp.get('logit_change', 0.0) for exp in experiments], dtype=float)
        kl_divergences = np.array([exp.get('kl_divergence', 0.0) for exp in experiments], dtype=float)
        entropy_changes = np.array([exp.get('entropy_change', 0.0) for exp in experiments], dtype=float)
        hidden_norms = np.array([exp.get('hidden_state_delta_norm', 0.0) for exp in experiments], dtype=float)
        hidden_relative = np.array([exp.get('hidden_state_relative_change', 0.0) for exp in experiments], dtype=float)
        flip_flags = np.array([1.0 if exp.get('ablation_flips_top_prediction') else 0.0 for exp in experiments], dtype=float)
        
        # Helper to safely compute mean/std ignoring NaNs
        def safe_mean(arr):
            with np.errstate(all='ignore'):
                m = np.nanmean(arr)
                return float(m) if np.isfinite(m) else 0.0
            
        def safe_std(arr):
            with np.errstate(all='ignore'):
                s = np.nanstd(arr)
                return float(s) if np.isfinite(s) else 0.0

        summary.update({
            'avg_probability_change': safe_mean(probability_changes),
            'avg_abs_probability_change': safe_mean(np.abs(probability_changes)),
            'std_probability_change': safe_std(probability_changes),
            'avg_logit_change': safe_mean(logit_changes),
            'avg_abs_logit_change': safe_mean(np.abs(logit_changes)),
            'std_logit_change': safe_std(logit_changes),
            'avg_kl_divergence': safe_mean(kl_divergences),
            'avg_entropy_change': safe_mean(entropy_changes),
            'avg_hidden_state_delta_norm': safe_mean(hidden_norms),
            'avg_hidden_state_relative_change': safe_mean(hidden_relative),
            'flip_rate': safe_mean(flip_flags),
            'count_flipped': int(np.round(np.nansum(flip_flags)))
        })
        return summary
    
    def analyze_prompts_batch(self, prompts: List[str]) -> Dict[str, Any]:
        analyses: Dict[str, Dict[str, Any]] = {}
        aggregated_targeted: List[Dict[str, Any]] = []
        aggregated_random: List[Dict[str, Any]] = []
        aggregated_path: List[Dict[str, Any]] = []
        
        for idx, prompt in enumerate(prompts):
            logger.info(f"[Batch Eval] Processing prompt {idx + 1}/{len(prompts)}")
            analysis = self.analyze_prompt(prompt)
            key = f"prompt_{idx + 1}"
            analyses[key] = analysis
            aggregated_targeted.extend(analysis.get('perturbation_experiments', []))
            aggregated_random.extend(analysis.get('random_baseline_experiments', []))
            aggregated_path.extend(analysis.get('path_ablation_experiments', []))
        
        aggregate_summary = {
            'targeted': self._summarize_ablation_results(aggregated_targeted),
            'random_baseline': self._summarize_ablation_results(aggregated_random),
            'path': self._summarize_ablation_results(aggregated_path),
            'random_path_baseline': self._summarize_ablation_results(
                [
                    exp
                    for analysis in analyses.values()
                    for exp in analysis.get('random_path_baseline_experiments', [])
                ]
            )
        }
        aggregate_summary['target_minus_random_abs_probability_change'] = (
            aggregate_summary['targeted'].get('avg_abs_probability_change', 0.0)
            - aggregate_summary['random_baseline'].get('avg_abs_probability_change', 0.0)
        )
        aggregate_summary['target_flip_rate_minus_random'] = (
            aggregate_summary['targeted'].get('flip_rate', 0.0)
            - aggregate_summary['random_baseline'].get('flip_rate', 0.0)
        )
        aggregate_summary['path_minus_random_abs_probability_change'] = (
            aggregate_summary['path'].get('avg_abs_probability_change', 0.0)
            - aggregate_summary['random_path_baseline'].get('avg_abs_probability_change', 0.0)
        )
        aggregate_summary['path_flip_rate_minus_random'] = (
            aggregate_summary['path'].get('flip_rate', 0.0)
            - aggregate_summary['random_path_baseline'].get('flip_rate', 0.0)
        )
        
        return {
            'analyses': analyses,
            'aggregate_summary': aggregate_summary,
            'prompt_texts': prompts
        }
    
    def _describe_path(self, path: List[str]) -> str:
        # Generates a human-readable description of a path.
        descriptions = []
        for node in path:
            if self.attribution_graph.node_types[node] == "embedding":
                token = node.split('_')[2]
                descriptions.append(f"Token '{token}'")
            elif self.attribution_graph.node_types[node] == "feature":
                parts = node.split('_')
                layer = parts[1][1:]  # Remove 'L'
                feature = parts[3][1:]  # Remove 'F'
                # Try to get the interpretation.
                key = f"L{layer}_F{feature}"
                interpretation = self.feature_visualizer.feature_interpretations.get(key, "unknown")
                descriptions.append(f"Feature L{layer}F{feature} ({interpretation})")
            elif self.attribution_graph.node_types[node] == "output":
                descriptions.append("Output")
        
        return " → ".join(descriptions)
    
    def save_results(self, results: Dict, save_path: str):
        # Saves the analysis results to a file.
        serializable_results = copy.deepcopy(results)
        
        if 'graph' in serializable_results:
            serializable_results['graph'] = nx.node_link_data(serializable_results['graph'])
        
        analyses = serializable_results.get('analyses', {})
        for key, analysis in analyses.items():
            if 'graph' in analysis:
                analysis['graph'] = nx.node_link_data(analysis['graph'])
        
        with open(save_path, 'w') as f:
            json.dump(serializable_results, f, indent=2, default=str)
        
        logger.info(f"Results saved to {save_path}")

    def save_clt(self, path: str):
        # Saves the trained CLT model.
        torch.save(self.clt.state_dict(), path)
        logger.info(f"CLT model saved to {path}")

    def load_clt(self, path: str):
        # Loads a trained CLT model.
        self.clt.load_state_dict(torch.load(path, map_location=self.device))
        self.clt.to(self.device)
        self.clt.eval()  # Set the model to evaluation mode
        logger.info(f"Loaded CLT model from {path}")

# --- Configuration ---
MAX_SEQ_LEN = 256
N_FEATURES_PER_LAYER = 512
TRAINING_STEPS = 2500
BATCH_SIZE = 64
LEARNING_RATE = 1e-3

# Prompts for generating the final analysis.
ANALYSIS_PROMPTS = [
    "The capital of France is",
    "def factorial(n):",
    "The literary device in the phrase 'The wind whispered through the trees' is"
]

# A larger set of prompts for training.
TRAINING_PROMPTS = [
    "The capital of France is", "To be or not to be, that is the", "A stitch in time saves",
    "The first person to walk on the moon was", "The chemical formula for water is H2O.",
    "Translate to German: 'The cat sits on the mat.'", "def factorial(n):", "import numpy as np",
    "The main ingredients in a pizza are", "What is the powerhouse of the cell?",
    "The equation E=mc^2 relates energy to", "Continue the story: Once upon a time, there was a",
    "Classify the sentiment: 'I am overjoyed!'", "Extract the entities: 'Apple Inc. is in Cupertino.'",
    "What is the next number: 2, 4, 8, 16, __?", "A rolling stone gathers no",
    "The opposite of hot is", "import torch", "import pandas as pd", "class MyClass:",
    "def __init__(self):", "The primary colors are", "What is the capital of Japan?",
    "Who wrote 'Hamlet'?", "The square root of 64 is", "The sun rises in the",
    "The Pacific Ocean is the largest ocean on Earth.", "The mitochondria is the powerhouse of the cell.",
    "What is the capital of Mongolia?", "The movie 'The Matrix' can be classified into the following genre:",
    "The French translation of 'I would like to order a coffee, please.' is:",
    "The literary device in the phrase 'The wind whispered through the trees' is",
    "A Python function that calculates the factorial of a number is:",
    "The main ingredient in a Negroni cocktail is",
    "Summarize the plot of 'Hamlet' in one sentence:",
    "The sentence 'The cake was eaten by the dog' is in the following voice:",
    "A good headline for an article about a new breakthrough in battery technology would be:"
]


# --- Qwen API for Feature Interpretation ---
@torch.no_grad()
def get_feature_interpretation_with_qwen(
    api_config: dict, 
    top_tokens: list[str], 
    feature_name: str, 
    layer_index: int,
    max_retries: int = 3,
    initial_backoff: float = 2.0
) -> str:
    # Generates a high-quality interpretation for a feature using the Qwen API.
    if not api_config or not api_config.get('api_key'):
        logger.warning("Qwen API not configured. Skipping interpretation.")
        return "API not configured"

    headers = {
        "Authorization": f"Bearer {api_config['api_key']}",
        "Content-Type": "application/json"
    }
    
    # Create a specialized prompt.
    prompt_text = f"""
You are an expert in transformer interpretability. A feature in a language model (feature '{feature_name}' at layer {layer_index}) is most strongly activated by the following tokens:

{', '.join(f"'{token}'" for token in top_tokens)}

Based *only* on these tokens, what is the most likely function or role of this feature?
Your answer must be a short, concise phrase (e.g., "Detecting proper nouns", "Identifying JSON syntax", "Completing lists", "Recognizing negative sentiment"). Do not write a full sentence.
"""
    
    data = {
        "model": api_config["model"],
        "messages": [
            {
                "role": "user",
                "content": [{"type": "text", "text": prompt_text}]
            }
        ],
        "max_tokens": 50,
        "temperature": 0.1,
        "top_p": 0.9,
        "seed": 42
    }

    logger.info(f"  > Interpreting {feature_name} (Layer {layer_index})...")

    for attempt in range(max_retries):
        try:
            logger.info(f"    - Attempt {attempt + 1}/{max_retries}: Sending request to Qwen API...")
            response = requests.post(
                f"{api_config['api_endpoint']}/chat/completions",
                headers=headers,
                json=data,
                timeout=60
            )
            response.raise_for_status()  # Raise an exception for bad status codes (4xx or 5xx)
            
            result = response.json()
            interpretation = result["choices"][0]["message"]["content"].strip()
            
            # Remove quotes from the output.
            if interpretation.startswith('"') and interpretation.endswith('"'):
                interpretation = interpretation[1:-1]
            
            logger.info(f"    - Success! Interpretation: '{interpretation}'")
            return interpretation
            
        except requests.exceptions.RequestException as e:
            logger.warning(f"    - Qwen API request failed (Attempt {attempt + 1}/{max_retries}): {e}")
            if attempt < max_retries - 1:
                backoff_time = initial_backoff * (2 ** attempt)
                logger.info(f"    - Retrying in {backoff_time:.1f} seconds...")
                time.sleep(backoff_time)
            else:
                logger.error("    - Max retries reached. Failing.")
                return f"API Error: {e}"
        except (KeyError, IndexError) as e:
            logger.error(f"    - Failed to parse Qwen API response: {e}")
            return "API Error: Invalid response format"
        finally:
            # Add a delay to respect API rate limits.
            time.sleep(2.1)
            
    return "API Error: Max retries exceeded"


def train_transcoder(transcoder, model, tokenizer, training_prompts, device, steps=1000, batch_size=16, optimizer=None):
    # Trains the Cross-Layer Transcoder.
    transcoder.train()
    
    # Use a progress bar for visual feedback.
    progress_bar = tqdm(range(steps), desc="Training CLT")
    
    for step in progress_bar:
        # Get a random batch of prompts.
        batch_prompts = random.choices(training_prompts, k=batch_size)
        
        # Tokenize the batch.
        inputs = tokenizer(
            batch_prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=MAX_SEQ_LEN
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}

        # Get the model activations.
        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)
            hidden_states = outputs.hidden_states[1:]
        
        # Forward pass through the CLT.
        feature_activations, reconstructed_outputs = transcoder(hidden_states)
        
        # Compute the reconstruction loss.
        recon_loss = 0.0
        for i, (target, pred) in enumerate(zip(hidden_states, reconstructed_outputs)):
            recon_loss += F.mse_loss(pred, target)
        
        # Compute the sparsity loss.
        sparsity_loss = 0.0
        for features in feature_activations:
            sparsity_loss += torch.mean(torch.tanh(0.01 * features)) # Use config.sparsity_lambda
        
        # Total loss.
        loss = (0.8 * recon_loss + 0.2 * sparsity_loss) # Use config.reconstruction_loss_weight
        
        if optimizer:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        progress_bar.set_postfix({
            "Recon Loss": f"{recon_loss.item():.4f}",
            "Sparsity Loss": f"{sparsity_loss.item():.4f}",
            "Total Loss": f"{loss.item():.4f}"
        })

def generate_feature_visualizations(transcoder, model, tokenizer, prompt, device, qwen_api_config=None, graph_config: Optional[AttributionGraphConfig] = None):
    # Generates feature visualizations and interpretations for a prompt.
    # Tokenize the prompt.
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=MAX_SEQ_LEN
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # Get the model activations.
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
        hidden_states = outputs.hidden_states[1:]
    
    # Forward pass through the CLT.
    feature_activations, reconstructed_outputs = transcoder(hidden_states)

    # Visualize the features.
    feature_visualizations = {}
    for layer_idx, features in enumerate(feature_activations):
        layer_viz = {}
        # Analyze the top features for this layer.
        # features shape: [batch_size, seq_len, n_features]
        feature_importance = torch.mean(features, dim=(0, 1))  # Average over batch and sequence
        top_features = torch.topk(feature_importance, k=min(5, feature_importance.size(0))).indices
        
        for feat_idx in top_features:
            viz = FeatureVisualizer(tokenizer).visualize_feature(
                feat_idx.item(), layer_idx, features[0], tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
            )
            interpretation = FeatureVisualizer(tokenizer).interpret_feature(
                feat_idx.item(), layer_idx, viz, qwen_api_config
            )
            viz['interpretation'] = interpretation
            layer_viz[f"feature_{feat_idx.item()}"] = viz
        
        feature_visualizations[f"layer_{layer_idx}"] = layer_viz

    # Construct the attribution graph.
    if graph_config is None:
        graph_config = AttributionGraphConfig()
    attribution_graph = AttributionGraph(transcoder, tokenizer, graph_config)
    graph = attribution_graph.construct_graph(
        tokenizer.convert_ids_to_tokens(inputs["input_ids"][0]), feature_activations, -1 # No target token for visualization
    )

    # Prune the graph.
    pruned_graph = attribution_graph.prune_graph(0.8) # Use config.pruning_threshold

    # Analyze the most important paths.
    important_paths = []
    if len(pruned_graph.nodes()) > 0:
        # Find paths from embeddings to the output.
        embedding_nodes = [node for node, type_ in attribution_graph.node_types.items() 
                         if type_ == "embedding" and node in pruned_graph]
        output_nodes = [node for node, type_ in attribution_graph.node_types.items() 
                       if type_ == "output" and node in pruned_graph]
        
        for emb_node in embedding_nodes[:3]:  # Top 3 embedding nodes
            for out_node in output_nodes:
                try:
                    paths = list(nx.all_simple_paths(pruned_graph, emb_node, out_node, cutoff=5))
                    for path in paths[:2]:  # Top 2 paths
                        path_weight = 1.0
                        for i in range(len(path) - 1):
                            edge_weight = attribution_graph.edge_weights.get(
                                (path[i], path[i+1]), 0.0
                            )
                            path_weight *= abs(edge_weight)
                        
                        important_paths.append({
                            'path': path,
                            'weight': path_weight,
                            'description': attribution_graph._describe_path(path)
                        })
                except nx.NetworkXNoPath:
                    continue
    
    # Sort paths by importance.
    important_paths.sort(key=lambda x: x['weight'], reverse=True)

    return {
        "prompt": prompt,
        "full_graph_stats": {
            "n_nodes": len(graph.nodes()),
            "n_edges": len(graph.edges()),
            "node_types": dict(attribution_graph.node_types)
        },
        "pruned_graph_stats": {
            "n_nodes": len(pruned_graph.nodes()),
            "n_edges": len(pruned_graph.edges())
        },
        "feature_visualizations": feature_visualizations,
        "important_paths": important_paths[:5] # Top 5 paths
    }

def main():
    # Main function to run the analysis for a single prompt.
    
    # Set a seed for reproducibility.
    set_seed()

    # --- Argument Parser ---
    parser = argparse.ArgumentParser(description="Run Attribution Graph analysis for a single prompt.")
    parser.add_argument(
        '--prompt-index',
        type=int,
        required=True,
        help=f"The 0-based index of the prompt to analyze from the ANALYSIS_PROMPTS list (0 to {len(ANALYSIS_PROMPTS) - 1})."
    )
    parser.add_argument(
        '--force-retrain-clt',
        action='store_true',
        help="Force re-training of the Cross-Layer Transcoder, even if a saved model exists."
    )
    parser.add_argument(
        '--batch-eval',
        action='store_true',
        help="Analyze all predefined prompts and compute aggregate faithfulness metrics."
    )
    args = parser.parse_args()

    prompt_idx = args.prompt_index
    if not (0 <= prompt_idx < len(ANALYSIS_PROMPTS)):
        print(f"❌ Error: --prompt-index must be between 0 and {len(ANALYSIS_PROMPTS) - 1}.")
        return

    # Get the API config from the utility function.
    qwen_api_config = init_qwen_api()

    # Configuration - Use consistent settings matching trained CLT
    config = AttributionGraphConfig(
        model_path="./models/OLMo-2-1124-7B",
        n_features_per_layer=512,           # Match trained CLT
        training_steps=500,
        batch_size=4,
        max_seq_length=256,
        learning_rate=1e-4,
        sparsity_lambda=1e-3,                # Match training (L1 sparsity)
        graph_feature_activation_threshold=0.01,
        graph_edge_weight_threshold=0.003,
        graph_max_features_per_layer=40,
        graph_max_edges_per_node=20,
        qwen_api_config=qwen_api_config
    )
    
    print("Attribution Graphs for OLMo2 7B - Single Prompt Pipeline")
    print("=" * 50)
    print(f"Model path: {config.model_path}")
    print(f"Device: {config.device}")
    
    try:
        # Initialize the full pipeline.
        print("🚀 Initializing Attribution Graphs Pipeline...")
        pipeline = AttributionGraphsPipeline(config)
        print("✓ Pipeline initialized successfully")
        print()
        
        # Load an existing CLT model or train a new one.
        if os.path.exists(CLT_SAVE_PATH) and not args.force_retrain_clt:
            print(f"🧠 Loading existing CLT model from {CLT_SAVE_PATH}...")
            pipeline.load_clt(CLT_SAVE_PATH)
            print("✓ CLT model loaded successfully.")
        else:
            if args.force_retrain_clt and os.path.exists(CLT_SAVE_PATH):
                print("��‍♂️ --force-retrain-clt flag is set. Overwriting existing model.")
            
            # Train a new CLT model.
            print("📚 Training a new CLT model...")
            print(f"   Training on {len(TRAINING_PROMPTS)} example texts...")
            training_stats = pipeline.train_clt(TRAINING_PROMPTS)
            print("✓ CLT training completed.")

            # Save the training statistics.
            stats_save_path = os.path.join(RESULTS_DIR, "clt_training_stats.json")
            with open(stats_save_path, 'w') as f:
                json.dump(training_stats, f, indent=2)
            print(f"   Saved training stats to {stats_save_path}")
            
            # Save the new model.
            pipeline.save_clt(CLT_SAVE_PATH)
            print(f"   Saved trained model to {CLT_SAVE_PATH} for future use.")
        
        print()

        if args.batch_eval:
            print("📊 Running batch faithfulness evaluation across all prompts...")
            batch_payload = pipeline.analyze_prompts_batch(ANALYSIS_PROMPTS)
            final_results = copy.deepcopy(batch_payload)
            final_results['config'] = config.__dict__
            final_results['timestamp'] = str(time.time())
            for analysis_entry in final_results['analyses'].values():
                analysis_entry.pop('graph', None)
            batch_save_path = os.path.join(RESULTS_DIR, "attribution_graphs_batch_results.json")
            pipeline.save_results(final_results, batch_save_path)
            print(f"💾 Batch results saved to {batch_save_path}")
            
            aggregate_summary = batch_payload['aggregate_summary']
            targeted_summary = aggregate_summary.get('targeted', {})
            random_summary = aggregate_summary.get('random_baseline', {})
            path_summary = aggregate_summary.get('path', {})
            
            def _format_summary(label: str, summary: Dict[str, Any]) -> str:
                return (
                    f"{label}: count={summary.get('count', 0)}, "
                    f"avg|Δp|={summary.get('avg_abs_probability_change', 0.0):.4f}, "
                    f"flip_rate={summary.get('flip_rate', 0.0):.2%}"
                )
            
            print("📈 Aggregate faithfulness summary")
            print(f"    {_format_summary('Targeted', targeted_summary)}")
            print(f"    {_format_summary('Random baseline', random_summary)}")
            print(f"    {_format_summary('Path', path_summary)}")
            print(f"    {_format_summary('Random path baseline', aggregate_summary.get('random_path_baseline', {}))}")
            diff_abs = aggregate_summary.get('target_minus_random_abs_probability_change', 0.0)
            diff_flip = aggregate_summary.get('target_flip_rate_minus_random', 0.0)
            path_diff_abs = aggregate_summary.get('path_minus_random_abs_probability_change', 0.0)
            path_diff_flip = aggregate_summary.get('path_flip_rate_minus_random', 0.0)
            print(f"    Targeted vs Random |Δp| difference: {diff_abs:.4f}")
            print(f"    Targeted vs Random flip rate difference: {diff_flip:.4f}")
            print(f"    Path vs Random path |Δp| difference: {path_diff_abs:.4f}")
            print(f"    Path vs Random path flip rate difference: {path_diff_flip:.4f}")
            print("\n🎉 Batch evaluation completed successfully!")
            return

        # Analyze the selected prompt.
        prompt_to_analyze = ANALYSIS_PROMPTS[prompt_idx]
        print(f"🔍 Analyzing prompt {prompt_idx + 1}/{len(ANALYSIS_PROMPTS)}: '{prompt_to_analyze}'")
        
        analysis = pipeline.analyze_prompt(prompt_to_analyze, target_token_idx=-1)
        
        # Display the key results.
        print(f"  ✓ Tokenized into {len(analysis['input_tokens'])} tokens")
        print(f"  ✓ Full graph: {analysis['full_graph_stats']['n_nodes']} nodes, {analysis['full_graph_stats']['n_edges']} edges")
        print(f"  ✓ Pruned graph: {analysis['pruned_graph_stats']['n_nodes']} nodes, {analysis['pruned_graph_stats']['n_edges']} edges")
        
        # Show the top features.
        print("  📊 Top active features:")
        feature_layers_items = list(analysis['feature_visualizations'].items())
        if config.summary_max_layers is not None:
            feature_layers_items = feature_layers_items[:config.summary_max_layers]
        for layer_name, layer_features in feature_layers_items:
            print(f"    {layer_name}:")
            feature_items = layer_features.items()
            if config.summary_features_per_layer is not None:
                feature_items = list(feature_items)[:config.summary_features_per_layer]
            for feat_name, feat_data in feature_items:
                print(f"      {feat_name}: {feat_data['interpretation']} (max: {feat_data['max_activation']:.3f})")
        
        print()

        # Summarize perturbation experiments and baselines.
        print("🧪 Targeted feature ablations:")
        targeted_results = analysis.get('perturbation_experiments', [])
        if targeted_results:
            for experiment in targeted_results:
                layer_name = experiment.get('layer_name', f"L{experiment.get('feature_set', [{}])[0].get('layer', '?')}")
                feature_name = experiment.get('feature_name', f"F{experiment.get('feature_set', [{}])[0].get('feature', '?')}")
                prob_delta = experiment.get('probability_change', 0.0)
                logit_delta = experiment.get('logit_change', 0.0)
                flips = experiment.get('ablation_flips_top_prediction', False)
                print(f"    {layer_name}/{feature_name}: Δp={prob_delta:.4f}, Δlogit={logit_delta:.4f}, flips_top={flips}")
        else:
            print("    - No targeted ablations were recorded.")
        
        print("\n🎲 Random baseline ablations:")
        random_baseline = analysis.get('random_baseline_experiments', [])
        if random_baseline:
            for experiment in random_baseline:
                prob_delta = experiment.get('probability_change', 0.0)
                logit_delta = experiment.get('logit_change', 0.0)
                flips = experiment.get('ablation_flips_top_prediction', False)
                trial_idx = experiment.get('trial_index', '?')
                print(f"    Trial {trial_idx}: Δp={prob_delta:.4f}, Δlogit={logit_delta:.4f}, flips_top={flips}")
        else:
            print("    - No random baseline trials were run.")
        
        print("\n🛤️ Path ablations:")
        path_results = analysis.get('path_ablation_experiments', [])
        if path_results:
            for path_exp in path_results:
                description = path_exp.get('path_description', 'Path')
                prob_delta = path_exp.get('probability_change', 0.0)
                logit_delta = path_exp.get('logit_change', 0.0)
                flips = path_exp.get('ablation_flips_top_prediction', False)
                print(f"    {description}: Δp={prob_delta:.4f}, Δlogit={logit_delta:.4f}, flips_top={flips}")
        else:
            print("    - No path ablations were run.")
        
        summary_stats = analysis.get('summary_statistics', {})
        targeted_summary = summary_stats.get('targeted', {})
        random_summary = summary_stats.get('random_baseline', {})
        path_summary = summary_stats.get('path', {})
        random_path_summary = summary_stats.get('random_path_baseline', {})
        print("\n📈 Summary statistics:")
        print(f"    Targeted: avg|Δp|={targeted_summary.get('avg_abs_probability_change', 0.0):.4f}, flip_rate={targeted_summary.get('flip_rate', 0.0):.2%}")
        print(f"    Random baseline: avg|Δp|={random_summary.get('avg_abs_probability_change', 0.0):.4f}, flip_rate={random_summary.get('flip_rate', 0.0):.2%}")
        print(f"    Path: avg|Δp|={path_summary.get('avg_abs_probability_change', 0.0):.4f}, flip_rate={path_summary.get('flip_rate', 0.0):.2%}")
        print(f"    Random path baseline: avg|Δp|={random_path_summary.get('avg_abs_probability_change', 0.0):.4f}, flip_rate={random_path_summary.get('flip_rate', 0.0):.2%}")
        print(f"    Targeted vs Random |Δp| diff: {summary_stats.get('target_minus_random_abs_probability_change', 0.0):.4f}")
        print(f"    Targeted vs Random flip diff: {summary_stats.get('target_flip_rate_minus_random', 0.0):.4f}")
        print(f"    Path vs Random path |Δp| diff: {summary_stats.get('path_minus_random_abs_probability_change', 0.0):.4f}")
        print(f"    Path vs Random path flip diff: {summary_stats.get('path_flip_rate_minus_random', 0.0):.4f}")
        print("\n✓ Faithfulness experiments summarized\n")
        
        # Generate a visualization for the prompt.
        print("📈 Generating visualization...")
        if 'graph' in analysis and analysis['pruned_graph_stats']['n_nodes'] > 0:
            viz_path = os.path.join(RESULTS_DIR, f"attribution_graph_prompt_{prompt_idx + 1}.png")
            pipeline.attribution_graph.visualize_graph(analysis['graph'], save_path=viz_path)
            print(f"  ✓ Graph visualization saved to {viz_path}")
        else:
            print("  - Skipping visualization as no graph was generated or it was empty.")
        
        # Save the results in a format for the web app.
        save_path = os.path.join(RESULTS_DIR, f"attribution_graphs_results_prompt_{prompt_idx + 1}.json")
        
        # Create a JSON file that can be merged with others.
        final_results = {
            "analyses": {
                f"prompt_{prompt_idx + 1}": analysis
            },
            "config": config.__dict__,
            "timestamp": str(time.time())
        }
        
        # The web page doesn't use the graph object, so remove it.
        if 'graph' in final_results['analyses'][f"prompt_{prompt_idx + 1}"]:
            del final_results['analyses'][f"prompt_{prompt_idx + 1}"]['graph']

        pipeline.save_results(final_results, save_path)
        print(f"💾 Results saved to {save_path}")
        
        print("\n🎉 Analysis for this prompt completed successfully!")
        
    except Exception as e:
        print(f"❌ Error during execution: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 
