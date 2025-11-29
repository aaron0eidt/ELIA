# This script trains the Cross-Layer Transcoder (CLT) and plots its training loss.

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import json
import logging
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import Dict, List, Tuple
from dataclasses import dataclass
from tqdm import tqdm
import os
import random
import argparse
import glob
import itertools
from torch.optim.lr_scheduler import CosineAnnealingLR

# --- Fix import path ---
import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))
from utilities.utils import set_seed

# --- Constants ---
PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = Path(__file__).parent / "results"
CLT_SAVE_PATH = Path(__file__).parent / "models" / "clt_model.pth"
STATS_SAVE_PATH = RESULTS_DIR / "clt_training_stats.json"
PLOT_SAVE_PATH = RESULTS_DIR / "clt_training_loss.png"
DOLMA_DIR = PROJECT_ROOT / "influence_tracer" / "dolma_dataset_sample_1.6v"

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
    max_seq_length: int = 128
    n_features_per_layer: int = 512    # Back to 512 due to memory constraints
    sparsity_lambda: float = 1e-3     # Reduced from 0.01 for L1
    reconstruction_loss_weight: float = 1.0
    batch_size: int = 16               # Can be higher with 512 features
    learning_rate: float = 3e-4       # Increased from 1e-4
    training_steps: int = 1500       # Increased from 500
    device: str = str(DEVICE)

class JumpReLU(nn.Module):
    # JumpReLU activation function.
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
        
        self.encoders = nn.ModuleList([
            nn.Linear(self.hidden_size, self.n_features, bias=False)
            for _ in range(self.n_layers)
        ])
        self.decoders = nn.ModuleDict()
        for source_layer in range(self.n_layers):
            for target_layer in range(source_layer, self.n_layers):
                key = f"{source_layer}_to_{target_layer}"
                self.decoders[key] = nn.Linear(self.n_features, self.hidden_size, bias=False)
        self.activation = JumpReLU(threshold=0.0)
        self._init_weights()
    
    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                # Improved initialization (Xavier/Glorot)
                nn.init.xavier_uniform_(module.weight, gain=0.1)
    
    def encode(self, layer_idx: int, residual_activations: torch.Tensor) -> torch.Tensor:
        return self.activation(self.encoders[layer_idx](residual_activations))
    
    def decode(self, source_layer: int, target_layer: int, feature_activations: torch.Tensor) -> torch.Tensor:
        key = f"{source_layer}_to_{target_layer}"
        return self.decoders[key](feature_activations)
    
    def forward(self, residual_activations: List[torch.Tensor]) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        feature_activations = [self.encode(i, r) for i, r in enumerate(residual_activations)]
        reconstructed_mlp_outputs = []
        for target_layer in range(self.n_layers):
            reconstruction = torch.zeros_like(residual_activations[target_layer])
            for source_layer in range(target_layer + 1):
                reconstruction += self.decode(source_layer, target_layer, feature_activations[source_layer])
            reconstructed_mlp_outputs.append(reconstruction)
        return feature_activations, reconstructed_mlp_outputs

class TrainingPipeline:
    # A pipeline for training the CLT model.
    def __init__(self, config: AttributionGraphConfig):
        self.config = config
        self.device = torch.device(config.device)
        logger.info(f"Loading OLMo model from {config.model_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(config.model_path)
        
        # Configure model loading based on the device.
        model_args = {'torch_dtype': torch.float16 if "cpu" not in config.device else torch.float32}
        if "cuda" in config.device:
            model_args['device_map'] = "auto"
        
        self.model = AutoModelForCausalLM.from_pretrained(config.model_path, **model_args).to(self.device)
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        model_config = self.model.config.to_dict()
        self.clt = CrossLayerTranscoder(model_config, config).to(self.device)
        logger.info("Training Pipeline initialized successfully")

    def load_dolma_data(self, buffer_size=10000):
        """Generator that yields text samples from the Dolma dataset with shuffling."""
        json_files = glob.glob(str(DOLMA_DIR / "*.json"))
        if not json_files:
            logger.error(f"No JSON files found in {DOLMA_DIR}")
            raise FileNotFoundError(f"No training data found in {DOLMA_DIR}")
        
        logger.info(f"Found {len(json_files)} training files in {DOLMA_DIR}")
        random.shuffle(json_files)
        
        buffer = []
        
        while True:
            for file_path in json_files:
                try:
                    # Use a larger buffer size for reading
                    with open(file_path, 'r', buffering=8192*1024) as f:
                        for line in f:
                            try:
                                doc = json.loads(line)
                                text = doc.get('text', '')
                                if len(text) > 100: # Filter very short texts
                                    buffer.append(text)
                                    
                                    if len(buffer) >= buffer_size:
                                        random.shuffle(buffer)
                                        yield from buffer
                                        buffer = []
                            except json.JSONDecodeError:
                                continue
                except Exception as e:
                    logger.warning(f"Error reading {file_path}: {e}")
            
            # Yield remaining items in buffer
            if buffer:
                random.shuffle(buffer)
                yield from buffer
                buffer = []
            
            # Shuffle and restart for next epoch
            random.shuffle(json_files)

    def train_clt(self) -> Dict:
        # Trains the Cross-Layer Transcoder.
        logger.info("Starting CLT training...")
        optimizer = torch.optim.Adam(self.clt.parameters(), lr=self.config.learning_rate)
        scheduler = CosineAnnealingLR(optimizer, T_max=self.config.training_steps, eta_min=1e-6)
        
        stats = {
            'reconstruction_losses': [],
            'sparsity_losses': [],
            'total_losses': []
        }
        
        self.clt.train()
        progress_bar = tqdm(range(self.config.training_steps), desc="Training CLT")
        
        data_generator = self.load_dolma_data()

        for step in progress_bar:
            # Sample a batch of texts.
            batch_texts = []
            try:
                for _ in range(self.config.batch_size):
                    batch_texts.append(next(data_generator))
            except StopIteration:
                logger.warning("Data generator ran out of data!")
                break
            
            # Tokenize all texts at once (True batch processing)
            inputs = self.tokenizer(
                batch_texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.config.max_seq_length
            ).to(self.device)
            
            with torch.no_grad():
                outputs = self.model(**inputs, output_hidden_states=True)
                hidden_states = outputs.hidden_states[1:]
            
            feature_activations, reconstructed_outputs = self.clt(hidden_states)
            
            # --- Loss calculation ---
            # Recon loss: Sum over batch, then average later implicitly via batch division or explicit mean
            # To match previous scale: sum of MSE per sample
            recon_loss = sum(F.mse_loss(pred, target) for target, pred in zip(hidden_states, reconstructed_outputs))
            
            # L1 Sparsity Loss (Better than tanh)
            sparsity_loss = sum(torch.mean(torch.abs(features)) for features in feature_activations)
            
            loss = (self.config.reconstruction_loss_weight * recon_loss + 
                   self.config.sparsity_lambda * sparsity_loss)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            
            # Gradient Clipping (New)
            torch.nn.utils.clip_grad_norm_(self.clt.parameters(), max_norm=1.0)
            
            optimizer.step()
            scheduler.step() # Learning Rate Schedule
            
            # Normalize losses for logging (divide by number of layers approx or keep as sum)
            # Previous code accumulated and then divided by batch size.
            # Here F.mse_loss is mean over batch by default? 
            # F.mse_loss(input, target) -> mean over all elements.
            # So recon_loss is sum(mean_mse_per_layer).
            # This is fine, scale is consistent.
            
            stats['total_losses'].append(loss.item())
            stats['reconstruction_losses'].append(recon_loss.item())
            stats['sparsity_losses'].append(sparsity_loss.item())
            
            if step % 10 == 0:
                progress_bar.set_postfix({
                    "Total": f"{loss.item():.4f}",
                    "Recon": f"{recon_loss.item():.4f}",
                    "Sparsity": f"{sparsity_loss.item():.4f}",
                    "LR": f"{scheduler.get_last_lr()[0]:.2e}"
                })
            
        logger.info("CLT training completed.")
        return stats

    def save_clt(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(self.clt.state_dict(), path)
        logger.info(f"CLT model saved to {path}")

def plot_training_stats(stats_path: str, save_path: str):
    # Loads training stats and generates a plot.
    logger.info(f"Loading training stats from {stats_path}")
    with open(stats_path, 'r') as f:
        stats = json.load(f)
    
    plt.style.use('seaborn-v0_8-darkgrid')
    fig, ax1 = plt.subplots(figsize=(12, 6))

    steps = range(len(stats['total_losses']))
    
    color = 'tab:red'
    ax1.set_xlabel('Training Steps')
    ax1.set_ylabel('Total & Reconstruction Loss', color=color)
    ax1.plot(steps, stats['total_losses'], color=color, label='Total Loss', alpha=0.9, linewidth=2)
    ax1.plot(steps, stats['reconstruction_losses'], color='tab:blue', linestyle='--', label='Reconstruction Loss', alpha=1.0)
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.grid(True, which='major', linestyle='--', linewidth='0.5', color='grey')

    ax2 = ax1.twinx()
    color2 = 'tab:green'
    ax2.set_ylabel('Sparsity Loss (L1)', color=color2)
    ax2.plot(steps, stats['sparsity_losses'], color=color2, linestyle=':', label='Sparsity Loss')
    ax2.tick_params(axis='y', labelcolor=color2)
    ax2.grid(True, which='major', linestyle=':', linewidth='0.5', color='darkgrey')

    # Combine legends into a single box.
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines + lines2, labels + labels2, loc='center right', frameon=True, facecolor='white', framealpha=0.8, edgecolor='grey')

    logger.info(f"Full training plot saved to {save_path}")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def main():
    # Main function to handle training and plotting.
    
    # --- Argument Parser ---
    parser = argparse.ArgumentParser(description="Train CLT model and/or plot training stats.")
    parser.add_argument(
        '--skip-training',
        action='store_true',
        help="Skip the training process and only generate the plot from existing stats."
    )
    args = parser.parse_args()

    # Set a seed for reproducibility.
    set_seed()
    
    # Config is now updated with improvements
    config = AttributionGraphConfig()
    
    try:
        pipeline = TrainingPipeline(config)
        logger.info("Training Pipeline initialized successfully")

        if not args.skip_training:
            # Train the Cross-Layer Transcoder using Dolma dataset
            training_stats = pipeline.train_clt()
            
            os.makedirs(RESULTS_DIR, exist_ok=True)
            
            with open(STATS_SAVE_PATH, 'w') as f:
                json.dump(training_stats, f, indent=2)
            logger.info(f"Saved training stats to {STATS_SAVE_PATH}")
            
            pipeline.save_clt(CLT_SAVE_PATH)
        else:
            logger.info("--skip-training flag is set. Loading existing stats for plotting.")

        # Always plot, using either new or existing stats.
        if os.path.exists(STATS_SAVE_PATH):
            plot_training_stats(STATS_SAVE_PATH, PLOT_SAVE_PATH)
        else:
            logger.error(f"Statistics file not found at {STATS_SAVE_PATH}. Cannot generate plot. Run training first.")

        print("\n🎉 CLT training and plotting completed successfully!")
        
    except Exception as e:
        logger.error(f"❌ Error during execution: {e}", exc_info=True)

if __name__ == "__main__":
    main() 
