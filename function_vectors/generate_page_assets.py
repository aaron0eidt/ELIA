import os
import sys
from pathlib import Path
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
import json
import plotly.graph_objects as go
from sklearn.decomposition import PCA

# Adjust path to import from the new 'data' directory
sys.path.append(str(Path(__file__).resolve().parent.parent))
from function_vectors.data.multilingual_function_categories import FUNCTION_CATEGORIES, FUNCTION_TYPES

def generate_all_assets():
    # Generates all pre-computed assets for the Function Vectors page.
    print("🚀 Starting generation of all page assets...")
    
    # Load the model and tokenizer.
    print("🔧 Loading OLMo-2-7B model...")
    try:
        model_path = "./models/OLMo-2-1124-7B"
        device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
        
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left"
        
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
            device_map="auto",
            output_hidden_states=True
        )
        print(f"✅ Model loaded successfully on device: {device}")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return

    # Function to get activation vectors.
    def get_activation_for_prompt(prompt):
        inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=512)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)
        last_token_pos = inputs['attention_mask'].sum(dim=1) - 1
        last_hidden_state = outputs.hidden_states[-1]
        activation = last_hidden_state[0, last_token_pos[0], :].cpu().numpy()
        return activation.astype(np.float64)

    # Generate and save function vectors.
    output_dir = Path(__file__).parent / "data" / "vectors"
    output_dir.mkdir(parents=True, exist_ok=True)
    all_vectors_by_lang = {}

    for lang in ["en", "de"]:
        print(f"\n🌍 Generating vectors for {lang.upper()} prompts...")
        category_vectors = {}
        for category_key, data in tqdm(FUNCTION_CATEGORIES.items(), desc=f"Processing {lang.upper()}"):
            prompts = data.get(lang, [])
            if not prompts: continue
            activations = [get_activation_for_prompt(p) for p in prompts]
            if activations:
                category_vectors[category_key] = np.mean(activations, axis=0)
        
        all_vectors_by_lang[lang] = category_vectors.copy()

        output_path = output_dir / f"{lang}_category_vectors.npz"
        np.savez_compressed(output_path, **category_vectors)
        print(f"✅ Saved {lang.upper()} vectors to: {output_path}")

    # Generate and save 3D PCA visualizations.
    viz_dir = Path(__file__).parent / "data" / "visualizations"
    viz_dir.mkdir(parents=True, exist_ok=True)
    
    for lang, vectors_to_plot in all_vectors_by_lang.items():
        print(f"\n🎨 Generating 3D PCA visualization for {lang.upper()}...")
        if not vectors_to_plot:
            print(f"⚠️ Skipping PCA for {lang.upper()} as vectors are missing.")
            continue
        
        try:
            categories = list(vectors_to_plot.keys())
            vectors = np.vstack([vectors_to_plot[cat] for cat in categories])
            
            pca = PCA(n_components=3)
            reduced_vectors = pca.fit_transform(vectors)
            
            # Define colors and symbols for the plot.
            func_type_keys = list(FUNCTION_TYPES.keys())
            colors = ["skyblue", "lightgreen", "salmon", "orchid", "gold", "lightcoral"]
            symbols = ["circle", "diamond", "square", "cross", "diamond-open", "square-open"]
            function_type_colors = {key: colors[i % len(colors)] for i, key in enumerate(func_type_keys)}
            plotly_symbols = {key: symbols[i % len(symbols)] for i, key in enumerate(func_type_keys)}
            
            fig = go.Figure()
            for func_type_key, cats in FUNCTION_TYPES.items():
                func_categories = [cat for cat in cats if cat in categories]
                if func_categories:
                    indices = [categories.index(cat) for cat in func_categories]
                    fig.add_trace(go.Scatter3d(
                        x=reduced_vectors[indices, 0], y=reduced_vectors[indices, 1], z=reduced_vectors[indices, 2],
                        mode='markers',
                        marker=dict(size=8, color=function_type_colors.get(func_type_key, 'gray'), symbol=plotly_symbols.get(func_type_key, 'circle'), line=dict(width=1, color='black'), opacity=0.8),
                        name=func_type_key.replace("_", " ").title(),
                        text=[cat.replace("_", " ").title() for cat in func_categories],
                        hovertemplate="<b>%{text}</b><br>PC1: %{x:.3f}<br>PC2: %{y:.3f}<br>PC3: %{z:.3f}<extra></extra>"
                    ))
            
            fig.update_layout(
                title=f"3D PCA of {lang.upper()} Function Vector Categories",
                width=1400, height=900,
                scene=dict(xaxis_title='PC1', yaxis_title='PC2', zaxis_title='PC3'),
                legend_title_text='Function Types'
            )
            
            # Save the plot to an HTML file.
            file_suffix = "pca_3d_categories_layer_-1.html"
            viz_path = viz_dir / f"{lang}_{file_suffix}"
            fig.write_html(viz_path)
            print(f"✅ Saved {lang.upper()} 3D PCA visualization to: {viz_path}")
        except Exception as e:
            print(f"❌ Failed to generate PCA plot for {lang.upper()}: {e}")

    # Layer evolution data is handled dynamically in the app.
    print("\n✅ Layer Evolution analysis is handled dynamically in the app. No pre-computation needed.")
    print("\n🎉 All assets generated successfully!")

if __name__ == "__main__":
    generate_all_assets() 