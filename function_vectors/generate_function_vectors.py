import os
import sys
from pathlib import Path
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm

# Adjust path to import from the new 'data' directory
sys.path.append(str(Path(__file__).resolve().parent.parent))
from function_vectors.data.multilingual_function_categories import FUNCTION_CATEGORIES

def generate_all_vectors():
    # Generates and saves function vectors for all English and German prompts.
    print("🚀 Starting function vector generation for both English and German...")
    
    # Load the model and tokenizer.
    print("🔧 Loading OLMo-2-7B model and tokenizer...")
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
        # Calculates the model's activation for a given prompt.
        inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=512)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)
        
        last_token_pos = inputs['attention_mask'].sum(dim=1) - 1
        last_hidden_state = outputs.hidden_states[-1]
        activation = last_hidden_state[0, last_token_pos[0], :].cpu().numpy()
        return activation.astype(np.float64)

    # Generate and save vectors for both languages.
    output_dir = Path(__file__).parent / "data" / "vectors"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for lang in ["en", "de"]:
        print(f"\n🌍 Generating vectors for {lang.upper()} prompts...")
        category_vectors = {}
        
        for category_key, data in tqdm(FUNCTION_CATEGORIES.items(), desc=f"Processing {lang.upper()} Categories"):
            prompts = data.get(lang, [])
            
            if not prompts:
                print(f"⚠️ Warning: No {lang.upper()} prompts for '{category_key}'. Skipping.")
                continue
            
            activations = [get_activation_for_prompt(p) for p in prompts]
            
            if activations:
                category_vectors[category_key] = np.mean(activations, axis=0)

        if not category_vectors:
            print(f"❌ No vectors were generated for {lang.upper()}. Aborting save.")
            continue
        
        output_path = output_dir / f"{lang}_category_vectors.npz"
        try:
            np.savez_compressed(output_path, **category_vectors)
            print(f"✅ Successfully saved {lang.upper()} vectors to: {output_path}")
        except Exception as e:
            print(f"❌ Error saving {lang.upper()} vectors: {e}")

if __name__ == "__main__":
    generate_all_vectors() 