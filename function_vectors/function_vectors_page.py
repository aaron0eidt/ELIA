import streamlit as st
import os
from pathlib import Path
import base64
import sys
import numpy as np
import matplotlib.pyplot as plt
import torch
import pandas as pd
from utilities.localization import tr
import plotly.graph_objects as go
from sklearn.decomposition import PCA
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import Dict, List
import requests
import json
from PIL import Image
from io import BytesIO
import base64
import markdown
from datetime import datetime
from utilities.feedback_survey import display_function_vector_feedback
import gc
import colorsys
import re
from thefuzz import process
import threading

# Visualization directory.
VIZ_DIR = Path(__file__).parent / "data" / "visualizations"

# Add project root to path.
sys.path.append(str(Path(__file__).resolve().parent.parent))
from function_vectors.data.multilingual_function_categories import FUNCTION_TYPES, FUNCTION_CATEGORIES
from utilities.utils import init_qwen_api

# Plot colors and symbols.
FUNCTION_TYPE_COLORS = {
    "abstractive_tasks": "#87CEEB",      # skyblue
    "multiple_choice_qa": "#90EE90",    # lightgreen
    "text_classification": "#FA8072",    # salmon
    "extractive_tasks": "#DA70D6",       # orchid
    "named_entity_recognition": "#FFD700", # gold
    "text_generation": "#F08080"         # lightcoral
}

# Legend symbols (HTML).
PLOTLY_SYMBOLS_HTML = {
    "abstractive_tasks": "●", "multiple_choice_qa": "◆",
    "text_classification": "■", "extractive_tasks": "✚",
    "named_entity_recognition": "◇", "text_generation": "□"
}

# Plotly symbols.
PLOTLY_SYMBOLS = {
    "abstractive_tasks": "circle", "multiple_choice_qa": "diamond",
    "text_classification": "square", "extractive_tasks": "cross", 
    "named_entity_recognition": "diamond-open", "text_generation": "square-open"
}

# Format category names.
def format_category_name(name):
    # Format category key to readable name.
    if name.lower().endswith('_qa'):
        # Handle '_qa' suffix.
        prefix = name[:-3].replace('_', ' ').replace('-', ' ').title()
        formatted_name = f"{prefix} QA"
    else:
        # Default formatting.
        formatted_name = name.replace('_', ' ').replace('-', ' ').title()
    
    return tr(formatted_name)


def show_function_vectors_page():
    # Function Vector Analysis page.
    # CSS icons.
    st.markdown('<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/bootstrap-icons@1.10.5/font/bootstrap-icons.css">', unsafe_allow_html=True)
    
    # API lock.
    if 'api_lock' not in st.session_state:
        st.session_state.api_lock = threading.Lock()
    
    st.markdown(f"<h1>{tr('fv_page_title')}</h1>", unsafe_allow_html=True)
    st.markdown(f"""{tr('fv_page_desc')}""", unsafe_allow_html=True)
    
    # Check viz directory.
    if not VIZ_DIR.exists():
        st.error(tr('viz_dir_not_found_error'))
        return
    
    # Category examples.
    st.header(tr('dataset_overview'))
    st.markdown(tr('dataset_overview_desc_long'))
    display_category_examples()

    st.markdown("---")

    # Visual explanation.
    st.html(f"""
    <div style='color: #ffffff; margin: 2rem 0;'>
        <h4 style='color: #87CEEB; margin-top: 0; text-align: center; margin-bottom: 1.5rem;'>{tr('how_vectors_are_made_header')}</h4>
        <p style="text-align: center; max-width: 600px; margin: auto; margin-bottom: 2rem;">{tr('how_vectors_are_made_desc')}</p>
        
        <div style="display: flex; flex-direction: column; align-items: center; font-family: 'SF Mono', 'Consolas', 'Menlo', monospace; gap: 0.2rem;">
            
            <!-- STEP 1: INPUT -->
            <div style="background-color: #333; padding: 0.8rem; border-radius: 8px; width: 90%; max-width: 600px; text-align: center; border: 1px solid #444;">
                <h5 style="margin: 0 0 0.5rem 0; color: #87CEEB; font-size: 0.9rem; letter-spacing: 1px; font-weight: bold;"><i class="bi bi-keyboard"></i> {tr('how_vectors_are_made_step1_title')}</h5>
                <code style="background: none; color: #EAEAEA; font-size: 1em;">"{tr('how_vectors_are_made_step1_example')}"</code>
            </div>
            
            <i class="bi bi-arrow-down" style="font-size: 2rem; color: #666; margin: 0.5rem 0;"></i>

            <!-- STEP 2: TOKENIZER -->
            <div style="background-color: #333; padding: 0.8rem; border-radius: 8px; width: 90%; max-width: 600px; text-align: center; border: 1px solid #444;">
                <h5 style="margin: 0 0 0.5rem 0; color: #87CEEB; font-size: 0.9rem; letter-spacing: 1px; font-weight: bold;"><i class="bi bi-segmented-nav"></i> {tr('how_vectors_are_made_step2_title')}</h5>
                <code style="background: none; color: #EAEAEA; font-size: 1em;">{tr('how_vectors_are_made_step2_example')}</code>
            </div>
            
            <i class="bi bi-arrow-down" style="font-size: 2rem; color: #666; margin: 0.5rem 0;"></i>

            <!-- STEP 3: MODEL -->
            <div style="background-color: #333; padding: 0.8rem; border-radius: 8px; width: 90%; max-width: 600px; text-align: center; border: 1px solid #444;">
                <h5 style="margin: 0 0 0.5rem 0; color: #87CEEB; font-size: 0.9rem; letter-spacing: 1px; font-weight: bold;"><i class="bi bi-cpu-fill"></i> {tr('how_vectors_are_made_step3_title')}</h5>
                <code style="background: none; color: #EAEAEA; font-size: 1em;">{tr('how_vectors_are_made_step3_desc')}</code>
            </div>

            <i class="bi bi-arrow-down" style="font-size: 2rem; color: #666; margin: 0.5rem 0;"></i>

            <!-- STEP 4: FINAL LAYER -->
            <div style="background-color: #333; padding: 0.8rem; border-radius: 8px; width: 90%; max-width: 600px; text-align: center; border: 1px solid #444;">
                <h5 style="margin: 0 0 0.5rem 0; color: #87CEEB; font-size: 0.9rem; letter-spacing: 1px; font-weight: bold;"><i class="bi bi-layer-forward"></i> {tr('how_vectors_are_made_step4_title')}</h5>
                <code style="background: none; color: #EAEAEA; font-size: 1em;">{tr('how_vectors_are_made_step4_desc')}</code>
            </div>

            <i class="bi bi-arrow-down" style="font-size: 2rem; color: #666; margin: 0.5rem 0;"></i>

            <!-- STEP 5: OUTPUT -->
            <div style="background-color: #1e1e1e; padding: 1.2rem; border-radius: 8px; width: 90%; max-width: 600px; text-align: center; border: 2px solid #90EE90;">
                <h5 style="margin: 0 0 0.5rem 0; color: #90EE90; font-size: 1rem; letter-spacing: 1px; font-weight: bold;"><i class="bi bi-check-circle-fill"></i> {tr('how_vectors_are_made_step5_title')}</h5>
                <code style="background: none; color: #90EE90; font-weight: bold; font-size: 1.1em;">[ -0.23, 1.45, -0.89, ... ]</code>
            </div>

        </div>
    </div>
    """)
    
    st.markdown("---")
    
    analysis_run = 'analysis_results' in st.session_state and 'user_input' in st.session_state

    # --- Initial Visualization ---
    # Show the 3D PCA plot before an analysis is run.
    if not analysis_run:
        st.markdown(f"<h2>{tr('pca_3d_section_header')}</h2>", unsafe_allow_html=True)
        display_3d_pca_visualization(show_description=True)
        st.markdown("---")

    # The interactive analysis section is always visible.
    st.markdown(f"<h2>{tr('interactive_analysis_section_header')}</h2>", unsafe_allow_html=True)
    display_interactive_analysis()

    # If an analysis was run, show the results.
    if analysis_run:
        st.markdown("---")
        with st.spinner(tr('running_analysis_spinner')):
            display_analysis_results(st.session_state.analysis_results, st.session_state.user_input)

    #if 'analysis_results' in st.session_state:
     #   display_function_vector_feedback()


def _trigger_and_rerun_analysis(input_text, include_attribution, include_evolution, enable_ai_explanation):
    # Triggers an analysis, saves the results, and reruns the app.
    if not input_text.strip():
        st.warning("Please enter a prompt to analyze.")
        return

    st.session_state.user_input = input_text.strip()
    st.session_state.enable_ai_explanation = enable_ai_explanation
    
    with st.spinner(tr('running_analysis_spinner')):
        try:
            results = run_interactive_analysis(input_text.strip(), True, True, enable_ai_explanation)

            if results:
                st.session_state.analysis_results = results

                # Process and store AI explanations if enabled.
                if enable_ai_explanation or "pca_explanation" in results: # Also process if loaded from cache
                    if 'api_error' in results:
                        st.warning(results['api_error'])
                    
                    if 'pca_explanation' in results and results['pca_explanation']:
                        # Split the explanation into parts based on headings.
                        explanation_parts = re.split(r'(?=\n####\s)', results['pca_explanation'].strip())
                        explanation_parts = [p.strip() for p in explanation_parts if p.strip()]
                        st.session_state.explanation_part_1 = explanation_parts[0] if len(explanation_parts) > 0 else ""
                        st.session_state.explanation_part_2 = explanation_parts[1] if len(explanation_parts) > 1 else ""
                        st.session_state.explanation_part_3 = explanation_parts[2] if len(explanation_parts) > 2 else ""

                    if 'evolution_explanation' in results and results['evolution_explanation']:
                        # Split the evolution explanation into parts.
                        evo_parts = re.split(r'(?=\n####\s)', results['evolution_explanation'].strip())
                        evo_parts = [p.strip() for p in evo_parts if p.strip()]
                        st.session_state.evolution_explanation_part_1 = evo_parts[0] if len(evo_parts) > 0 else ""
                        st.session_state.evolution_explanation_part_2 = evo_parts[1] if len(evo_parts) > 1 else ""

                if 'example_text' in st.session_state:
                    del st.session_state['example_text']
                st.rerun()
            else:
                st.error(tr('analysis_failed_error'))
        except Exception as e:
            st.error(tr('analysis_error').format(e=str(e)))
            st.info(tr('ensure_model_and_data_info'))


def display_interactive_analysis():
    # Shows the interactive analysis section of the page.
    
    # Show a section with example queries.
    st.markdown(f"**{tr('example_queries_header')}**", unsafe_allow_html=True)
    st.markdown(tr('example_queries_desc'))
    
    current_lang = st.session_state.get('lang', 'en')
    examples = {
        'en': [
            "Summarize the plot of 'Hamlet' in one sentence:",
            "The main ingredient in a Negroni cocktail is",
            "A Python function that calculates the factorial of a number is:",
            "The sentence 'The cake was eaten by the dog' is in the following voice:",
            "A good headline for an article about a new breakthrough in battery technology would be:",
            "The capital of Mongolia is",
            "The literary device in the phrase 'The wind whispered through the trees' is",
            "The French translation of 'I would like to order a coffee, please.' is:",
            "The movie 'The Matrix' can be classified into the following genre:"
        ],
        'de': [
            "Fassen Sie die Handlung von 'Hamlet' in einem Satz zusammen:",
            "Die Hauptzutat in einem Negroni-Cocktail ist",
            "Eine Python-Funktion, die die Fakultät einer Zahl berechnet, lautet:",
            "Der Satz 'Der Kuchen wurde vom Hund gefressen' steht in folgender Form:",
            "Eine gute Überschrift für einen Artikel über einen neuen Durchbruch in der Batterietechnologie wäre:",
            "Die Hauptstadt der Mongolei ist",
            "Das literarische Stilmittel im Satz 'Der Wind flüsterte durch die Bäume' ist",
            "Die französische Übersetzung von 'Ich möchte bitte einen Kaffee bestellen.' lautet:",
            "Der Film 'Die Matrix' lässt sich in folgendes Genre einteilen:"
        ]
    }
    
    # Display the examples in a 3-column grid.
    example_cols = st.columns(3)
    for i, example in enumerate(examples[current_lang]):
        with example_cols[i % 3]:
            if st.button(example, key=f"fv_example_{i}", use_container_width=True):
                # Trigger an analysis when an example is clicked.
                _trigger_and_rerun_analysis(example, True, True, True)

    # Input section
    # Add some custom CSS to style the text area.
    st.markdown("""
    <style>
    .stTextArea > div > div > textarea {
        background-color: #2b2b2b !important;
        border: 2px solid #4a90e2 !important;
        border-radius: 10px !important;
        color: #ffffff !important;
    }
    .stTextArea > div > div > textarea::placeholder {
        color: #888888 !important;
    }
    .stTextArea > div > div > textarea:focus {
        border-color: #4a90e2 !important;
        box-shadow: 0 0 0 2px rgba(74, 144, 226, 0.2) !important;
    }
    .custom-label {
        font-size: 1.25rem !important;
        font-weight: bold !important;
        margin-bottom: 0.5rem !important;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Text input area that uses the session state.
    # Use an example as the default value if one was clicked.
    default_value = st.session_state.get('user_input', '')
    
    st.markdown(f"<div class='custom-label'>{tr('input_text_label')}</div>", unsafe_allow_html=True)
    input_text = st.text_area(
        "text_area_for_analysis",
        value=default_value,
        placeholder=tr('input_text_placeholder'),
        height=100,
        help=tr('input_text_help'),
        label_visibility="collapsed"
    )
    
    # Checkbox for AI explanations.
    enable_ai_explanation = st.checkbox(tr('enable_ai_explanation_checkbox'), value=True, help=tr('enable_ai_explanation_help'))

    # Analysis button.
    if st.button(tr('analyze_button'), type="primary"):
        _trigger_and_rerun_analysis(input_text, True, True, enable_ai_explanation)


def load_model_and_tokenizer():
    # Loads and caches the model and tokenizer.
    MODEL_PATH = "./models/OLMo-2-1124-7B"
    device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        device_map="auto",
        output_hidden_states=True
    )
    return model, tokenizer, device

@st.cache_data
def _load_precomputed_vectors(lang='en', cache_version="function-vectors-2025-11-09"):
    # Loads pre-computed vectors from a file.
    vector_path = Path(__file__).parent / f"data/vectors/{lang}_category_vectors.npz"
    if not vector_path.exists():
        return None, None, f"Vector file not found for language '{lang}': {vector_path}"
    
    try:
        loaded_data = np.load(vector_path, allow_pickle=True)
        category_vectors = {key: loaded_data[key] for key in loaded_data.files}
        
        function_type_vectors = {}
        for func_type_key, category_keys in FUNCTION_TYPES.items():
            type_vectors = [category_vectors[cat_key] for cat_key in category_keys if cat_key in category_vectors]
            if type_vectors:
                function_type_vectors[func_type_key] = np.mean(type_vectors, axis=0)
        
        return function_type_vectors, category_vectors, None
    except Exception as e:
        return None, None, f"Error loading vectors for language '{lang}': {e}"

@st.cache_data(persist=True)
def _perform_analysis(input_text, include_attribution, include_evolution, lang, enable_ai_explanation, cache_version="function-vectors-2025-11-09"):
    # This function is cached and performs the main analysis.
    results = {}
    model, tokenizer, device = None, None, None

    if include_attribution or include_evolution:
        model, tokenizer, device = load_model_and_tokenizer()

    if include_attribution:
        function_type_vectors, category_vectors, error = _load_precomputed_vectors(lang)
        if error:
            results['error'] = error
            return results

        def get_input_activation(text):
            inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            with torch.no_grad():
                outputs = model(**inputs, output_hidden_states=True)
            last_token_pos = inputs['attention_mask'].sum(dim=1) - 1
            last_hidden_state = outputs.hidden_states[-1]
            activation = last_hidden_state[0, last_token_pos[0], :].cpu().numpy()
            return activation.astype(np.float64)

        def calculate_similarity(activation, vectors_dict):
            similarities = {}
            norm_activation = activation / (np.linalg.norm(activation) + 1e-8)
            for label, vector in vectors_dict.items():
                norm_vector = vector / (np.linalg.norm(vector) + 1e-8)
                similarity = np.dot(norm_activation, norm_vector)
                similarities[label] = float(similarity)
            return similarities

        input_activation = get_input_activation(input_text)
        function_type_scores = calculate_similarity(input_activation, function_type_vectors)
        category_scores = calculate_similarity(input_activation, category_vectors)
        
        results['attribution'] = {
            'function_type_scores': dict(sorted(function_type_scores.items(), key=lambda x: x[1], reverse=True)),
            'category_scores': dict(sorted(category_scores.items(), key=lambda x: x[1], reverse=True)),
            'function_types_mapping': FUNCTION_TYPES,
            'input_text': input_text,
            'input_activation': input_activation,
            'category_vectors': category_vectors,
            'function_type_vectors': function_type_vectors
        }

    if include_evolution:
        try:
            analyzer = LayerEvolutionAnalyzer(model, tokenizer, device)
            evolution_results = analyzer.analyze_text(input_text)
            results['evolution'] = evolution_results
        except Exception as e:
            results['evolution_error'] = str(e)

    if enable_ai_explanation:
        with st.spinner(tr('generating_ai_explanation_spinner')):
            api_config = init_qwen_api()
            if api_config:
                if 'attribution' in results:
                    attribution_results = results['attribution']
                    sorted_category_scores = list(attribution_results['category_scores'].items())
                    
                    # Get the top 3 categories.
                    top_3_cats_data = sorted_category_scores[:3]
                    top_cats_for_prompt = [format_category_name(cat_key) for cat_key, _ in top_3_cats_data]

                    top_types_raw = list(attribution_results['function_type_scores'].keys())[:3]
                    top_types_formatted = [format_category_name(t) for t in top_types_raw]
                    results['pca_explanation'] = explain_pca_with_llm(api_config, input_text, top_types_formatted, top_cats_for_prompt)

                if 'evolution' in results:
                    results['evolution_explanation'] = explain_evolution_with_llm(api_config, input_text, results['evolution'])
            else:
                results['api_error'] = "Qwen API key not configured. Skipping AI explanation."

    # Clean up to free memory.
    if model is not None:
        del model
        del tokenizer
        gc.collect()
        if device == 'mps':
            torch.mps.empty_cache()
        elif device == 'cuda':
            torch.cuda.empty_cache()
                        
    return results

class LayerEvolutionAnalyzer:
    def __init__(self, model, tokenizer, device):
        # Initialize the analyzer with a pre-loaded model.
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        
        # Get the number of layers.
        self.num_layers = self.model.config.num_hidden_layers
        
        # Set the model to evaluation mode.
        self.model.eval()
        
    def extract_layer_vectors(self, text: str) -> Dict[int, np.ndarray]:
        # Extracts function vectors from each layer for a given text.
        import numpy as np
        import torch
        # Tokenize the input text.
        inputs = self.tokenizer(
            text, 
            return_tensors="pt", 
            padding=True, 
            truncation=True, 
            max_length=512
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(**inputs, output_hidden_states=True)
            
        hidden_states = outputs.hidden_states
        
        layer_vectors = {}
        for i, state in enumerate(hidden_states):
            vec = state[0].mean(dim=0).cpu().numpy()
            vec = vec.astype(np.float64)
            vec = np.nan_to_num(vec, nan=0.0, posinf=1.0, neginf=-1.0)
            layer_vectors[i] = vec
        
        return layer_vectors
    
    def compute_layer_similarities(self, layer_vectors: Dict[int, np.ndarray]) -> np.ndarray:
        # Computes the cosine similarity between vectors from different layers.
        import numpy as np
        n_layers = len(layer_vectors)
        vectors = np.array([layer_vectors[i] for i in range(n_layers)])
        
        normalized_vectors = vectors / (np.linalg.norm(vectors, axis=1, keepdims=True) + 1e-8)
        
        similarity_matrix = np.dot(normalized_vectors, normalized_vectors.T)
        
        return similarity_matrix
    
    def calculate_layer_changes(self, layer_vectors: Dict[int, np.ndarray]) -> List[float]:
        # Calculates the amount of change between consecutive layers.
        import numpy as np
        changes = []
        for i in range(1, len(layer_vectors)):
            vec1 = layer_vectors[i-1]
            vec2 = layer_vectors[i]
            
            norm1 = np.linalg.norm(vec1)
            norm2 = np.linalg.norm(vec2)
            
            if norm1 == 0 or norm2 == 0:
                sim = 0
            else:
                sim = np.dot(vec1, vec2) / (norm1 * norm2)
                
            distance = 1 - sim
            changes.append(distance)
            
        return changes
    
    def analyze_text(self, text: str):
        # Performs a complete layer evolution analysis on a text.
        layer_vectors = self.extract_layer_vectors(text)
        similarity_matrix = self.compute_layer_similarities(layer_vectors)
        layer_changes = self.calculate_layer_changes(layer_vectors)

        return {
            'layer_vectors': layer_vectors,
            'similarity_matrix': similarity_matrix,
            'layer_changes': layer_changes
        }

def run_interactive_analysis(input_text, include_attribution=True, include_evolution=True, enable_ai_explanation=True):
    # A wrapper function for running the analysis from the UI.
    
    # Before running, check if models exist if not using a cached value.
    # This check relies on the fact that caching is attempted first.
    model_path = "./models/OLMo-2-1124-7B"
    if not os.path.exists(model_path):
        # We assume if the model path is missing, we are in a static environment.
        # The calling function should have already checked the cache.
        st.info("This live demo is running in a static environment. Only the pre-cached example prompts are available. Please select an example to view its analysis.")
        return None

    current_lang = st.session_state.get('lang', 'en')
    results = _perform_analysis(input_text, include_attribution, include_evolution, current_lang, enable_ai_explanation)
    
    if 'error' in results and results['error']:
        st.error(results['error'])
        return None
        
    if 'evolution_error' in results:
        st.warning(f"Layer evolution analysis failed: {results['evolution_error']}")

    if 'api_error' in results:
        st.error(results['api_error'])

    if 'attribution' in results:
        st.session_state.user_input_3d_data = results['attribution']
    
    return results

def explain_pca_with_llm(api_config, input_text, top_types, top_cats):
    # Generates an explanation for the PCA plot with an LLM.
    lang = st.session_state.get('lang', 'en')
    prompt_key = 'pca_explanation_prompt_de' if lang == 'de' else 'pca_explanation_prompt'
    
    prompt = tr(prompt_key).format(
        input_text=input_text,
        top_types=", ".join(top_types),
        top_cats=", ".join(top_cats)
    )
    explanation = _explain_with_llm(api_config, prompt)
    if "API request failed" in explanation or "Failed to generate explanation" in explanation:
        st.error(explanation)
        return None
    return explanation


def explain_evolution_with_llm(api_config, input_text, evolution_results):
    # Generates an explanation for the layer evolution charts with an LLM.
    # Extract data for the prompt.
    activation_strengths = [float(np.sqrt(np.sum(vec ** 2))) for vec in evolution_results['layer_vectors'].values()]
    layer_changes = evolution_results['layer_changes']
    
    peak_activation_layer = np.argmax(activation_strengths)
    peak_activation_strength = activation_strengths[peak_activation_layer]
    
    biggest_change_idx = np.argmax(layer_changes)
    biggest_change_start_layer = biggest_change_idx + 1
    biggest_change_end_layer = biggest_change_idx + 2
    biggest_change_magnitude = layer_changes[biggest_change_idx]

    lang = st.session_state.get('lang', 'en')
    prompt_key = 'evolution_explanation_prompt_de' if lang == 'de' else 'evolution_explanation_prompt'

    prompt = tr(prompt_key).format(
        input_text=input_text,
        peak_activation_layer=peak_activation_layer,
        peak_activation_strength=peak_activation_strength,
        biggest_change_start_layer=biggest_change_start_layer,
        biggest_change_end_layer=biggest_change_end_layer,
        biggest_change_magnitude=biggest_change_magnitude
    )
    
    explanation = _explain_with_llm(api_config, prompt)
    if "API request failed" in explanation or "Failed to generate explanation" in explanation:
        st.error(explanation)
        return None
    return explanation


@st.cache_data(persist=True)
def _explain_with_llm(_api_config, prompt, cache_version="function-vectors-2025-11-09"):
    # Makes a cached API call to the LLM.
    with st.session_state.api_lock:
        headers = {
            "Authorization": f"Bearer {_api_config['api_key']}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": "qwen2.5-vl-72b-instruct",
            "messages": [{"role": "user", "content": prompt}]
        }
        response = requests.post(
            f"{_api_config['api_endpoint']}/chat/completions",
            headers=headers,
            json=payload,
            timeout=300
        )
        # Raise an exception if the API call fails.
        response.raise_for_status()
        return response.json().get('choices', [{}])[0].get('message', {}).get('content', '')


# --- Faithfulness Verification for Function Vectors ---

def find_closest_match(query, choices):
    # Wrapper for fuzzy matching to find the best choice.
    if not query or not choices:
        return None
    match, score = process.extractOne(query, choices)
    if score > 80: # Using a similarity threshold
        return match
    return None

@st.cache_data(persist=True)
def _cached_extract_fv_claims(api_config, explanation_text, context, cache_version="function-vectors-2025-11-09"):
    # Extracts verifiable claims from an AI explanation on the function vectors page.
    with st.session_state.api_lock:
        headers = {
            "Authorization": f"Bearer {api_config['api_key']}",
            "Content-Type": "application/json"
        }
        
        # The prompt is dynamically adjusted based on the context (PCA or Evolution).
        if context == "pca":
            claim_types_details = tr("fv_claim_extraction_prompt_pca_types_details")
        elif context == "evolution":
            claim_types_details = tr("fv_claim_extraction_prompt_evolution_types_details")
        else:
            return []

        # Dynamically set the example based on context.
        if context == "pca":
            example_block = f"""{tr('fv_claim_extraction_prompt_pca_example_header')}
{tr('fv_claim_extraction_prompt_pca_example_explanation')}
{tr('fv_claim_extraction_prompt_pca_example_json')}
"""
        elif context == "evolution":
            example_block = f"""{tr('fv_claim_extraction_prompt_evolution_example_header')}
{tr('fv_claim_extraction_prompt_evolution_example_explanation')}
{tr('fv_claim_extraction_prompt_evolution_example_json')}
"""
        else:
            example_block = ""
        
        claim_extraction_prompt = f"""{tr('fv_claim_extraction_prompt_header')}

{tr('fv_claim_extraction_prompt_instruction')}

{tr('fv_claim_extraction_prompt_context_header').format(context=context)}

{tr('fv_claim_extraction_prompt_types_header')}
{claim_types_details}

{example_block}

{tr('fv_claim_extraction_prompt_analyze_header')}
"{explanation_text}"

{tr('fv_claim_extraction_prompt_footer')}
"""
        
        data = {
            "model": "qwen2.5-vl-72b-instruct",
            "messages": [{"role": "user", "content": claim_extraction_prompt}],
            "max_tokens": 1500,
            "temperature": 0.0,
            "seed": 42
        }
        
        response = requests.post(
            f"{api_config['api_endpoint']}/chat/completions",
            headers=headers,
            json=data,
            timeout=300
        )
        response.raise_for_status()
        claims_text = response.json()["choices"][0]["message"]["content"]
        
        try:
            if '```json' in claims_text:
                claims_text = re.search(r'```json\n(.*?)\n```', claims_text, re.DOTALL).group(1)
            return json.loads(claims_text)
        except (AttributeError, json.JSONDecodeError):
            return []

@st.cache_data(persist=True)
def _cached_verify_semantic_cluster_claim(api_config, claimed_clusters, actual_top_clusters, cache_version="function-vectors-2025-11-09"):
    # Uses an LLM to verify if a semantic summary of clusters is faithful to the actual top clusters.
    with st.session_state.api_lock:
        headers = {
            "Authorization": f"Bearer {api_config['api_key']}",
            "Content-Type": "application/json"
        }
        
        verification_prompt = f"""{tr('fv_semantic_verification_prompt_header')}

{tr('fv_semantic_verification_prompt_rule')}

{tr('fv_semantic_verification_prompt_actual_header')}
{actual_top_clusters}

{tr('fv_semantic_verification_prompt_claimed_header')}
"{', '.join(claimed_clusters)}"

{tr('fv_semantic_verification_prompt_task_header')}
{tr('fv_semantic_verification_prompt_task_instruction')}

{tr('fv_semantic_verification_prompt_json_instruction')}

{tr('fv_semantic_verification_prompt_footer')}
"""
    
    data = {
        "model": "qwen2.5-vl-72b-instruct",
        "messages": [{"role": "user", "content": verification_prompt}],
        "max_tokens": 400,
        "temperature": 0.0,
        "seed": 42,
        "response_format": {"type": "json_object"}
    }
    
    response = requests.post(
        f"{api_config['api_endpoint']}/chat/completions",
        headers=headers,
        json=data,
        timeout=300
    )
    response.raise_for_status()
    
    try:
        result_json = response.json()["choices"][0]["message"]["content"]
        return json.loads(result_json)
    except (json.JSONDecodeError, KeyError):
        return {"is_verified": False, "reasoning": "Could not parse the semantic verification result."}

@st.cache_data(persist=True)
def _cached_verify_justification_claim(api_config, input_prompt, category_name, justification, cache_version="function-vectors-2025-11-09"):
    # Uses an LLM to verify if a justification for a category's relevance is sound.
    with st.session_state.api_lock:
        headers = {
            "Authorization": f"Bearer {api_config['api_key']}",
            "Content-Type": "application/json"
        }
        
        verification_prompt = f"""{tr('fv_justification_verification_prompt_header')}

{tr('fv_justification_verification_prompt_rule')}

{tr('fv_justification_verification_prompt_input_header')}
"{input_prompt}"

{tr('fv_justification_verification_prompt_category_header')}
"{category_name}"

{tr('fv_justification_verification_prompt_justification_header')}
"{justification}"

{tr('fv_justification_verification_prompt_task_header')}
{tr('fv_justification_verification_prompt_task_instruction')}

{tr('fv_justification_verification_prompt_json_instruction')}

{tr('fv_justification_verification_prompt_footer')}
"""
    
    data = {
        "model": "qwen2.5-vl-72b-instruct",
        "messages": [{"role": "user", "content": verification_prompt}],
        "max_tokens": 600,
        "temperature": 0.0,
        "seed": 42,
        "response_format": {"type": "json_object"}
    }
    
    response = requests.post(
        f"{api_config['api_endpoint']}/chat/completions",
        headers=headers,
        json=data,
        timeout=300
    )
    response.raise_for_status()
    
    try:
        result_json = response.json()["choices"][0]["message"]["content"]
        return json.loads(result_json)
    except (json.JSONDecodeError, KeyError):
        return {"is_verified": False, "reasoning": "Could not parse the semantic justification result."}

def verify_fv_claims(claims, analysis_results, context):
    # Verifies claims for the function vector page.
    verification_results = []
    
    if not analysis_results:
        return [{"claim_text": c.get('claim_text', 'N/A'), "verified": False, "evidence": "Analysis results not available."} for c in claims]

    for claim in claims:
        is_verified = False
        evidence = "Could not be verified."
        details = claim.get('details', {})
        
        try:
            if context == "pca" and 'attribution' in analysis_results:
                attribution_data = analysis_results['attribution']
                claim_type = claim.get('claim_type')
                
                if claim_type == 'top_k_similarity':
                    item_type = details.get('item_type')
                    items_claimed = details.get('items', [])
                    items_claimed_lower = [str(i).lower() for i in items_claimed]
                    rank_description = details.get('rank_description')
                    
                    TOP_K = 3

                    if item_type == 'function_type':
                        actual_scores_raw = list(attribution_data['function_type_scores'].keys())
                        actual_scores_formatted = [tr(i) for i in actual_scores_raw]
                        actual_scores_lower = [name.lower() for name in actual_scores_formatted]

                        if rank_description == 'most':
                            num_claimed = len(items_claimed_lower)
                            top_n_actual_formatted = actual_scores_formatted[:num_claimed]
                            top_n_actual_lower = actual_scores_lower[:num_claimed]
                            
                            is_verified = set(items_claimed_lower) == set(top_n_actual_lower)
                            evidence = f"The top {num_claimed} function type(s) are: {top_n_actual_formatted}. "
                            if is_verified:
                                evidence += "The claim correctly identified them."
                            else:
                                evidence += f"The claimed type(s) {items_claimed} did not match the top {num_claimed}."
                        else:
                            # Default: check for presence in top K
                            top_k_actual_formatted = actual_scores_formatted[:TOP_K]
                            top_k_actual_lower = actual_scores_lower[:TOP_K]
                            unverified_items = [item for item in items_claimed_lower if item not in top_k_actual_lower]
                            is_verified = not unverified_items
                            evidence = f"Top {TOP_K} actual function types are: {top_k_actual_formatted}. "
                            if not is_verified:
                                unverified_items_original_case = [c for c in items_claimed if c.lower() in unverified_items]
                                evidence += f"The following claimed types were not found in the top {TOP_K}: {unverified_items_original_case}."
                            else:
                                evidence += f"The claimed types {items_claimed} were successfully found within the top {TOP_K}."

                    elif item_type == 'category':
                        actual_scores_raw = list(attribution_data['category_scores'].keys())
                        actual_scores_formatted = [format_category_name(i) for i in actual_scores_raw]
                        actual_scores_lower = [name.lower() for name in actual_scores_formatted]

                        if rank_description == 'most':
                            num_claimed = len(items_claimed_lower)
                            top_n_actual_formatted = actual_scores_formatted[:num_claimed]
                            top_n_actual_lower = actual_scores_lower[:num_claimed]
                            
                            is_verified = set(items_claimed_lower) == set(top_n_actual_lower)
                            evidence = f"The top {num_claimed} category/categories are: {top_n_actual_formatted}. "
                            if is_verified:
                                evidence += "The claim correctly identified them."
                            else:
                                evidence += f"The claimed category/categories {items_claimed} did not match the top {num_claimed}."
                        else:
                            # Default: check for presence in top K
                            top_k_actual_formatted = actual_scores_formatted[:TOP_K]
                            top_k_actual_lower = actual_scores_lower[:TOP_K]
                            unverified_items = [item for item in items_claimed_lower if item not in top_k_actual_lower]
                            is_verified = not unverified_items
                            evidence = f"Top {TOP_K} actual categories are: {top_k_actual_formatted}. "
                            if not is_verified:
                                unverified_items_original_case = [c for c in items_claimed if c.lower() in unverified_items]
                                evidence += f"The following claimed categories were not found in the top {TOP_K}: {unverified_items_original_case}."
                            else:
                                evidence += f"The claimed categories {items_claimed} were successfully found within the top {TOP_K}."
                
                elif claim_type == 'positional_claim':
                    cluster_names_claimed = details.get('cluster_names', [])
                    position = details.get('position')
                    
                    if position == 'near':
                        top_3_types_raw = list(attribution_data['function_type_scores'].keys())[:3]
                        top_3_types_formatted = [tr(i) for i in top_3_types_raw]
                        
                        api_config = init_qwen_api()
                        if api_config:
                            verification = _cached_verify_semantic_cluster_claim(api_config, cluster_names_claimed, top_3_types_formatted)
                            is_verified = verification.get('is_verified', False)
                            evidence = verification.get('reasoning', "Failed to get reasoning.")
                        else:
                            is_verified = False
                            evidence = "API key not configured for semantic verification."

                elif claim_type == 'category_justification_claim':
                    category_name = details.get('category_name')
                    justification = details.get('justification')
                    input_prompt = analysis_results.get('attribution', {}).get('input_text', '')

                    if not all([category_name, justification, input_prompt]):
                        evidence = "Missing data for justification verification (category, justification, or input prompt)."
                    else:
                        api_config = init_qwen_api()
                        if api_config:
                            verification = _cached_verify_justification_claim(api_config, input_prompt, category_name, justification)
                            is_verified = verification.get('is_verified', False)
                            evidence = verification.get('reasoning', "Failed to get semantic reasoning for justification.")
                        else:
                            is_verified = False
                            evidence = "API key not configured for semantic verification."

            elif context == "evolution" and 'evolution' in analysis_results:
                evolution_data = analysis_results['evolution']
                claim_type = claim.get('claim_type')
                
                if claim_type == 'peak_activation':
                    claimed_layer = details.get('layer_index')
                    activation_strengths = [float(np.sqrt(np.sum(vec ** 2))) for vec in evolution_data['layer_vectors'].values()]
                    actual_peak_layer = np.argmax(activation_strengths)
                    is_verified = (claimed_layer == actual_peak_layer)
                    evidence = f"Claimed peak activation at layer {claimed_layer}. Actual peak is at layer {actual_peak_layer}."
                
                elif claim_type == 'biggest_change':
                    claimed_start = details.get('start_layer')
                    layer_changes = evolution_data['layer_changes']
                    actual_biggest_change_idx = np.argmax(layer_changes)
                    actual_start_layer = actual_biggest_change_idx + 1
                    is_verified = (claimed_start == actual_start_layer)
                    evidence = f"Claimed biggest change starts at layer {claimed_start}. Actual biggest change is at layer {actual_start_layer} -> {actual_start_layer + 1}."

                elif claim_type == 'specific_value_claim':
                    metric = details.get('metric')
                    layer_index = details.get('layer_index')
                    value = details.get('value')

                    if metric == 'activation_strength':
                        activation_strengths = [float(np.sqrt(np.sum(vec ** 2))) for vec in evolution_data['layer_vectors'].values()]
                        # Check if layer_index is valid
                        if layer_index < len(activation_strengths):
                            actual_value = activation_strengths[layer_index]
                            is_verified = round(actual_value, 2) == round(value, 2)
                            evidence = f"Claimed activation strength for layer {layer_index} was {value}. Actual strength is {actual_value:.2f}."
                        else:
                            evidence = f"Invalid layer index {layer_index} provided."
                    
                    elif metric == 'change_magnitude':
                        layer_changes = evolution_data['layer_changes']
                        # change between L and L+1 is at index L-1 in the list
                        # So for layer_index 1 (1->2), we need list index 0.
                        change_index = layer_index - 1
                        if 0 <= change_index < len(layer_changes):
                            actual_value = layer_changes[change_index]
                            is_verified = round(actual_value, 2) == round(value, 2)
                            evidence = f"Claimed change magnitude for transition starting at layer {layer_index} was {value}. Actual magnitude is {actual_value:.2f}."
                        else:
                            evidence = f"Invalid starting layer index {layer_index} for change magnitude."

        except Exception as e:
            evidence = f"An error occurred during verification: {str(e)}"

        verification_results.append({
            'claim_text': claim.get('claim_text', 'N/A'),
            'verified': is_verified,
            'evidence': evidence
        })
        
    return verification_results

# --- End Faithfulness Verification ---


def display_category_examples():
    # Displays an explorer for the function category examples.
    st.markdown(tr('category_examples_desc'))
    
    # Add an expander with descriptions for each function type.
    with st.expander(tr('what_is_this_function_type')):
        for func_type_key in FUNCTION_TYPES.keys():
            color = FUNCTION_TYPE_COLORS.get(func_type_key, '#CCCCCC')
            st.markdown(f"""
            <div style="border-left: 5px solid {color}; padding: 0.5rem 1rem; margin-top: 1rem; background-color: #2b2b2b; border-radius: 5px;">
                <h5 style="margin: 0; color: {color};">{tr(func_type_key)}</h5>
                <p style="margin-top: 0.5rem; color: #EAEAEA;">{tr(f"desc_{func_type_key}")}</p>
            </div>
            """, unsafe_allow_html=True)

    if 'show_all_states' not in st.session_state:
        st.session_state.show_all_states = {}

    current_lang = st.session_state.get('lang', 'en')
    col1, col2 = st.columns([1, 3])

    with col1:
        st.subheader(tr('function_types_subheader'))

        # --- Restore st.radio and add CSS for highlighting ---
        func_type_keys = list(FUNCTION_TYPES.keys())
        display_names = [tr(key) for key in func_type_keys]

        # Set a default selection.
        if 'selected_func_type_key' not in st.session_state:
            st.session_state.selected_func_type_key = func_type_keys[0]
        
        # Find the index of the current selection.
        try:
            current_index = func_type_keys.index(st.session_state.selected_func_type_key)
        except ValueError:
            current_index = 0

        def on_radio_change():
            # A callback to update the session state when the radio button changes.
            selected_display_name = st.session_state.radio_selector
            if selected_display_name in display_names:
                idx = display_names.index(selected_display_name)
                st.session_state.selected_func_type_key = func_type_keys[idx]

        # Create the radio button selector.
        st.radio(
            label="Function Types",
            options=display_names,
            index=current_index,
            on_change=on_radio_change,
            key='radio_selector',
            label_visibility="collapsed"
        )
        
        # Get the key and color for the selected function type.
        selected_func_type_key = st.session_state.selected_func_type_key
        selected_color = FUNCTION_TYPE_COLORS.get(selected_func_type_key, 'lightgrey')
        
        # Add some CSS to highlight the selected radio button.
        st.markdown(f"""
        <style>
            [data-testid="stAppViewBlockContainer"] div[role="radiogroup"] > label:has(input[type="radio"]:checked) {{
                background-color: {selected_color} !important;
                border-radius: 10px;
                padding: 0.5rem 1rem;
                color: white !important;
                font-weight: bold;
            }}
            /* Ensure the text itself is white for contrast */
            [data-testid="stAppViewBlockContainer"] div[role="radiogroup"] > label:has(input[type="radio"]:checked) div {{
                color: white !important;
            }}
        </style>
        """, unsafe_allow_html=True)


    with col2:
        category_keys = FUNCTION_TYPES[selected_func_type_key]
        available_cats = [
            cat_key for cat_key in category_keys
            if cat_key in FUNCTION_CATEGORIES and current_lang in FUNCTION_CATEGORIES[cat_key]
        ]

        if not available_cats:
            st.warning(tr('no_examples_for_type'))
        else:
            # Get the color and symbol for the selected type.
            selected_display_name = tr(selected_func_type_key)
            
            # Display the header.
            st.markdown(f"<h4 style='color: #3498db; font-weight: bold;'>{tr('prompt_examples_for_category_header').format(category=selected_display_name)}</h4>", unsafe_allow_html=True)
            
            num_to_show_by_default = 9
            show_all = st.session_state.show_all_states.get(selected_func_type_key, False)
            
            if len(available_cats) > num_to_show_by_default and not show_all:
                cats_to_display = available_cats[:num_to_show_by_default]
            else:
                cats_to_display = available_cats

            # --- Display Cards ---
            num_columns = 3
            example_cols = st.columns(num_columns)
            for i, cat_key in enumerate(cats_to_display):
                examples = FUNCTION_CATEGORIES.get(cat_key, {}).get(current_lang, [])
                if examples:
                    # Use the formatter for the display name.
                    display_name = format_category_name(cat_key)
                    with example_cols[i % num_columns]:
                        with st.container():
                            st.markdown(f"""
                            <div style="border: 1px solid #e0e0e0; border-radius: 10px; padding: 1rem; height: 140px; margin-bottom: 1rem; display: flex; flex-direction: column; justify-content: space-between;">
                                <div>
                                    <p style="font-weight: bold; color: #3498db;">{display_name}</p>
                                </div>
                                <div>
                                    <p style="font-style: italic; font-size: 0.9em; color: #6c757d;">"{examples[0]}"</p>
                                </div>
                            </div>
                            """, unsafe_allow_html=True)
            
            # --- "Show More/Less" Buttons ---
            if len(available_cats) > num_to_show_by_default:
                if not show_all:
                    if st.button(tr('show_all_button').format(count=len(available_cats)), key=f"show_all_{selected_func_type_key}"):
                        st.session_state.show_all_states[selected_func_type_key] = True
                        st.rerun()
                else:
                    if st.button(tr('show_less_button'), key=f"show_less_{selected_func_type_key}"):
                        # Set to False or remove the key.
                        st.session_state.show_all_states[selected_func_type_key] = False
                        st.rerun()

def display_3d_pca_visualization(user_input_data=None, show_description=True):
    # Displays the interactive 3D PCA plot.
    import numpy as np
    current_lang = st.session_state.get('lang', 'en')

    if show_description:
        if current_lang == 'de':
            st.markdown("""
            <div style='background-color: #2b2b2b; color: #ffffff; padding: 1.5rem; border-radius: 10px; margin: 1rem 0; border-left: 5px solid #4a90e2;'>
                <h4 style='color: #4a90e2; margin-top: 0;'>Interaktive 3D-PCA von Funktionsvektoren</h4>
                <p>Diese Visualisierung stellt die hochdimensionalen 'Funktionsvektoren' verschiedener Anweisungs-Prompts in einem vereinfachten 3D-Raum mittels Hauptkomponentenanalyse (PCA) dar. Hier ist eine Aufschlüsselung dessen, was Sie sehen:</p>
                <ul>
                    <li><strong>Was sind Funktionsvektoren?</strong> Jeder Punkt in diesem Diagramm repräsentiert einen 'Funktionsvektor' – einen numerischen Fingerabdruck (ein Embedding), der den zentralen funktionalen Zweck eines bestimmten Prompts erfasst. Diese Vektoren werden aus dem letzten verborgenen Zustand des OLMo-Modells extrahiert, nachdem es einen Prompt verarbeitet hat. Prompts mit ähnlichen Funktionen haben Vektoren, die im hochdimensionalen Raum nahe beieinander liegen.</li>
                    <li><strong>Wie funktioniert PCA?</strong> PCA ist eine Technik zur Dimensionsreduktion, die komplexe, hochdimensionale Daten in ein neues, kleineres Koordinatensystem (in diesem Fall 3D) umwandelt. Dies geschieht durch die Identifizierung der Richtungen (Hauptkomponenten), in denen die Daten am stärksten variieren. Durch die Darstellung der ersten drei Hauptkomponenten können wir die wichtigsten Beziehungen zwischen den Funktionsvektoren auf eine für uns leicht interpretierbare Weise visualisieren.</li>
                    <li><strong>Worauf ist zu achten?</strong> Suchen Sie nach Punktclustern. Diese Cluster repräsentieren Gruppen von Funktionen, die das Modell als ähnlich wahrnimmt. Der Abstand zwischen den Punkten gibt ihre funktionale Ähnlichkeit an – nähere Punkte sind ähnlicher.</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
    <div style='background-color: #2b2b2b; color: #ffffff; padding: 1.5rem; border-radius: 10px; margin: 1rem 0; border-left: 5px solid #4a90e2;'>
        <h4 style='color: #4a90e2; margin-top: 0;'>Interactive 3D PCA of Function Vectors</h4>
        <p>This visualization plots the high-dimensional 'function vectors' of different instructional prompts in a simplified 3D space using <strong>Principal Component Analysis (PCA)</strong>. Here's a breakdown of what you're seeing:</p>
        <ul>
            <li><strong>What are Function Vectors?</strong> Each point on this plot represents a 'function vector'—a numerical fingerprint (an embedding) that captures the core functional purpose of a specific prompt. These vectors are extracted from the final hidden state of the OLMo model after it processes a prompt. Prompts with similar functions will have vectors that are close to each other in the high-dimensional space.</li>
            <li><strong>How does PCA work?</strong> PCA is a dimensionality reduction technique that transforms the complex, high-dimensional data into a new, smaller coordinate system (in this case, 3D). It does this by identifying the directions (principal components) where the data varies the most. By plotting the first three principal components, we can visualize the most significant relationships between the function vectors in a way that's easy for us to interpret.</li>
            <li><strong>What to look for:</strong> Look for clusters of points. These clusters represent groups of functions that the model perceives as similar. The distance between points indicates their functional similarity—closer points are more alike.</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
        st.markdown(tr('run_analysis_for_viz_info'), unsafe_allow_html=True)
    
    # --- Load the base vectors for the selected language ---
    @st.cache_data
    def load_base_vectors(lang, cache_version="function-vectors-2025-11-09"):
        import numpy as np
        vector_path = Path(__file__).parent / f"data/vectors/{lang}_category_vectors.npz"
        if not vector_path.exists():
            st.error(f"Could not find vector file for language '{lang}' at {vector_path}")
            return None
        try:
            loaded_data = np.load(vector_path, allow_pickle=True)
            return {key: loaded_data[key] for key in loaded_data.files}
        except Exception as e:
            st.error(f"Error loading vectors: {e}")
            return None

    category_vectors = load_base_vectors(current_lang)

    if category_vectors is None:
        return # Stop if we can't load the necessary data

    try:
        # Prepare data for PCA using the loaded base vectors
        categories = list(category_vectors.keys())
        vectors = np.vstack([category_vectors[cat] for cat in categories])
            
        # If user input exists, add it to the data
        if user_input_data is not None:
            input_activation = user_input_data['input_activation']
            input_text = user_input_data['input_text']
            all_vectors = np.vstack([vectors, input_activation.reshape(1, -1)])
            plot_title = tr('pca_3d_with_input_title')
        else:
            all_vectors = vectors
            plot_title = tr('pca_3d_title').format(lang=current_lang.upper())
            
        # Perform PCA
        pca = PCA(n_components=3)
        reduced_vectors = pca.fit_transform(all_vectors)
            
        # Create plotly figure
        fig = go.Figure()
            
        # Add category points grouped by function type
        category_points = reduced_vectors[:len(categories)]
        for func_type_key, cats in FUNCTION_TYPES.items():
            func_categories = [cat for cat in cats if cat in categories]
            if func_categories:
                indices = [categories.index(cat) for cat in func_categories]
                fig.add_trace(go.Scatter3d(
                    x=category_points[indices, 0], y=category_points[indices, 1], z=category_points[indices, 2],
                    mode='markers',
                    marker=dict(size=8, color=FUNCTION_TYPE_COLORS.get(func_type_key, 'gray'), symbol=PLOTLY_SYMBOLS.get(func_type_key, 'circle'), line=dict(width=1, color='black'), opacity=0.7),
                    name=tr(func_type_key),
                    text=[format_category_name(cat) for cat in func_categories],
                    hovertemplate="<b>%{text}</b><br>PC1: %{x:.3f}<br>PC2: %{y:.3f}<br>PC3: %{z:.3f}<extra></extra>"
                ))
        
        # If user input exists, add it as a special point
        if user_input_data is not None:
            user_point = reduced_vectors[-1]
            fig.add_trace(go.Scatter3d(
                x=[user_point[0]], y=[user_point[1]], z=[user_point[2]],
                mode='markers',
                marker=dict(size=12, color='red', symbol='diamond', line=dict(width=2, color='darkred')),
                name=tr('your_input_legend'),
                text=[f"{tr('your_input_legend')}: {input_text[:50]}..."],
                hovertemplate=f"<b>{tr('your_input_hover_title')}</b><br>%{{text}}<br>PC1: %{{x:.3f}}<br>PC2: %{{y:.3f}}<br>PC3: %{{z:.3f}}<extra></extra>"
            ))
            
        fig.update_layout(
            title=plot_title,
            width=1400, height=900,
            scene=dict(xaxis_title='PC1', yaxis_title='PC2', zaxis_title='PC3', camera=dict(eye=dict(x=1.5, y=1.5, z=1.5))),
            legend=dict(orientation="v", yanchor="top", y=1, xanchor="left", x=1.02, font=dict(size=10), title_text=tr('legend_title'))
        )
            
        st.plotly_chart(fig, use_container_width=True)
    
        if user_input_data is not None:
            st.markdown(tr('your_input_analysis_desc').format(input_text=input_text))
        else:
            st.markdown(f"""{tr('pca_key_insights')}""", unsafe_allow_html=True)
            
    except Exception as e:
        st.error(tr('error_creating_enhanced_pca').format(e=str(e)))

def display_analysis_results(results, input_text):
    # Displays the results of the analysis.
    
    st.success(tr('analysis_complete_success'))
    
    st.markdown(f"""
    <div style='background: linear-gradient(135deg, #2f3f70 0%, #3a4c86 100%); padding: 1rem; border-radius: 10px; color: #f5f7fb; margin: 1rem 0; border-left: 4px solid #dcae36;'>
        <h4 style='margin: 0; color: #f5f7fb;'>{tr('analyzed_text_header')}</h4>
        <p style='margin: 0.5rem 0 0 0; font-size: 1.1rem; font-style: italic; color: #e8ecf8;'>"{input_text}"</p>
    </div>
    """, unsafe_allow_html=True)
    
    # --- Show the 3D plot with the user's data first ---
    st.markdown(f"<h2>{tr('pca_3d_section_header')}</h2>", unsafe_allow_html=True)
    user_input_data = st.session_state.get('user_input_3d_data')
    display_3d_pca_visualization(user_input_data, show_description=False)

    # --- AI Explanation for PCA Plot ---
    if st.session_state.get('enable_ai_explanation') and 'explanation_part_1' in st.session_state:
        # Display the first part of the explanation.
        if st.session_state.explanation_part_1:
            explanation_html = markdown.markdown(st.session_state.explanation_part_1)
            st.markdown(
                f"<div style='background-color: #2b2b2b; color: #ffffff; padding: 1.2rem; border-radius: 10px; margin: 1rem 0; border-left: 5px solid #6EE7B7; font-size: 0.9rem;'>{explanation_html}</div>",
                unsafe_allow_html=True
            )
                
                # Faithfulness Check for PCA plot
            with st.expander(tr('faithfulness_check_expander')):
                    st.markdown(tr('fv_faithfulness_explanation_pca_html'), unsafe_allow_html=True)
                    
                    # Check for pre-cached faithfulness results first
                    if 'pca_faithfulness' in st.session_state.analysis_results:
                        verification_results = st.session_state.analysis_results['pca_faithfulness']
                    else:
                        api_config = init_qwen_api()
                        if api_config:
                            with st.spinner(tr('running_faithfulness_check_spinner')):
                                claims = _cached_extract_fv_claims(api_config, st.session_state.explanation_part_1, "pca")
                                verification_results = verify_fv_claims(claims, results, "pca")
                        else:
                            verification_results = []
                            st.warning(tr('api_key_not_configured_warning'))

                    if verification_results:
                        for result in verification_results:
                            status_text = tr('verified_status') if result['verified'] else tr('contradicted_status')
                            st.markdown(f"""
                            <div style="margin-bottom: 1rem; padding: 0.8rem; border-radius: 8px; border-left: 5px solid {'#28a745' if result['verified'] else '#dc3545'}; background-color: #1a1a1a;">
                                <p style="margin-bottom: 0.3rem;"><strong>{tr('claim_label')}:</strong> <em>"{result['claim_text']}"</em></p>
                                <p style="margin-bottom: 0.3rem;"><strong>{tr('status_label')}:</strong> {status_text}</p>
                                <p style="margin-bottom: 0;"><strong>{tr('evidence_label')}:</strong> {result['evidence']}</p>
                            </div>
                            """, unsafe_allow_html=True)
                    else:
                        st.info(tr('no_verifiable_claims_info'))

    st.markdown("---")
    
    # --- Function Type and Category Analysis ---
    if 'attribution' in results:
        attribution = results['attribution']
        
        # --- Section 1: Function Type Attribution ---
        st.markdown(f"<h2>{tr('function_types_tab')}</h2>", unsafe_allow_html=True)
        st.markdown(tr('function_type_attribution_header'))
        
        function_type_scores = attribution['function_type_scores']
        top_types = list(function_type_scores.items())[:6]
        
        # Reverse for a horizontal bar chart.
        top_types.reverse()
        
        fig = go.Figure()
        colors = [FUNCTION_TYPE_COLORS.get(name, '#CCCCCC') for name, _ in top_types]
        
        fig.add_trace(go.Bar(
            x=[score for _, score in top_types],
            y=[tr(name) for name, _ in top_types],
            orientation='h',
            marker=dict(color=colors),
            text=[f"{score:.3f}" for _, score in top_types],
            textposition='outside',
            hovertemplate='<b>%{y}</b><br>Score: %{x:.3f}<extra></extra>'
        ))
        
        fig.update_layout(
            xaxis_title=tr('attribution_score_xaxis'),
            yaxis=dict(autorange="reversed"), # Ensures y-axis is not reversed
            height=500,
            margin=dict(l=200, r=100, t=50, b=50)
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # --- AI Explanation for Function Type Plot ---
        if st.session_state.get('enable_ai_explanation') and 'explanation_part_2' in st.session_state:
            if st.session_state.explanation_part_2:
                explanation_html = markdown.markdown(st.session_state.explanation_part_2)
                st.markdown(
                    f"<div style='background-color: #2b2b2b; color: #ffffff; padding: 1.2rem; border-radius: 10px; margin: 1rem 0; border-left: 5px solid #A78BFA; font-size: 0.9rem;'>{explanation_html}</div>",
                    unsafe_allow_html=True
                )

                # Faithfulness Check for Function Type plot
                with st.expander(tr('faithfulness_check_expander')):
                    st.markdown(tr('fv_faithfulness_explanation_pca_html'), unsafe_allow_html=True)
                    
                    if 'pca_faithfulness' in st.session_state.analysis_results:
                        verification_results = st.session_state.analysis_results['pca_faithfulness']
                    else:
                        api_config = init_qwen_api()
                        if api_config:
                            with st.spinner(tr('running_faithfulness_check_spinner')):
                                claims = _cached_extract_fv_claims(api_config, st.session_state.explanation_part_2, "pca")
                                verification_results = verify_fv_claims(claims, results, "pca")
                        else:
                            verification_results = []
                            st.warning(tr('api_key_not_configured_warning'))

                    if verification_results:
                        for result in verification_results:
                            status_text = tr('verified_status') if result['verified'] else tr('contradicted_status')
                            st.markdown(f"""
                            <div style="margin-bottom: 1rem; padding: 0.8rem; border-radius: 8px; border-left: 5px solid {'#28a745' if result['verified'] else '#dc3545'}; background-color: #1a1a1a;">
                                <p style="margin-bottom: 0.3rem;"><strong>{tr('claim_label')}:</strong> <em>"{result['claim_text']}"</em></p>
                                <p style="margin-bottom: 0.3rem;"><strong>{tr('status_label')}:</strong> {status_text}</p>
                                <p style="margin-bottom: 0;"><strong>{tr('evidence_label')}:</strong> {result['evidence']}</p>
                            </div>
                            """, unsafe_allow_html=True)
                    else:
                        st.info(tr('no_verifiable_claims_info'))

        st.markdown("---")

        # --- Section 2: Category Analysis ---
        st.markdown(f"<h2>{tr('category_analysis_tab')}</h2>", unsafe_allow_html=True)
        st.markdown(tr('top_category_attribution_header'))

        category_scores = attribution['category_scores']
        top_categories = list(category_scores.items())[:20]
        
        if top_categories:
            # Get the function type for each category to color the chart.
            function_type_mapping = attribution.get('function_types_mapping', FUNCTION_TYPES)
            category_to_func_type = {
                cat: func_type
                for func_type, cats in function_type_mapping.items()
                for cat in cats
            }

            missing_categories = [cat for cat, _ in top_categories if cat not in category_to_func_type]
            if missing_categories:
                st.warning(tr('missing_category_mapping_warning').format(categories=", ".join(missing_categories)))

            filtered_categories = [(cat, score) for cat, score in top_categories if cat in category_to_func_type]

            if not filtered_categories:
                st.info(tr('no_mapped_categories_info'))
            else:
                # Restructure the data for the sunburst chart.
                leaf_labels = [format_category_name(cat_key) for cat_key, score in filtered_categories]
                leaf_values = [score for _, score in filtered_categories]

                leaf_parent_keys = [category_to_func_type[cat_key] for cat_key, _ in filtered_categories]
                function_type_order = {key: idx for idx, key in enumerate(function_type_mapping.keys())}
                parent_keys = sorted(
                    set(leaf_parent_keys),
                    key=lambda key: function_type_order.get(key, len(function_type_order))
                )
                parent_labels_map = {key: tr(key) for key in parent_keys}

                parent_values = [
                    sum(leaf_values[i] for i, parent_key in enumerate(leaf_parent_keys) if parent_key == key)
                    for key in parent_keys
                ]

                sunburst_labels = [parent_labels_map[key] for key in parent_keys] + leaf_labels
                sunburst_parents = [""] * len(parent_keys) + [parent_labels_map[key] for key in leaf_parent_keys]
                sunburst_values = parent_values + leaf_values
            
                # Create a color map for the labels.
                label_to_color_map = {
                    parent_labels_map[key]: FUNCTION_TYPE_COLORS.get(key, '#CCCCCC')
                    for key in parent_keys
                }
            
                # --- Generate gradient colors for leaves based on score ---
                def hex_to_rgb_float(h):
                    h = h.lstrip('#')
                    return [int(h[i:i+2], 16) / 255.0 for i in (0, 2, 4)]

                def rgb_float_to_hex(rgb):
                    return '#%02x%02x%02x' % tuple(int(c * 255) for c in rgb)

                leaf_scores = leaf_values
                min_score = min(leaf_scores) if leaf_scores else 0
                max_score = max(leaf_scores) if leaf_scores else 1
                score_range = max_score - min_score

                sunburst_marker_colors = []
                # Add solid colors for the parent categories.
                for key in parent_keys:
                    parent_label = parent_labels_map[key]
                    sunburst_marker_colors.append(label_to_color_map[parent_label])
                
                # Add gradient colors for the leaf categories.
                for i, parent_key in enumerate(leaf_parent_keys):
                    base_color_hex = FUNCTION_TYPE_COLORS.get(parent_key, '#CCCCCC')
                    
                    # Normalize the score for this leaf.
                    normalized_score = (leaf_scores[i] - min_score) / score_range if score_range > 0 else 0.5
                    
                    # Convert to HLS to get the original lightness.
                    r, g, b = hex_to_rgb_float(base_color_hex)
                    h, base_l, s = colorsys.rgb_to_hls(r, g, b)
                    
                    # Define a lightness range.
                    lightest_shade = 0.9
                    lightness_range = lightest_shade - base_l
                    
                    # Interpolate the lightness.
                    new_l = lightest_shade - (normalized_score * lightness_range)
                    
                    # Convert back to RGB and then to Hex.
                    new_r, new_g, new_b = colorsys.hls_to_rgb(h, new_l, s)
                    new_hex = rgb_float_to_hex((new_r, new_g, new_b))
                    sunburst_marker_colors.append(new_hex)

                # --- Highlight the top match with a stronger visual cue ---
                top_category_name, _ = filtered_categories[0]
                formatted_top_category_name = format_category_name(top_category_name)
                top_parent_key = category_to_func_type.get(top_category_name)
                top_category_parent_str = parent_labels_map.get(top_parent_key, tr('unmapped_function_type'))

                sunburst_line_widths = [1] * len(sunburst_labels)
                sunburst_line_colors = ['#333'] * len(sunburst_labels)

                try:
                    top_leaf_index = sunburst_labels.index(formatted_top_category_name)
                    sunburst_line_widths[top_leaf_index] = 5
                    sunburst_line_colors[top_leaf_index] = '#FFFFFF'
                except ValueError:
                    pass

                try:
                    top_parent_index = sunburst_labels.index(top_category_parent_str)
                    sunburst_line_widths[top_parent_index] = 5
                    sunburst_line_colors[top_parent_index] = '#FFFFFF'
                except ValueError:
                    pass

                fig = go.Figure(go.Sunburst(
                    labels=sunburst_labels,
                    parents=sunburst_parents,
                    values=sunburst_values,
                    branchvalues="total",
                    hovertemplate='<b>%{label}</b><br>Score: %{value:.3f}<extra></extra>',
                    marker=dict(
                            colors=sunburst_marker_colors,
                            line=dict(color=sunburst_line_colors, width=sunburst_line_widths)
                    ),
                        maxdepth=2,
                        textfont=dict(color='black'),
                        leaf=dict(opacity=1)
                ))
                
                fig.update_layout(
                    title=dict(
                        text=tr('sunburst_chart_title'),
                            font=dict(size=18, family="Arial", color="#EAEAEA"),
                        x=0.5
                    ),
                    height=600,
                    font=dict(family='Arial', size=12)
                )
                
                st.plotly_chart(fig, use_container_width=True)
            
            # --- AI Explanation for Category Plot ---
            if st.session_state.get('enable_ai_explanation') and 'explanation_part_3' in st.session_state:
                if st.session_state.explanation_part_3:
                    explanation_html = markdown.markdown(st.session_state.explanation_part_3)
                    st.markdown(
                        f"<div style='background-color: #2b2b2b; color: #ffffff; padding: 1.2rem; border-radius: 10px; margin: 1rem 0; border-left: 5px solid #FBBF24; font-size: 0.9rem;'>{explanation_html}</div>",
                        unsafe_allow_html=True
                    )

                    # Faithfulness Check for Category Plot
                    with st.expander(tr('faithfulness_check_expander')):
                        st.markdown(tr('fv_faithfulness_explanation_pca_html'), unsafe_allow_html=True)
                        
                        if 'pca_faithfulness' in st.session_state.analysis_results:
                            verification_results = st.session_state.analysis_results['pca_faithfulness']
                        else:
                            api_config = init_qwen_api()
                            if api_config:
                                with st.spinner(tr('running_faithfulness_check_spinner')):
                                    claims = _cached_extract_fv_claims(api_config, st.session_state.explanation_part_3, "pca")
                                    verification_results = verify_fv_claims(claims, results, "pca")
                            else:
                                verification_results = []
                                st.warning(tr('api_key_not_configured_warning'))

                        if verification_results:
                            for result in verification_results:
                                status_text = tr('verified_status') if result['verified'] else tr('contradicted_status')
                                st.markdown(f"""
                                <div style="margin-bottom: 1rem; padding: 0.8rem; border-radius: 8px; border-left: 5px solid {'#28a745' if result['verified'] else '#dc3545'}; background-color: #1a1a1a;">
                                    <p style="margin-bottom: 0.3rem;"><strong>{tr('claim_label')}:</strong> <em>"{result['claim_text']}"</em></p>
                                    <p style="margin-bottom: 0.3rem;"><strong>{tr('status_label')}:</strong> {status_text}</p>
                                    <p style="margin-bottom: 0;"><strong>{tr('evidence_label')}:</strong> {result['evidence']}</p>
                                </div>
                                """, unsafe_allow_html=True)
                        else:
                            st.info(tr('no_verifiable_claims_info'))
            else:
                st.warning("No category attribution data available to display.")
        
        st.markdown("---")

        # --- Section 3: Layer Evolution ---
        st.markdown(f"<h2>{tr('layer_evolution_tab')}</h2>", unsafe_allow_html=True)
        st.markdown(tr('layer_evolution_header'))
        if 'evolution' in results and results['evolution']:
            display_evolution_results(results['evolution'])
        else:
            st.info(tr('evolution_not_available_info'))


def display_evolution_results(evolution_results):
    # Displays the layer evolution analysis results.
    
    import plotly.graph_objects as go
    import numpy as np
    
    # Extract key metrics from the results.
    layer_vectors = evolution_results['layer_vectors']
    similarity_matrix = evolution_results['similarity_matrix']
    layer_changes = evolution_results['layer_changes']
    
    # Calculate activation strengths.
    activation_strengths = [float(np.sqrt(np.sum(vec ** 2))) for vec in layer_vectors.values()]
    
    # Display the key insights.
    col1, col2, col3 = st.columns(3)
    
    with col1:
        max_change_layer = np.argmax(layer_changes) + 1
        st.metric(
            "Biggest Change",
            f"Layer {max_change_layer}→{max_change_layer+1}",
            f"{layer_changes[max_change_layer-1]:.3f}",
            help="Layer transition with the largest representational change"
        )
    
    with col2:
        max_activation_layer = np.argmax(activation_strengths)
        st.metric(
            "Peak Activation", 
            f"Layer {max_activation_layer}",
            f"{activation_strengths[max_activation_layer]:.3f}",
            help="Layer with strongest overall activation"
        )
    
    with col3:
        avg_change = np.mean(layer_changes)
        st.metric(
            "Avg Change",
            f"{avg_change:.3f}",
            help="Average change magnitude across all layer transitions"
        )
    
    # Plot the activation strength.
    st.markdown("<h3><i class='bi bi-lightning-charge-fill'></i> Activation Strength Across Layers</h3>", unsafe_allow_html=True)
    
    # Create the line plot.
    peak_idx = np.argmax(activation_strengths)
    
    fig = go.Figure()
    
    # Add the main line with gradient colors.
    fig.add_trace(go.Scatter(
        x=list(range(len(activation_strengths))),
        y=activation_strengths,
        mode='lines+markers',
        line=dict(color='#4ECDC4', width=4),
        marker=dict(size=10, color='#45B7D1', line=dict(color='white', width=2)),
        name='Activation Strength',
        hovertemplate='<b>Layer %{x}</b><br>Strength: %{y:.3f}<extra></extra>'
    ))
    
    # Highlight the peak activation.
    fig.add_vline(
        x=peak_idx, 
        line_dash="dash", 
        line_color="#FF6B6B",
        line_width=3,
        annotation_text=f"Peak at Layer {peak_idx}",
        annotation_position="top"
    )
    
    # Add a marker for the peak.
    fig.add_trace(go.Scatter(
        x=[peak_idx],
        y=[activation_strengths[peak_idx]],
        mode='markers',
        marker=dict(size=15, color='#FF6B6B', symbol='star', line=dict(color='white', width=2)),
        name=f'Peak Layer {peak_idx}',
        hovertemplate=f'<b>Peak Layer {peak_idx}</b><br>Strength: {activation_strengths[peak_idx]:.3f}<extra></extra>'
    ))
    
    fig.update_layout(
        xaxis=dict(
            title=dict(text="Layer Index", font=dict(size=16, color='#EAEAEA'), standoff=50),
            tickfont=dict(size=14, color='#EAEAEA'),
            gridcolor='rgba(200,200,200,0.3)',
            showgrid=True,
            zeroline=False
        ),
        yaxis=dict(
            title=dict(text="Activation Strength (L2 norm)", font=dict(size=16, color='#EAEAEA')),
            tickfont=dict(size=14, color='#EAEAEA'),
            gridcolor='rgba(200,200,200,0.3)',
            showgrid=True,
            zeroline=False
        ),
        height=500,
        margin=dict(l=80, r=80, t=100, b=80),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.2,
            xanchor="center",
            x=0.5,
            font=dict(size=12, color='#EAEAEA')
        ),
        font=dict(family='Arial'),
        hovermode='x'
    )
    
    st.plotly_chart(fig, use_container_width=True)

    # --- AI Explanation for Activation Strength ---
    if st.session_state.get('enable_ai_explanation') and 'evolution_explanation_part_1' in st.session_state:
        if st.session_state.evolution_explanation_part_1:
            explanation_html = markdown.markdown(st.session_state.evolution_explanation_part_1)
            st.markdown(
                f"<div style='background-color: #2b2b2b; color: #ffffff; padding: 1.2rem; border-radius: 10px; margin: 1rem 0; border-left: 5px solid #A78BFA; font-size: 0.9rem;'>{explanation_html}</div>",
                unsafe_allow_html=True
            )

            # Faithfulness Check for Activation Strength plot
            with st.expander(tr('faithfulness_check_expander')):
                st.markdown(tr('fv_faithfulness_explanation_evolution_html'), unsafe_allow_html=True)
                
                if 'evolution_faithfulness' in st.session_state.analysis_results:
                    verification_results = st.session_state.analysis_results['evolution_faithfulness']
                else:
                    api_config = init_qwen_api()
                    if api_config:
                        with st.spinner(tr('running_faithfulness_check_spinner')):
                            claims = _cached_extract_fv_claims(api_config, st.session_state.evolution_explanation_part_1, "evolution")
                            verification_results = verify_fv_claims(claims, st.session_state.analysis_results, "evolution")
                    else:
                        verification_results = []
                        st.warning(tr('api_key_not_configured_warning'))

                if verification_results:
                    for result in verification_results:
                        status_text = tr('verified_status') if result['verified'] else tr('contradicted_status')
                        st.markdown(f"""
                        <div style="margin-bottom: 1rem; padding: 0.8rem; border-radius: 8px; border-left: 5px solid {'#28a745' if result['verified'] else '#dc3545'}; background-color: #1a1a1a;">
                            <p style="margin-bottom: 0.3rem;"><strong>{tr('claim_label')}:</strong> <em>"{result['claim_text']}"</em></p>
                            <p style="margin-bottom: 0.3rem;"><strong>{tr('status_label')}:</strong> {status_text}</p>
                            <p style="margin-bottom: 0;"><strong>{tr('evidence_label')}:</strong> {result['evidence']}</p>
                        </div>
                        """, unsafe_allow_html=True)
                else:
                    st.info(tr('no_verifiable_claims_info'))
    
    # Plot the layer changes.
    st.markdown("<h3><i class='bi bi-arrow-repeat'></i> Layer-to-Layer Changes</h3>", unsafe_allow_html=True)
    
    max_change_idx = np.argmax(layer_changes)
    
    fig2 = go.Figure()
    
    # Add the main line with gradient colors.
    fig2.add_trace(go.Scatter(
        x=list(range(1, len(layer_changes) + 1)),
        y=layer_changes,
        mode='lines+markers',
        line=dict(color='#FECA57', width=4),
        marker=dict(size=10, color='#FF9FF3', line=dict(color='white', width=2)),
        name='Layer Changes',
        hovertemplate='<b>Layer %{x}→%{customdata}</b><br>Change: %{y:.3f}<extra></extra>',
        customdata=[i+2 for i in range(len(layer_changes))]
    ))
    
    # Highlight the biggest change.
    fig2.add_vline(
        x=max_change_idx + 1, 
        line_dash="dash", 
        line_color="#FF6B6B",
        line_width=3,
        annotation_text=f"Biggest Change: {max_change_idx+1}→{max_change_idx+2}",
        annotation_position="top"
    )
    
    # Add a marker for the peak.
    fig2.add_trace(go.Scatter(
        x=[max_change_idx + 1],
        y=[layer_changes[max_change_idx]],
        mode='markers',
        marker=dict(size=15, color='#FF6B6B', symbol='diamond', line=dict(color='white', width=2)),
        name=f'Max Change: L{max_change_idx+1}→L{max_change_idx+2}',
        hovertemplate=f'<b>Max Change: Layer {max_change_idx+1}→{max_change_idx+2}</b><br>Change: {layer_changes[max_change_idx]:.3f}<extra></extra>'
    ))
    
    fig2.update_layout(
        xaxis=dict(
            title=dict(text="Layer Transition", font=dict(size=16, color='#EAEAEA'), standoff=50),
            tickfont=dict(size=14, color='#EAEAEA'),
            gridcolor='rgba(200,200,200,0.3)',
            showgrid=True,
            zeroline=False
        ),
        yaxis=dict(
            title=dict(text="Change Magnitude (Cosine Distance)", font=dict(size=16, color='#EAEAEA')),
            tickfont=dict(size=14, color='#EAEAEA'),
            gridcolor='rgba(200,200,200,0.3)',
            showgrid=True,
            zeroline=False
        ),
        height=500,
        margin=dict(l=80, r=80, t=100, b=80),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.2,
            xanchor="center",
            x=0.5,
            font=dict(size=12, color='#EAEAEA')
        ),
        font=dict(family='Arial'),
        hovermode='x'
    )
    
    st.plotly_chart(fig2, use_container_width=True)

    # --- AI Explanation for Layer Changes ---
    if st.session_state.get('enable_ai_explanation') and 'evolution_explanation_part_2' in st.session_state:
        if st.session_state.evolution_explanation_part_2:
            explanation_html = markdown.markdown(st.session_state.evolution_explanation_part_2)
            st.markdown(
                f"<div style='background-color: #2b2b2b; color: #ffffff; padding: 1.2rem; border-radius: 10px; margin: 1rem 0; border-left: 5px solid #6EE7B7; font-size: 0.9rem;'>{explanation_html}</div>",
                unsafe_allow_html=True
            )

            # Faithfulness Check for Layer Changes plot
            with st.expander(tr('faithfulness_check_expander')):
                st.markdown(tr('fv_faithfulness_explanation_evolution_html'), unsafe_allow_html=True)
                
                if 'evolution_faithfulness' in st.session_state.analysis_results:
                    verification_results = st.session_state.analysis_results['evolution_faithfulness']
                else:
                    api_config = init_qwen_api()
                    if api_config:
                        with st.spinner(tr('running_faithfulness_check_spinner')):
                            claims = _cached_extract_fv_claims(api_config, st.session_state.evolution_explanation_part_2, "evolution")
                            verification_results = verify_fv_claims(claims, st.session_state.analysis_results, "evolution")
                    else:
                        verification_results = []
                        st.warning(tr('api_key_not_configured_warning'))

                if verification_results:
                    for result in verification_results:
                        status_text = tr('verified_status') if result['verified'] else tr('contradicted_status')
                        st.markdown(f"""
                        <div style="margin-bottom: 1rem; padding: 0.8rem; border-radius: 8px; border-left: 5px solid {'#28a745' if result['verified'] else '#dc3545'}; background-color: #1a1a1a;">
                            <p style="margin-bottom: 0.3rem;"><strong>{tr('claim_label')}:</strong> <em>"{result['claim_text']}"</em></p>
                            <p style="margin-bottom: 0.3rem;"><strong>{tr('status_label')}:</strong> {status_text}</p>
                            <p style="margin-bottom: 0;"><strong>{tr('evidence_label')}:</strong> {result['evidence']}</p>
                        </div>
                        """, unsafe_allow_html=True)
                else:
                    st.info(tr('no_verifiable_claims_info'))


if __name__ == "__main__":
    from utilities.localization import initialize_localization, tr
    initialize_localization()
    show_function_vectors_page()