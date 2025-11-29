import streamlit as st
import inseq
import torch
import os
from transformers import AutoTokenizer, AutoModelForCausalLM
import json
import requests
from bs4 import BeautifulSoup
import pandas as pd
import numpy as np
from inseq.models.huggingface_model import HuggingfaceDecoderOnlyModel
import base64
from io import BytesIO
from PIL import Image
import plotly.graph_objects as go
import re
import markdown
from utilities.localization import tr
import faiss
from sentence_transformers import SentenceTransformer, util
from sentence_splitter import SentenceSplitter
import html
from utilities.utils import init_qwen_api
from utilities.feedback_survey import display_attribution_feedback
from thefuzz import process, fuzz
import gc
import time
import sys
from pathlib import Path

# Map method names to translation keys.
METHOD_DESC_KEYS = {
    "integrated_gradients": "desc_integrated_gradients",
    "occlusion": "desc_occlusion",
    "saliency": "desc_saliency"
}

# Influence tracer configuration.
sys.path.append(str(Path(__file__).resolve().parent.parent))
INDEX_DIR = os.path.join("influence_tracer", "influence_tracer_data")
INDEX_PATH = os.path.join(INDEX_DIR, "dolma_index_multi.faiss")
MAPPING_PATH = os.path.join(INDEX_DIR, "dolma_mapping_multi.json")
TRACER_MODEL_NAME = 'paraphrase-multilingual-mpnet-base-v2'

class CachedAttribution:
    # Mock object for cached attribution results.
    def __init__(self, html_content):
        self.html_content = html_content

    def show(self, display=False, return_html=True):
        return self.html_content

def load_all_attribution_models():
    # Load attribution models.
    try:
        # Device selection.
        device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
        
        # Model path.
        model_path = "./models/OLMo-2-1124-7B"
        # Use environment variable for token if needed (though local loading usually doesn't require it)
        hf_token = os.environ.get("HF_TOKEN")
        
        # Load tokenizer.
        tokenizer = AutoTokenizer.from_pretrained(model_path, token=hf_token, trust_remote_code=True)
        tokenizer.model_max_length = 512
        
        # Load model (half precision).
        base_model = AutoModelForCausalLM.from_pretrained(
            model_path,
            token=hf_token,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
            trust_remote_code=True
        )
        
        # Move to device.
        base_model = base_model.to(device)
        
        # Add special tokens.
        if tokenizer.bos_token is None:
            tokenizer.add_special_tokens({'bos_token': '<s>'})
            base_model.resize_token_embeddings(len(tokenizer))
        
        # Patch config.
        if base_model.config.bos_token_id is None:
            base_model.config.bos_token_id = tokenizer.bos_token_id
        
        
        attribution_models = {}
        
        # Integrated Gradients.
        attribution_models["integrated_gradients"] = HuggingfaceDecoderOnlyModel(
            model=base_model,
            tokenizer=tokenizer,
            device=device,
            attribution_method="integrated_gradients",
            attribution_kwargs={"n_steps": 10}
        )
        
        
        # Occlusion.
        attribution_models["occlusion"] = HuggingfaceDecoderOnlyModel(
            model=base_model,
            tokenizer=tokenizer,
            device=device,
            attribution_method="occlusion"
        )
        
        # Saliency.
        attribution_models["saliency"] = HuggingfaceDecoderOnlyModel(
            model=base_model,
            tokenizer=tokenizer,
            device=device,
            attribution_method="saliency"
        )
        
        return attribution_models, tokenizer, base_model, device
        
    except Exception as e:
        st.error(f"Error loading models: {str(e)}")
        return None, None, None, None


def load_influence_tracer_data():
    # Load influence tracer data.
    if not os.path.exists(INDEX_PATH) or not os.path.exists(MAPPING_PATH):
        return None, None, None
    
    index = faiss.read_index(INDEX_PATH)
    with open(MAPPING_PATH, 'r', encoding='utf-8') as f:
        mapping = json.load(f)
    
    device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
    model = SentenceTransformer(TRACER_MODEL_NAME, device=device)
    return index, mapping, model

@st.cache_data(persist=True)
def get_influential_docs(text_to_trace: str, lang: str):
    # Find influential training documents.
    faiss_index, doc_mapping, tracer_model = load_influence_tracer_data()
    if not faiss_index:
        return []

    # Embed input text.
    doc_embedding = tracer_model.encode([text_to_trace], convert_to_numpy=True, normalize_embeddings=True)

    # Search index (top k).
    k = 3
    similarities, indices = faiss_index.search(doc_embedding.astype('float32'), k)

    # Find similar sentences.
    results = []
    query_embedding = tracer_model.encode([text_to_trace], normalize_embeddings=True)

    for i in range(k):
        doc_id = str(indices[0][i])
        if doc_id in doc_mapping:
            doc_info = doc_mapping[doc_id]
            file_path = os.path.join("influence_tracer", "dolma_dataset_sample_1.6v", doc_info['file'])
            try:
                full_doc_text = ""
                with open(file_path, 'r', encoding='utf-8') as f:
                    for line in f:
                        try:
                            line_data = json.loads(line)
                            line_text = line_data.get('text', '')
                            # Fuzzy match snippet.
                            if fuzz.partial_ratio(doc_info['text_snippet'], line_text) > 95:
                                full_doc_text = line_text
                                break
                        except json.JSONDecodeError:
                            continue
                
                # Skip if not found.
                if not full_doc_text:
                    print(f"Warning: Could not find document snippet for doc {doc_id} in {file_path}. Skipping.")
                    continue

                # Find most similar sentence.
                splitter = SentenceSplitter(language=lang)
                sentences = splitter.split(text=full_doc_text)
                if not sentences:
                    sentences = [full_doc_text]

                # Batch encode to avoid OOM.
                sentence_embeddings = tracer_model.encode(sentences, batch_size=64, show_progress_bar=False, normalize_embeddings=True)
                cos_scores = util.pytorch_cos_sim(query_embedding, sentence_embeddings)[0]
                best_sentence_idx = torch.argmax(cos_scores).item()
                most_similar_sentence = sentences[best_sentence_idx]
                
                results.append({
                    'id': doc_id,
                    'file': doc_info['file'],
                    'source': doc_info['source'],
                    'text': full_doc_text,
                    'similarity': similarities[0][i],
                    'highlight_sentence': most_similar_sentence
                })
            except (IOError, KeyError) as e:
                print(f"Could not retrieve full text for doc {doc_id}: {e}")
                continue
    return results

# --- Qwen API for Explanations ---

@st.cache_data(persist=True)
def _cached_explain_heatmap(api_config, img_base64, csv_text, structured_prompt):
    # Cached Qwen API call for heatmap explanation.
    headers = {
        "Authorization": f"Bearer {api_config['api_key']}",
        "Content-Type": "application/json"
    }
    
    content = [{"type": "text", "text": structured_prompt}]
    if img_base64:
        content.append({
            "type": "image_url",
            "image_url": {
                "url": f"data:image/png;base64,{img_base64}"
            }
        })
    
    data = {
        "model": api_config["model"],
        "messages": [
            {
                "role": "user",
                "content": content
            }
        ],
        "max_tokens": 1200,
        "temperature": 0.2,
        "top_p": 0.95,
        "seed": 42
    }
    
    response = requests.post(
        f"{api_config['api_endpoint']}/chat/completions",
        headers=headers,
        json=data,
        timeout=300
    )
    
    # Raise exception if API call fails.
    response.raise_for_status()
    
    result = response.json()
    return result["choices"][0]["message"]["content"]

@st.cache_data(persist=True)
def generate_all_attribution_analyses(_attribution_models, _tokenizer, _base_model, _device, prompt, max_tokens, force_exact_num_tokens=False):
    # Generate text and run attributions.
    # Generate text.
    inputs = _tokenizer(prompt, return_tensors="pt").to(_device)
    
    generation_args = {
        'max_new_tokens': max_tokens,
        'do_sample': False
    }
    if force_exact_num_tokens:
        generation_args['min_new_tokens'] = max_tokens

    generated_ids = _base_model.generate(
        inputs.input_ids,
        **generation_args
    )
    generated_text = _tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    
    # Run attributions.
    all_attributions = {}
    methods = ["integrated_gradients", "occlusion", "saliency"]
    
    for method in methods:
        attributions = _attribution_models[method].attribute(
        input_texts=prompt,
        generated_texts=generated_text
    )
        all_attributions[method] = attributions
    
    return generated_text, all_attributions

def explain_heatmap_with_csv_data(api_config, image_buffer, csv_data, context_prompt, generated_text, method_name="Attribution"):
    # Generate AI explanation for heatmap.
    try:
        # Convert image to base64.
        img_base64 = None
        if image_buffer:
            image_buffer.seek(0)
            image = Image.open(image_buffer)
            
            buffered = BytesIO()
            image.save(buffered, format="PNG")
            img_base64 = base64.b64encode(buffered.getvalue()).decode()
        
        # Clean dataframe duplicates.
        df_clean = csv_data.copy()
        
        cols = pd.Series(df_clean.columns)
        if cols.duplicated().any():
            for dup in cols[cols.duplicated()].unique():
                dup_indices = cols[cols == dup].index.values
                new_names = [f"{dup} ({i+1})" for i in range(len(dup_indices))]
                cols[dup_indices] = new_names
            df_clean.columns = cols
        
        if df_clean.index.has_duplicates:
            counts = {}
            new_index = list(df_clean.index)
            duplicated_indices = df_clean.index[df_clean.index.duplicated(keep=False)]
            for i, idx in enumerate(df_clean.index):
                if idx in duplicated_indices:
                    counts[idx] = counts.get(idx, 0) + 1
                    new_index[i] = f"{idx} ({counts[idx]})"
            df_clean.index = new_index

        # --- Rule-Based Analysis ---
        unstacked = df_clean.unstack()
        unstacked.index = unstacked.index.map('{0[1]} -> {0[0]}'.format)
        
        # Top 5 individual scores.
        top_5_individual = unstacked.abs().nlargest(5).sort_index()
        top_individual_text_lines = ["\n### Top 5 Strongest Individual Connections:"]
        for label in top_5_individual.index:
            score = unstacked[label]
            top_individual_text_lines.append(f"- **{label}**: score {score:.2f}")

        # Top 5 average input scores.
        avg_input_scores = df_clean.mean(axis=1)
        top_5_average = avg_input_scores.abs().nlargest(5).sort_index()
        top_average_text_lines = ["\n### Top 5 Most Influential Input Tokens (on average over the whole generation):"]
        for input_token in top_5_average.index:
            score = avg_input_scores[input_token]
            top_average_text_lines.append(f"- **'{input_token}'**: average score {score:.2f}")
            
        # Top output token sources.
        top_output_text_lines = []
        if not df_clean.empty:
            avg_output_scores = df_clean.mean(axis=0)
            top_3_output = avg_output_scores.abs().nlargest(min(3, len(df_clean.columns))).sort_index()
            if not top_3_output.empty:
                top_output_text_lines.append("\n### Top 3 Most Influenced Generated Tokens:")
                for output_token in top_3_output.index:
                    # Find which input tokens influenced this output token the most.
                    top_sources_for_output = df_clean[output_token].abs().nlargest(min(2, len(df_clean.index))).sort_index().index.tolist()
                    if top_sources_for_output:
                        top_output_text_lines.append(f"- **'{output_token}'** was most influenced by **'{', '.join(top_sources_for_output)}'**.")

        data_text_for_llm = "\n".join(top_individual_text_lines + top_average_text_lines + top_output_text_lines)
        
        # Method-specific context.
        desc_key = METHOD_DESC_KEYS.get(method_name, "unsupported_method_desc")
        method_context = tr(desc_key)
        
        # Instruction format.
        instruction_p1 = tr('instruction_part_1_desc').format(method_name=method_name.replace('_', ' ').title())
        
        # Create prompt.
        structured_prompt = f"""{tr('ai_expert_intro')}

## {tr('analysis_details')}
- **{tr('method_being_used')}** {method_name.replace('_', ' ').title()}
- **{tr('prompt_analyzed')}** "{context_prompt}"
- **{tr('full_generated_text')}** "{generated_text}"

## {tr('method_specific_context')}
{method_context}

## {tr('instructions_for_analysis')}

{tr('instruction_part_1_header')}
{instruction_p1}

{tr('instruction_synthesis_header')}
{tr('instruction_synthesis_desc')}

{tr('instruction_color_coding')}

## {tr('data_section_header')}
{data_text_for_llm}

{tr('begin_analysis_now')}"""
        
        # Call the cached function to get the explanation.
        explanation = _cached_explain_heatmap(api_config, img_base64, data_text_for_llm, structured_prompt)
        return explanation
        
    except Exception as e:
        # Catch errors from data prep or the API call.
        st.error(f"Error generating AI explanation: {str(e)}")
        return tr("unable_to_generate_explanation")

# --- Faithfulness Verification ---

@st.cache_data(persist=True)
def _cached_extract_claims_from_explanation(api_config, explanation_text, analysis_method):
    # Cached claim extraction.
    headers = {"Authorization": f"Bearer {api_config['api_key']}", "Content-Type": "application/json"}
     
    # Set claim types.
    claim_types_details = tr("claim_extraction_prompt_types_details")
 
    claim_extraction_prompt = f"""{tr('claim_extraction_prompt_header')}

{tr('claim_extraction_prompt_instruction')}

{tr('claim_extraction_prompt_context_header').format(analysis_method=analysis_method)}

{tr('claim_extraction_prompt_types_header')}
{claim_types_details}

{tr('claim_extraction_prompt_example_header')}
{tr('claim_extraction_prompt_example_explanation')}
{tr('claim_extraction_prompt_example_json')}

{tr('claim_extraction_prompt_analyze_header')}
"{explanation_text}"

{tr('claim_extraction_prompt_instruction_footer')}
"""
     
    data = {
        "model": api_config["model"],
        "messages": [
            {
                "role": "user",
                "content": [{"type": "text", "text": claim_extraction_prompt}]
            }
        ],
        "max_tokens": 1500,
        "temperature": 0.0,  # Deterministic.
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
        # Extract from markdown if present.
        if '```json' in claims_text:
            claims_text = re.search(r'```json\n(.*?)\n```', claims_text, re.DOTALL).group(1)
        
        return json.loads(claims_text)
    except (AttributeError, json.JSONDecodeError):
        return []

@st.cache_data(persist=True)
def _cached_verify_token_justification(api_config, analysis_method, input_prompt, generated_text, token, justification):
    # Verify token justification via LLM.
    headers = {"Authorization": f"Bearer {api_config['api_key']}", "Content-Type": "application/json"}
    
    verification_prompt = f"""{tr('justification_verification_prompt_header')}

{tr('justification_verification_prompt_crucial_rule')}

{tr('justification_verification_prompt_token_location')}

{tr('justification_verification_prompt_special_tokens')}

{tr('justification_verification_prompt_evaluating_justifications')}

{tr('justification_verification_prompt_linguistic_context')}

{tr('justification_verification_prompt_collective_reasoning')}

**Analysis Method:** {analysis_method}
**Input Prompt:** "{input_prompt}"
**Generated Text:** "{generated_text}"
**Token in Question:** "{token}"
**Provided Justification:** "{justification}"

{tr('justification_verification_prompt_task_header')}
{tr('justification_verification_prompt_task_instruction')}

{tr('justification_verification_prompt_json_instruction')}

{tr('justification_verification_prompt_footer')}
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
        return {"is_verified": False, "reasoning": "Could not parse the semantic justification result."}

def verify_claims(claims, analysis_data):
    # Verify extracted claims.
    verification_results = []
    
    # Pre-calculate thresholds.
    all_scores_flat = analysis_data['scores_df'].abs().values.flatten()

    # Average influence.
    avg_input_scores_abs = analysis_data['scores_df'].mean(axis=1).abs().sort_values(ascending=False)
    avg_input_scores_raw = analysis_data['scores_df'].mean(axis=1)
    avg_output_scores = analysis_data['scores_df'].mean(axis=0).abs().sort_values(ascending=False)

    input_tokens = analysis_data['scores_df'].index.tolist()
    generated_tokens = analysis_data['scores_df'].columns.tolist()

    for claim in claims:
        is_verified = False
        evidence = "Could not be verified."
        details = claim.get('details', {})
        claim_type = claim.get('claim_type')
        
        try:
            # Clean tokens.
            if 'token' in details and isinstance(details['token'], str):
                details['token'] = re.sub(r"^\s*['\"]|['\"]\s*$", '', details['token']).strip()
            if 'tokens' in details and isinstance(details['tokens'], list):
                details['tokens'] = [re.sub(r"^\s*['\"]|['\"]\s*$", '', t).strip() for t in details['tokens']]

            if claim_type == 'attribution_claim':
                tokens_claimed = details.get('tokens', [])
                qualifier = details.get('qualifier', 'significant')
                score_type = details.get('score_type', 'peak')

                # Calculate scores.
                if score_type == 'average':
                    score_series = analysis_data['scores_df'].abs().mean(axis=1)
                    score_name = "average score"
                else: # peak
                    score_series = analysis_data['scores_df'].abs().max(axis=1)
                    score_name = "peak score"

                if score_series.empty:
                    evidence = "No attribution data available to verify claim."
                else:
                    all_attributions = sorted(
                        [{'token': token, 'attribution': score} for token, score in score_series.items()],
                        key=lambda x: x['attribution'],
                        reverse=True
                    )
                    max_score = all_attributions[0]['attribution'] if all_attributions else 0

                    if qualifier == 'high':
                        threshold = 0.70 * max_score
                        threshold_name = "high"
                    else: # 'significant'
                        threshold = 0.50 * max_score
                        threshold_name = "significant"

                    token_scores_dict = {item['token'].lower().strip(): item['attribution'] for item in all_attributions}

                    unverified_tokens = []
                    verified_tokens_details = []

                    for token in tokens_claimed:
                        # Match claims.
                        token_lower = token.lower().strip()
                        if token_lower in token_scores_dict:
                            matching_keys = [token_lower]
                        else:
                            # Generic search.
                            matching_keys = [
                                k for k in token_scores_dict.keys() 
                                if re.sub(r'\s\(\d+\)$', '', k).strip() == token_lower
                            ]

                        if not matching_keys:
                            unverified_tokens.append(f"'{token}' (not found in analysis)")
                            continue
                        
                        # Check threshold.
                        for key in matching_keys:
                            actual_score = token_scores_dict.get(key)

                            if abs(actual_score) < threshold:
                                unverified_tokens.append(f"'{key}' ({score_name}: {abs(actual_score):.2f})")
                            else:
                                verified_tokens_details.append(f"'{key}' ({score_name}: {abs(actual_score):.2f})")
                
                    is_verified = not unverified_tokens
                    if is_verified:
                        evidence = f"Verified. All claimed tokens passed the {threshold_name} threshold (> {threshold:.2f}). Details: {', '.join(verified_tokens_details)}."
                    else:
                        fail_reason = f"the following did not meet the {threshold_name} threshold (> {threshold:.2f}): {', '.join(unverified_tokens)}"
                        if verified_tokens_details:
                            evidence = f"While some tokens passed ({', '.join(verified_tokens_details)}), {fail_reason}."
                        else:
                            evidence = f"The following did not meet the {threshold_name} threshold (> {threshold:.2f}): {', '.join(unverified_tokens)}."

            elif claim_type in ['token_justification_claim', 'token_begruendung_anspruch']:
                token_val = details.get('token') or details.get('tokens')
                if isinstance(token_val, list):
                    token = ", ".join(map(str, token_val))
                else:
                    token = token_val
                
                justification = details.get('justification') or details.get('begruendung')
                input_prompt = analysis_data.get('prompt', '')
                generated_text = analysis_data.get('generated_text', '')

                if not all([token, justification, input_prompt, generated_text]):
                    evidence = "Missing data for justification verification (token, justification, or prompt)."
                else:
                    api_config = init_qwen_api()
                    if api_config:
                        verification = _cached_verify_token_justification(api_config, analysis_data['method'], input_prompt, generated_text, token, justification)
                        is_verified = verification.get('is_verified', False)
                        evidence = verification.get('reasoning', "Failed to get semantic reasoning for justification.")
                    else:
                        is_verified = False
                        evidence = "API key not configured for semantic verification."

        except Exception as e:
            evidence = f"An error occurred during verification: {str(e)}"

        verification_results.append({
            'claim_text': claim.get('claim_text', 'N/A'),
            'verified': is_verified,
            'evidence': evidence
        })
    
    return verification_results

# --- End Faithfulness Verification ---

def create_heatmap_visualization(attributions, method_name="Attribution"):
    # Create heatmap visualization.
    try:
        # Get HTML content.
        html_content = attributions.show(display=False, return_html=True)

        if not html_content:
            st.error(tr("error_inseq_no_html").format(method_name=method_name))
            return None, None, None, None

        # Parse HTML.
        soup = BeautifulSoup(html_content, 'html.parser')
        table = soup.find('table')

        if not table:
            st.error(tr("error_no_table_in_html").format(method_name=method_name))
            return None, None, None, None

        # Parse table headers.
        header_row_element = table.find('thead')
        if header_row_element:
            headers = [th.get_text(strip=True) for th in header_row_element.find_all('th')[1:]]
        else:
            # Fallback.
            first_row = table.find('tr')
            if not first_row:
                st.error(tr("error_table_no_rows").format(method_name=method_name))
                return None, None, None, None
            headers = [th.get_text(strip=True) for th in first_row.find_all('th')[1:]]

        data_rows = []
        row_labels = []

        # Parse table body.
        table_bodies = table.find_all('tbody')
        if not table_bodies:
            # Fallback.
            all_trs = table.find_all('tr')
            data_trs = all_trs[1:] if len(all_trs) > 1 else []
        else:
            data_trs = []
            for tbody in table_bodies:
                data_trs.extend(tbody.find_all('tr'))

        for tr_element in data_trs:
            all_cells = tr_element.find_all(['th', 'td'])
            if not all_cells or len(all_cells) <= 1:
                continue

            row_labels.append(all_cells[0].get_text(strip=True))

            # Convert values.
            row_data = []
            for cell in all_cells[1:]:
                text_val = cell.get_text(strip=True)
                clean_text = text_val.replace('\xa0', '').strip()
                if clean_text:
                    try:
                        row_data.append(float(clean_text))
                    except ValueError:
                        row_data.append(0.0)
                else:
                    row_data.append(0.0)
            data_rows.append(row_data)

        # Create dataframe.
        if not data_rows or not data_rows[0]:
            st.error(tr("error_failed_to_parse_rows").format(method_name=method_name))
            return None, None, None, None
            
        # --- Make labels unique ---
        def make_labels_unique(labels):
            counts = {}
            new_labels = []
            label_counts = {label: labels.count(label) for label in set(labels)}
            
            for label in labels:
                if label_counts[label] > 1:
                    counts[label] = counts.get(label, 0) + 1
                    new_labels.append(f"{label} ({counts[label]})")
                else:
                    new_labels.append(label)
            return new_labels

        unique_row_labels = make_labels_unique(row_labels)
        unique_headers = make_labels_unique(headers)
        
        parsed_df = pd.DataFrame(data_rows, index=unique_row_labels, columns=unique_headers)
        attribution_scores = parsed_df.values

        # Clean display.
        clean_headers = parsed_df.columns.tolist()
        clean_row_labels = parsed_df.index.tolist()

        # Heatmap indices.
        x_indices = list(range(len(clean_headers)))
        y_indices = list(range(len(clean_row_labels)))

        # Custom hover data.
        custom_data = np.empty(attribution_scores.shape, dtype=object)
        for i in range(len(clean_row_labels)):
            for j in range(len(clean_headers)):
                custom_data[i, j] = (clean_row_labels[i], clean_headers[j])


        fig = go.Figure(data=go.Heatmap(
            z=attribution_scores,
            x=x_indices,
            y=y_indices,
            customdata=custom_data,
            hovertemplate="Input: %{customdata[0]}<br>Generated: %{customdata[1]}<br>Score: %{z:.4f}<extra></extra>",
            colorscale='Plasma',
            hoverongaps=False,
        ))
        
        fig.update_layout(
            title=tr('heatmap_title').format(method_name=method_name),
            xaxis_title=tr('heatmap_xaxis'),
            yaxis_title=tr('heatmap_yaxis'),
            xaxis=dict(
                tickmode='array',
                tickvals=x_indices,
                ticktext=clean_headers,
                tickangle=45
            ),
            yaxis=dict(
                tickmode='array',
                tickvals=y_indices,
                ticktext=clean_row_labels,
                autorange='reversed'
            ),
            height=max(400, len(clean_row_labels) * 30),
            width=max(600, len(clean_headers) * 50)
        )
        
        # Save plot.
        buffer = BytesIO()
        fig.write_image(buffer, format='png', scale=2)
        buffer.seek(0)
        
        return fig, html_content, buffer, parsed_df
        
    except Exception as e:
        st.error(tr("error_creating_heatmap").format(e=str(e)))
        return None, None, None, None

def start_new_analysis(prompt, max_tokens, enable_explanations):
    # Start new analysis.
    # Clear old results.
    keys_to_clear = [
        'generated_text', 
        'all_attributions'
    ]
    for key in keys_to_clear:
        if key in st.session_state:
            del st.session_state[key]
            
    # Clear cache.
    for key in list(st.session_state.keys()):
        if key.startswith('influential_docs_'):
            del st.session_state[key]

    # Update prompt.
    st.session_state.attr_prompt = prompt 
    
    # Set parameters.
    st.session_state.run_request = {
        "prompt": prompt,
        "max_tokens": max_tokens,
        "enable_explanations": enable_explanations
    }

def run_analysis(prompt, max_tokens, enable_explanations, force_exact_num_tokens=False):
    # Run full analysis pipeline.
    if not prompt.strip():
        st.warning(tr('please_enter_prompt_warning'))
        return

    # Check cache.
    cache_file = os.path.join("cache", "cached_attribution_results.json")
    if os.path.exists(cache_file):
        with open(cache_file, "r", encoding="utf-8") as f:
            cached_data = json.load(f)
        if prompt in cached_data:
            print("Loading full attribution analysis from cache.")
            cached_result = cached_data[prompt]
            
            # Populate session state.
            st.session_state.generated_text = cached_result["generated_text"]
            st.session_state.prompt = prompt
            st.session_state.enable_explanations = enable_explanations
            st.session_state.qwen_api_config = init_qwen_api() if enable_explanations else None
            
            # Reconstruct attributions.
            reconstructed_attributions = {}
            for method, data in cached_result["html_contents"].items():
                reconstructed_attributions[method] = CachedAttribution(data)
                
                # Cache explanations.
                cache_key_base = f"{method}_{cached_result['generated_text']}"
                if "explanation" in data:
                    st.session_state[f"explanation_{cache_key_base}"] = data["explanation"]
                if "faithfulness_results" in data:
                    st.session_state[f"faithfulness_check_{cache_key_base}"] = data["faithfulness_results"]

            st.session_state.all_attributions = reconstructed_attributions
            
            # Store influential docs.
            if "influential_docs" in cached_result:
                st.session_state.cached_influential_docs = cached_result["influential_docs"]

            st.success(tr('analysis_complete_success'))
            return

    # Check model path.
    model_path = "./models/OLMo-2-1124-7B"
    if not os.path.exists(model_path):
        st.info("This live demo is running in a static environment. Only the pre-cached example prompts are available. Please select an example to view its analysis.")
        return

    # Load models.
    with st.spinner(tr('loading_models_spinner')):
        attribution_models, tokenizer, base_model, device = load_all_attribution_models()
    
    if not attribution_models:
        st.error(tr('failed_to_load_models_error'))
        return

    st.session_state.qwen_api_config = init_qwen_api() if enable_explanations else None
    st.session_state.enable_explanations = enable_explanations
    st.session_state.prompt = prompt

    # Generate attributions.
    with st.spinner(tr('running_attribution_analysis_spinner')):
        try:
            generated_text, all_attributions = generate_all_attribution_analyses(
                attribution_models,
                tokenizer,
                base_model,
                device,
                prompt,
                max_tokens,
                force_exact_num_tokens=force_exact_num_tokens
            )
        except Exception as e:
            st.error(f"Error in attribution analysis: {str(e)}")
            generated_text, all_attributions = None, None
    
    if not generated_text or not all_attributions:
        st.error(tr('failed_to_generate_analysis_error'))
        return

    # Store results.
    st.session_state.generated_text = generated_text
    st.session_state.all_attributions = all_attributions

    # --- Save to cache ---
    try:
        cache_file = os.path.join("cache", "cached_attribution_results.json")
        os.makedirs("cache", exist_ok=True)
        
        # Load existing cache.
        if os.path.exists(cache_file):
            with open(cache_file, "r", encoding="utf-8") as f:
                cached_data = json.load(f)
        else:
            cached_data = {}
            
        # Add result.
        html_contents = {method: attr.show(display=False, return_html=True) for method, attr in all_attributions.items()}
        cached_data[prompt] = {
            "generated_text": generated_text,
            "html_contents": html_contents 
        }

        # Write to file.
        with open(cache_file, "w", encoding="utf-8") as f:
            json.dump(cached_data, f, ensure_ascii=False, indent=4)
        print(f"Saved new analysis for '{prompt}' to cache.")

    except Exception as e:
        print(f"Warning: Could not save result to cache file. {e}")
    # --- End ---

    # Clean up.
    del attribution_models
    del tokenizer
    del base_model
    gc.collect()
    if device == 'mps':
        torch.mps.empty_cache()
    elif device == 'cuda':
        torch.cuda.empty_cache()

    st.success(tr('analysis_complete_success'))

def show_attribution_analysis():
    # Display attribution analysis page.
    # CSS icons.
    st.markdown('<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/bootstrap-icons@1.10.5/font/bootstrap-icons.css">', unsafe_allow_html=True)
    
    st.markdown(f"<h1>{tr('attr_page_title')}</h1>", unsafe_allow_html=True)
    st.markdown(f"{tr('attr_page_desc')}", unsafe_allow_html=True)
    
    # Check for new analysis request.
    if 'run_request' in st.session_state:
        request = st.session_state.pop('run_request')
        run_analysis(
            prompt=request['prompt'],
            max_tokens=request['max_tokens'],
            enable_explanations=request['enable_explanations']
        )
    
    # Main layout.
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown(f"<h2>{tr('input_header')}</h2>", unsafe_allow_html=True)
        
        # Current language.
        lang = st.session_state.get('lang', 'en')

        # Example prompts.
        example_prompts = {
            'en': [
                "The capital of France is",
                "The first person to walk on the moon was",
                "To be or not to be, that is the",
                "Once upon a time, in a land far, far away,",
                "The chemical formula for water is",
                "A stitch in time saves",
                "The opposite of hot is",
                "The main ingredients of a pizza are",
                "She opened the door and saw"
            ],
            'de': [
                "Die Hauptstadt von Frankreich ist",
                "Die erste Person auf dem Mond war",
                "Sein oder Nichtsein, das ist hier die",
                "Es war einmal, in einem weit, weit entfernten Land,",
                "Die chemische Formel für Wasser ist",
                "Was du heute kannst besorgen, das verschiebe nicht auf",
                "Das Gegenteil von heiß ist",
                "Die Hauptzutaten einer Pizza sind",
                "Sie öffnete die Tür und sah"
            ]
        }

        st.markdown('**<i class="bi bi-lightbulb"></i> Example Prompts:**', unsafe_allow_html=True)
        cols = st.columns(3)
        for i, example in enumerate(example_prompts[lang][:9]):
            with cols[i % 3]:
                st.button(
                    example, 
                    key=f"example_{i}", 
                    use_container_width=True,
                    on_click=start_new_analysis,
                    args=(example, 10, st.session_state.get('enable_explanations', True))
                )
        
        # User prompt input.
        prompt = st.text_area(
            tr('enter_prompt'),
            value=st.session_state.get('attr_prompt', ""),
            height=100,
            help=tr('enter_prompt_help'),
            placeholder=tr('prompt_placeholder_text')
        )
        
        # Token slider.
        max_tokens = st.slider(
            tr('max_new_tokens_slider'),
            min_value=1,
            max_value=50,
            value=5,
            help=tr('max_new_tokens_slider_help')
        )
        
        # AI explanation checkbox.
        enable_explanations = st.checkbox(
            tr('enable_ai_explanations'),
            value=True,
            help=tr('enable_ai_explanations_help')
        )

        # Start button.
        st.button(
            tr('generate_and_analyze_button'), 
            type="primary",
            on_click=start_new_analysis,
            args=(prompt, max_tokens, enable_explanations)
        )
    
    with col2:
        st.markdown(f"<h2>{tr('output_header')}</h2>", unsafe_allow_html=True)
        
        if hasattr(st.session_state, 'generated_text'):
            st.subheader(tr('generated_text_subheader'))
            
            # Extract generated text.
            prompt_part = st.session_state.prompt
            full_text = st.session_state.generated_text
            
            generated_part = full_text
            if full_text.startswith(prompt_part):
                generated_part = full_text[len(prompt_part):].lstrip()
            else:
                # Fallback.
                generated_part = full_text.replace(prompt_part, "", 1).strip()

            # Clean up text.
            cleaned_generated_part = re.sub(r'\n{2,}', '\n', generated_part).strip()
            escaped_generated = html.escape(cleaned_generated_part)
            escaped_prompt = html.escape(prompt_part)
            
            st.markdown(f"""
            <div style="background-color: #2b2b2b; color: #ffffff; padding: 1.2rem; border-radius: 10px; margin: 1rem 0; border: 1px solid #444;">
                <strong>{tr('input_label')}</strong> <span style="color: #60a5fa;">{escaped_prompt}</span><br>
                <strong>{tr('generated_label')}</strong> <span style="font-weight: bold; color: #fca5a5; white-space: pre-wrap;">{escaped_generated}</span>
            </div>
            """, unsafe_allow_html=True)
    
    # Display visualizations.
    if hasattr(st.session_state, 'all_attributions'):
        st.header(tr('attribution_analysis_results_header'))
        
        # Method tabs.
        tab_titles = [
            tr('saliency_tab'),
            tr('attr_tab'),
            tr('occlusion_tab')
        ]
        tabs = st.tabs(tab_titles)
        
        # Method order.
        methods = {
            "saliency": {
                "tab": tabs[0],
                "title": tr('saliency_title'),
                "description": tr('saliency_viz_desc')
            },
            "integrated_gradients": {
                "tab": tabs[1],
                "title": tr('attr_title'),
                "description": tr('attr_viz_desc')
            },
            "occlusion": {
                "tab": tabs[2],
                "title": tr('occlusion_title'),
                "description": tr('occlusion_viz_desc')
            }
        }
        
        # Generate/display visualization.
        for method_name, method_info in methods.items():
            with method_info["tab"]:
                st.subheader(f"{method_info['title']} Analysis")
                
                # Create heatmap.
                with st.spinner(tr('creating_viz_spinner').format(method_title=method_info['title'])):
                    heatmap_fig, html_content, heatmap_buffer, scores_df = create_heatmap_visualization(
                        st.session_state.all_attributions[method_name],
                        method_name=method_info['title']
                    )
                
                if heatmap_fig:
                    st.plotly_chart(heatmap_fig, use_container_width=True)
                    
                    # Heatmap legend.
                    explanation_html = f"""
                    <div style="background-color: #0E1117; border-radius: 10px; padding: 15px; margin: 10px 0; border: 1px solid #262730;">
                        <h4 style="color: #FAFAFA; margin-bottom: 10px;">{tr('how_to_read_heatmap')}</h4>
                        <ul style="color: #DCDCDC; margin-left: 20px; padding-left: 0;">
                            <li style="margin-bottom: 5px;"><strong>{tr('xaxis_label')}:</strong> {tr('xaxis_desc')}</li>
                            <li style="margin-bottom: 5px;"><strong>{tr('yaxis_label')}:</strong> {tr('yaxis_desc')}</li>
                            <li style="margin-bottom: 5px;"><strong>{tr('color_intensity_label')}:</strong> {tr('color_intensity_desc')}</li>
                            <li style="margin-bottom: 5px;"><strong>{tr('interpretation_label')}:</strong> {tr('interpretation_desc')}</li>
                            <li style="margin-bottom: 5px;"><strong>{tr('special_tokens_label')}:</strong> {tr('special_tokens_desc')}</li>
                        </ul>
                    </div>
                    """
                    st.markdown(explanation_html, unsafe_allow_html=True)

                    # AI explanation.
                    if (st.session_state.get('enable_explanations') and
                        st.session_state.get('qwen_api_config') and
                        heatmap_buffer is not None and scores_df is not None):
                        
                        explanation_cache_key = f"explanation_{method_name}_{st.session_state.generated_text}"

                        # Get/generate explanation.
                        if explanation_cache_key not in st.session_state:
                            with st.spinner(tr('generating_ai_explanations_spinner').format(method_title=method_info['title'])):
                                explanation = explain_heatmap_with_csv_data(
                                    st.session_state.qwen_api_config,
                                    heatmap_buffer,
                                    scores_df,
                                    st.session_state.prompt,
                                    st.session_state.generated_text,
                                    method_name
                                )
                                st.session_state[explanation_cache_key] = explanation
                        
                        explanation = st.session_state.get(explanation_cache_key)
                            
                        if explanation and not explanation.startswith("Error:"):
                            simple_desc = tr(METHOD_DESC_KEYS.get(method_name, "unsupported_method_desc"))
                            st.markdown(f"#### {tr('what_this_method_shows')}")
                            st.markdown(f"""
                            <div style="background-color: #2f3f70; color: #f5f7fb; padding: 1.2rem; border-radius: 12px; margin-bottom: 1rem; box-shadow: 0 12px 24px rgba(47, 63, 112, 0.35);">
                                <p style='font-size: 1.05em; font-weight: 500; margin:0; color: #f5f7fb;'>{simple_desc}</p>
                            </div>
                            """, unsafe_allow_html=True)
                            
                            html_explanation = markdown.markdown(explanation)
                            st.markdown(f"#### {tr('ai_generated_analysis')}")
                            st.markdown(f"""
                            <div style="background-color: #2b2b2b; color: #ffffff; padding: 1.2rem; border-radius: 10px; border-left: 4px solid #dcae36; font-size: 0.9rem; margin-bottom: 1rem;">
                                    {html_explanation}
                            </div>
                            """, unsafe_allow_html=True)

                            # Faithfulness Check.
                            with st.expander(tr('faithfulness_check_expander')):
                                st.markdown(tr('faithfulness_check_explanation_html'), unsafe_allow_html=True)
                                with st.spinner(tr('running_faithfulness_check_spinner')):
                                    try:
                                        # Cache check.
                                        check_cache_key = f"faithfulness_check_{method_name}_{st.session_state.generated_text}"
                                        
                                        if check_cache_key not in st.session_state:
                                            claims = _cached_extract_claims_from_explanation(
                                                st.session_state.qwen_api_config,
                                                explanation,
                                                method_name
                                            )
                                            if claims:
                                                analysis_data = {
                                                    'scores_df': scores_df,
                                                    'method': method_name,
                                                    'prompt': st.session_state.prompt,
                                                    'generated_text': st.session_state.generated_text
                                                }
                                                verification_results = verify_claims(claims, analysis_data)
                                                st.session_state[check_cache_key] = verification_results
                                            else:
                                                st.session_state[check_cache_key] = []
                                        
                                        verification_results = st.session_state[check_cache_key]

                                        if verification_results:
                                            st.markdown(f"<h6>{tr('faithfulness_check_results_header')}</h6>", unsafe_allow_html=True)
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

                                    except Exception as e:
                                        st.error(tr('faithfulness_check_error').format(e=str(e)))
                
                # Download buttons.
                st.subheader(tr("download_results_subheader"))
                col1, col2 = st.columns(2)
                
                with col1:
                        if html_content:
                         st.download_button(
                                label=tr("download_html_button").format(method_title=method_info['title']),
                            data=html_content,
                                file_name=f"{method_name}_analysis.html",
                                mime="text/html",
                                key=f"html_{method_name}"
                        )
                        if scores_df is not None:
                            st.download_button(
                                label=tr("download_csv_button"),
                                data=scores_df.to_csv().encode('utf-8'),
                                file_name=f"{method_name}_scores.csv",
                                mime="text/csv",
                                key=f"csv_raw_{method_name}"
                            )
                
                with col2:
                        if heatmap_fig:
                            img_bytes = heatmap_fig.to_image(format="png", scale=2)
                            st.download_button(
                                label=tr("download_png_button").format(method_title=method_info['title']),
                                data=img_bytes,
                                file_name=f"{method_name}_heatmap.png",
                                mime="image/png",
                                key=f"png_{method_name}"
                            )

        # Influence tracer section.
        st.markdown("---")
        st.markdown(f'<h3><i class="bi bi-compass"></i> {tr("influence_tracer_title")}</h3>', unsafe_allow_html=True)
        st.markdown(f"<div style='font-size: 1.1rem;'>{tr('influence_tracer_desc')}</div>", unsafe_allow_html=True)
        
        # Cosine similarity visual explanation.
        sentence_a = tr('influence_example_sentence_a')
        sentence_b = tr('influence_example_sentence_b')

        # SVG Diagram.
        svg_code = f"""
        <svg width="250" height="150" viewBox="0 0 250 150" xmlns="http://www.w3.org/2000/svg">
            <line x1="10" y1="130" x2="240" y2="130" stroke="#555" stroke-width="2"></line>
            <line x1="10" y1="130" x2="10" y2="10" stroke="#555" stroke-width="2"></line>
            <!-- Corrected angle arc and theta position -->
            <path d="M 49 123 A 40 40 0 0 0 42 107" fill="none" stroke="#FFD700" stroke-width="2"></path>
            <text x="50" y="115" font-family="monospace" font-size="12" fill="#FFD700">θ</text>
            <line x1="10" y1="130" x2="150" y2="30" stroke="#87CEEB" stroke-width="3"></line>
            <text x="155" y="25" font-family="monospace" font-size="12" fill="#87CEEB">Vector A</text>
            <text x="155" y="40" font-family="monospace" font-size="10" fill="#aaa">{sentence_a}</text>
            <line x1="10" y1="130" x2="170" y2="100" stroke="#90EE90" stroke-width="3"></line>
            <text x="175" y="95" font-family="monospace" font-size="12" fill="#90EE90">Vector B</text>
            <text x="175" y="110" font-family="monospace" font-size="10" fill="#aaa">{sentence_b}</text>
        </svg>
        """
        
        # Encode SVG.
        encoded_svg = base64.b64encode(svg_code.encode("utf-8")).decode("utf-8")
        image_uri = f"data:image/svg+xml;base64,{encoded_svg}"

        # Display explanation.
        st.markdown(f"""
        <div style="background-color: #2b2b2b; border-radius: 10px; padding: 1.5rem; margin: 1rem 0; border-left: 4px solid #FFD700;">
            <h4 style="color: #FFD700; margin-top: 0; margin-bottom: 1rem;">{tr('how_influence_is_found_header')}</h4>
            <div>
                <p style="font-size: 1rem;">{tr('how_influence_is_found_desc')}</p>
                <div style="font-family: 'SF Mono', 'Consolas', 'Menlo', monospace; margin-top: 1.5rem; font-size: 0.95em;">
                    <p>{tr('influence_step_1_title')}: {tr('influence_step_1_desc')}</p>
                    <p>{tr('influence_step_2_title')}: {tr('influence_step_2_desc')}</p>
                    <p>{tr('influence_step_3_title')}: {tr('influence_step_3_desc')}</p>
                </div>
            </div>
            <div style="text-align: center; margin-top: 2rem;">
                <img src="{image_uri}" alt="Cosine Similarity Diagram" />
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.write("")

        if hasattr(st.session_state, 'generated_text'):
            # Check cache first.
            if 'cached_influential_docs' in st.session_state:
                influential_docs = st.session_state.pop('cached_influential_docs')
            else:
                with st.spinner(tr('running_influence_trace_spinner')):
                    lang = st.session_state.get('lang', 'en')
                    influential_docs = get_influential_docs(st.session_state.prompt, lang)

            # Display results.
            if influential_docs:
                st.markdown(f"#### {tr('top_influential_docs_header').format(num_docs=len(influential_docs))}")
                
                # Visualize documents.
                for i, doc in enumerate(influential_docs):
                    colors = ["#A78BFA", "#7F9CF5", "#6EE7B7", "#FBBF24", "#F472B6"]
                    card_color = colors[i % len(colors)]
                    
                    full_text = doc['text']
                    highlight_sentence = doc.get('highlight_sentence', '')
                    
                    highlighted_html = ""
                    lang = st.session_state.get('lang', 'en')

                    if highlight_sentence:
                        # Normalize sentence.
                        normalized_highlight = re.sub(r'\s+', ' ', highlight_sentence).strip()
                        
                        # Fuzzy match.
                        splitter = SentenceSplitter(language=lang)
                        sentences_in_doc = splitter.split(text=full_text)
                        
                        if sentences_in_doc:
                            best_match, score = process.extractOne(normalized_highlight, sentences_in_doc)
                            start_index = full_text.find(best_match)
                            
                            if start_index != -1:
                                end_index = start_index + len(best_match)
                                
                                # Context window.
                                context_window = 500
                                snippet_start = max(0, start_index - context_window)
                                snippet_end = min(len(full_text), end_index + context_window)
                                
                                # Reconstruct HTML.
                                before = html.escape(full_text[snippet_start:start_index])
                                highlight = html.escape(best_match)
                                after = html.escape(full_text[end_index:snippet_end])
                                
                                # Ellipses.
                                start_ellipsis = "... " if snippet_start > 0 else ""
                                end_ellipsis = " ..." if snippet_end < len(full_text) else ""
                                
                                highlighted_html = (
                                    f"{start_ellipsis}{before}"
                                    f'<mark style="background-color: {card_color}77; color: #DCDCDC; padding: 2px 4px; border-radius: 4px; font-weight: bold;">{highlight}</mark>'
                                    f"{after}{end_ellipsis}"
                                )

                    # Fallback.
                    if not highlighted_html:
                        highlighted_html = html.escape(full_text)
                    
                    st.markdown(f"""
                    <div style="border: 1px solid #262730; border-left: 5px solid {card_color}; border-radius: 10px; padding: 1.5rem; margin-bottom: 1.5rem; background-color: #0E1117; box-shadow: 0 4px 8px rgba(0,0,0,0.2);">
                        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 1rem;">
                            <span style="font-size: 1.1rem; color: #FAFAFA; font-weight: 600;"><i class="bi bi-journal-text"></i> {tr('source_label')}: {doc['source']}</span>
                            <span style="font-size: 1.1rem; color: {card_color}; background-color: {card_color}22; padding: 0.3rem 0.8rem; border-radius: 15px; font-weight: bold;">
                                <i class="bi bi-stars"></i> {tr('similarity_label')}: {doc['similarity']:.3f}
                            </span>
                        </div>
                        <div style="background-color: #1a1a1a; color: #DCDCDC; padding: 1rem; border-radius: 8px; font-family: 'Courier New', Courier, monospace; white-space: pre-wrap; word-wrap: break-word; max-height: 300px; overflow-y: auto;">
                            {highlighted_html.strip()}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                # Index check.
                if not os.path.exists(INDEX_PATH) or not os.path.exists(MAPPING_PATH):
                    st.warning(tr('influence_index_not_found_warning'))
                else:
                    st.info(tr('no_influential_docs_found'))
        else:
            st.info(tr('run_analysis_for_influence_info'))

    # Feedback survey (optional).
    #if 'all_attributions' in st.session_state:
    #    display_attribution_feedback()


if __name__ == "__main__":
    show_attribution_analysis() 