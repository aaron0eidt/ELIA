#!/usr/bin/env python3
# Interactive attribution graphs for circuit tracing.

import streamlit as st
import json
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import networkx as nx
import numpy as np
from pathlib import Path
import requests
import base64
from io import BytesIO
from PIL import Image
from utilities.localization import tr
import markdown
from utilities.utils import init_qwen_api
import re
from utilities.feedback_survey import display_circuit_trace_feedback
from fuzzywuzzy import process
from typing import Set, Optional, List

# --- Qwen API for Explanations ---
@st.cache_data(persist=True)
def explain_circuit_visualization(_api_config, img_base64, structured_prompt, max_tokens_for_request=750):
    # Generate circuit trace visualization explanation.
    try:
        # Prepare the API request.
        headers = {
            "Authorization": f"Bearer {_api_config['api_key']}",
            "Content-Type": "application/json"
        }
        
        data = {
            "model": _api_config["model"],
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": structured_prompt
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{img_base64}"
                            }
                        }
                    ]
                }
            ],
            "max_tokens": max_tokens_for_request,
            "temperature": 0.2,
            "top_p": 0.95
        }
        
        # Make the API request.
        response = requests.post(
            f"{_api_config['api_endpoint']}/chat/completions",
            headers=headers,
            json=data,
            timeout=120
        )
        
        if response.status_code == 200:
            result = response.json()
            return result["choices"][0]["message"]["content"]
        else:
            return f"API request failed with status {response.status_code}: {response.text}"
        
    except Exception as e:
        return f"Error generating explanation: {str(e)}"

# --- Faithfulness Verification for Circuit Tracing ---


def _extract_interpretations_from_text(claim_text):
    # Extract feature interpretations from claim.
    if not claim_text:
        return []

    candidates = []

    # Capture quoted phrases (supports standard and smart quotes).
    quote_pattern = r'[""'']([^""'']+)[""'']'
    for match in re.findall(quote_pattern, claim_text):
        cleaned = match.strip()
        cleaned = re.sub(r"\bfeature(s)?\b", "", cleaned, flags=re.IGNORECASE).strip()
        if cleaned and not re.match(r"feature_\d+", cleaned, re.IGNORECASE):
            candidates.append(cleaned)

    # Capture phrases introduced by connectors when not quoted.
    relation_patterns = [r"related to ([^.;,]+)", r"focused on ([^.;,]+)", r"tied to ([^.;,]+)"]
    for pattern in relation_patterns:
        for segment in re.findall(pattern, claim_text, flags=re.IGNORECASE):
            parts = re.split(r"\band\b", segment, flags=re.IGNORECASE)
            for part in parts:
                cleaned = part.strip(" .")
                cleaned = re.sub(r"\bfeature(s)?\b", "", cleaned, flags=re.IGNORECASE).strip()
                if cleaned and not re.match(r"feature_\d+", cleaned, re.IGNORECASE):
                    candidates.append(cleaned)

    unique_candidates = []
    seen = set()
    for cand in candidates:
        key = cand.lower()
        if key not in seen:
            seen.add(key)
            unique_candidates.append(cand)

    return unique_candidates


def _ensure_causal_claim_feature_lists(claims):
    # Ensure causal claims have feature lists.
    for claim in claims or []:
        if claim.get('claim_type') != 'causal_claim':
            continue

        details = claim.get('details') or {}
        if not isinstance(details, dict):
            details = {}

        relationship = (details.get('relationship') or '').lower()
        if relationship not in {'upstream', 'downstream'}:
            continue

        key = 'source_feature_interpretations' if relationship == 'upstream' else 'target_feature_interpretations'
        existing = details.get(key)
        if isinstance(existing, list) and existing:
            continue

        extracted = _extract_interpretations_from_text(claim.get('claim_text', ''))
        if extracted:
            details[key] = extracted
            claim['details'] = details


def _stringify_summary(summary):
    if summary is None:
        return ""
    if isinstance(summary, str):
        return summary
    if isinstance(summary, dict):
        return " ".join(str(v) for v in summary.values() if v)
    if isinstance(summary, (list, tuple, set)):
        return " ".join(filter(None, (_stringify_summary(item) for item in summary)))
    return str(summary)


@st.cache_data(persist=True)
def _cached_extract_circuit_claims(api_config, explanation_text, context, cache_version="faithfulness-2025-11-29"):
    # Extract verifiable claims from AI explanation.
    headers = {
        "Authorization": f"Bearer {api_config['api_key']}",
        "Content-Type": "application/json"
    }
    
    # Process paragraph by paragraph for main circuit graph.
    if context == "circuit_graph":
        paragraphs = re.split(r'(?=####\s)', explanation_text.strip())
        all_claims = []

        for paragraph in paragraphs:
            if not paragraph.strip():
                continue

            # Prompt building logic.
            claim_types_details = tr('circuit_graph_claim_types')
            rules = tr('claim_extraction_prompt_rule')
            
            claim_extraction_prompt = f"""{tr('claim_extraction_prompt_header')}

{tr('claim_extraction_prompt_instruction')}

{rules}

{tr('claim_extraction_prompt_context_header').format(analysis_method=context)}
{tr('claim_extraction_prompt_types_header')}
{claim_types_details}

{tr('claim_extraction_prompt_analyze_header')}
"{paragraph}"

{tr('claim_extraction_prompt_footer')}
"""
            data = {"model": "qwen2.5-vl-72b-instruct", "messages": [{"role": "user", "content": claim_extraction_prompt}], "temperature": 0.0}
            
            try:
                response = requests.post(f"{api_config['api_endpoint']}/chat/completions", headers=headers, json=data, timeout=60)
                response.raise_for_status()
                claims_text = response.json()["choices"][0]["message"]["content"]
                
                if '```json' in claims_text:
                    claims_text = re.search(r'```json\n(.*?)\n```', claims_text, re.DOTALL).group(1)
                
                claims_from_paragraph = json.loads(claims_text)
                _ensure_causal_claim_feature_lists(claims_from_paragraph)
                if claims_from_paragraph:
                    all_claims.extend(claims_from_paragraph)
            except (requests.RequestException, AttributeError, json.JSONDecodeError):
                continue # If a paragraph fails, continue to the next.
        
        _ensure_causal_claim_feature_lists(all_claims)
        return all_claims

    # Original logic for other, shorter contexts
    rules = tr('claim_extraction_prompt_rule')
    if context == "feature_explorer":
        claim_types_details = tr('feature_explorer_claim_types')
        rules += "\n3. **Group related sentences:** If a sentence states a factual observation (e.g., lists top activating tokens) and the immediately following sentence provides reasoning or an explanation for that observation, you MUST extract them as a single claim, combining their text."
    elif context == "subnetwork_graph":
        claim_types_details = tr('subnetwork_graph_claim_types')
        rules += "\n6. **For causal claims in the subnetwork context,** you MUST populate the `source_feature_interpretations` or `target_feature_interpretations` arrays with every feature interpretation referenced in the claim. Use the exact phrasing from the explanation whenever possible."
    else: # Should not happen, but as a fallback
        return []

    claim_extraction_prompt = f"""{tr('claim_extraction_prompt_header')}

{tr('claim_extraction_prompt_instruction')}

{rules}

{tr('claim_extraction_prompt_context_header').format(analysis_method=context)}
{tr('claim_extraction_prompt_types_header')}
{claim_types_details}

{tr('claim_extraction_prompt_analyze_header')}
"{explanation_text}"

{tr('claim_extraction_prompt_footer')}
"""
    
    data = {"model": "qwen2.5-vl-72b-instruct", "messages": [{"role": "user", "content": claim_extraction_prompt}], "temperature": 0.0}
    
    try:
        response = requests.post(f"{api_config['api_endpoint']}/chat/completions", headers=headers, json=data, timeout=60)
        response.raise_for_status()
        claims_text = response.json()["choices"][0]["message"]["content"]
    
        if '```json' in claims_text:
            claims_text = re.search(r'```json\n(.*?)\n```', claims_text, re.DOTALL).group(1)
        claims = json.loads(claims_text)
        _ensure_causal_claim_feature_lists(claims)
        return claims
    except (requests.RequestException, AttributeError, json.JSONDecodeError):
        return []

@st.cache_data(persist=True)
def _cached_verify_semantic_summary(api_config, claimed_summary, actual_data_points, layer_section, cache_version="faithfulness-2025-11-29"):
    # Verify summary faithfulness via LLM.
    headers = {"Authorization": f"Bearer {api_config['api_key']}", "Content-Type": "application/json"}
    
    # Get principle for layer section.
    principles = {
        "early": tr('semantic_verification_principle_early'),
        "middle": tr('semantic_verification_principle_middle'),
        "late": tr('semantic_verification_principle_late'),
    }
    principle = principles.get(layer_section, "No specific principle defined for this section.")

    # A more forceful, hardcoded rule to prevent incorrect contradictions on valid generalizations.
    rule_3_override = "3.  **A claim is valid even if it only describes one aspect of a layer's function.** The summary does not need to be comprehensive. As long as the aspect it describes is supported by the data or the general principles, you MUST verify it. You MUST NOT contradict a claim because it 'does not fully capture' all functions of the layer."

    section_synonym_guidance = {
        "early": "You MUST treat descriptions such as 'dissecting the input', 'breaking the sentence into fundamental components', 'token breakdown', or 'parsing basic structure' as equivalent to handling syntax, grammar, and basic patterns.",
        "middle": "You MUST treat descriptions referring to linking recognitions into themes, building relationships, developing context, or combining earlier syntax with more complex constructs as equivalent to developing thematic connections and abstract meaning. You MUST accept statements about gaining a nuanced understanding or moving toward higher-level abstractions as faithful summaries for middle layers, even if the listed features mention programming syntax.",
        "late": "You MUST treat descriptions about synthesizing information, finalizing outputs, or producing coherent answers as equivalent to the late layers' role of integrating all information to finalize the output.",
    }
    synonym_guidance = section_synonym_guidance.get(layer_section, "")

    prompt = f"""{tr('semantic_verification_prompt_header')}

{tr('semantic_verification_prompt_rules_header')}
{tr('semantic_verification_prompt_rule_1')}
{tr('semantic_verification_prompt_rule_2').format(layer_section=layer_section, principle=principle)}
{rule_3_override}
{synonym_guidance}

{tr('semantic_verification_prompt_actual_data_header')}
{actual_data_points}

{tr('semantic_verification_prompt_claimed_summary_header')}
"{claimed_summary}"

{tr('semantic_verification_prompt_task_header')}
{tr('semantic_verification_prompt_task_instruction')}
"""
    
    data = {"model": "qwen2.5-vl-72b-instruct", "messages": [{"role": "user", "content": prompt}], "temperature": 0.0, "response_format": {"type": "json_object"}}
    
    response = requests.post(f"{api_config['api_endpoint']}/chat/completions", headers=headers, json=data, timeout=60)
    response.raise_for_status()
    
    try:
        return json.loads(response.json()["choices"][0]["message"]["content"])
    except (json.JSONDecodeError, KeyError):
        return {"is_verified": False, "reasoning": "Could not parse semantic verification result."}

@st.cache_data(persist=True)
def _cached_verify_feature_role_claim(api_config, claimed_role, feature_data, layer_name, neighbor_info=None, cache_version="faithfulness-2025-11-29"):
    # Verify feature role claim via LLM.
    headers = {"Authorization": f"Bearer {api_config['api_key']}", "Content-Type": "application/json"}
    
    # Prepare evidence from feature data.
    interpretation = feature_data.get('interpretation', 'N/A')
    top_tokens = [act['token'] for act in feature_data.get('top_activations', [])[:5]]
    
    try:
        layer_index = int(layer_name.split('_')[1])
        if layer_index <= 10: layer_pos = "early"
        elif 11 <= layer_index <= 21: layer_pos = "middle"
        else: layer_pos = "late"
    except (IndexError, ValueError):
        layer_pos = "unknown"

    layer_guidance_map = {
        "early": tr('semantic_verification_prompt_feature_role_guidance_early'),
        "middle": tr('semantic_verification_prompt_feature_role_guidance_middle'),
        "late": tr('semantic_verification_prompt_feature_role_guidance_late'),
    }
    layer_guidance = layer_guidance_map.get(layer_pos, "")

    feature_evidence = f"""
- **Feature Interpretation:** "{interpretation}"
- **Top Activating Tokens:** {top_tokens}
- **Layer Position:** {layer_pos} ({layer_name})
"""
    
    # Add neighbor info if available
    if neighbor_info:
        if neighbor_info.get('upstream'):
            feature_evidence += "\n" + tr('semantic_verification_prompt_feature_role_upstream_header').format(interpretations=neighbor_info['upstream'])
        if neighbor_info.get('downstream'):
            feature_evidence += "\n" + tr('semantic_verification_prompt_feature_role_downstream_header').format(interpretations=neighbor_info['downstream'])

    rule_3 = tr('semantic_verification_prompt_feature_role_rule_3') if neighbor_info else ""

    prompt = f"""{tr('semantic_verification_prompt_feature_role_header')}

{tr('semantic_verification_prompt_feature_role_rules_header')}
{tr('semantic_verification_prompt_feature_role_rule_1')}
{tr('semantic_verification_prompt_feature_role_rule_2')}
{layer_guidance}
{rule_3}

{tr('semantic_verification_prompt_feature_role_evidence_header')}
{feature_evidence}

{tr('semantic_verification_prompt_feature_role_claimed_role_header')}
"{claimed_role}"

{tr('semantic_verification_prompt_task_header')}
{tr('semantic_verification_prompt_task_instruction')}
"""
    
    data = {"model": "qwen2.5-vl-72b-instruct", "messages": [{"role": "user", "content": prompt}], "temperature": 0.0, "response_format": {"type": "json_object"}}
    
    response = requests.post(f"{api_config['api_endpoint']}/chat/completions", headers=headers, json=data, timeout=60)
    response.raise_for_status()
    
    try:
        return json.loads(response.json()["choices"][0]["message"]["content"])
    except (json.JSONDecodeError, KeyError):
        return {"is_verified": False, "reasoning": "Could not parse semantic verification result."}

@st.cache_data(persist=True)
def _cached_verify_subnetwork_purpose(api_config, claimed_purpose, actual_data_points, cache_version="faithfulness-2025-11-29"):
    # Verify subnetwork purpose via LLM.
    headers = {"Authorization": f"Bearer {api_config['api_key']}", "Content-Type": "application/json"}
    
    prompt = f"""{tr('semantic_verification_prompt_subnetwork_header')}

{tr('semantic_verification_prompt_subnetwork_rules_header')}
{tr('semantic_verification_prompt_subnetwork_rule_1')}
{tr('semantic_verification_prompt_subnetwork_rule_2')}

{tr('semantic_verification_prompt_subnetwork_actual_data_header')}
{actual_data_points}

{tr('semantic_verification_prompt_subnetwork_claimed_purpose_header')}
"{claimed_purpose}"

{tr('semantic_verification_prompt_task_header')}
{tr('semantic_verification_prompt_task_instruction')}
"""
    
    data = {"model": "qwen2.5-vl-72b-instruct", "messages": [{"role": "user", "content": prompt}], "temperature": 0.0, "response_format": {"type": "json_object"}}
    
    response = requests.post(f"{api_config['api_endpoint']}/chat/completions", headers=headers, json=data, timeout=60)
    response.raise_for_status()
    
    try:
        return json.loads(response.json()["choices"][0]["message"]["content"])
    except (json.JSONDecodeError, KeyError):
        return {"is_verified": False, "reasoning": "Could not parse semantic verification result."}

@st.cache_data(persist=True)
def _cached_verify_token_reasoning(api_config, claimed_explanation, feature_data, layer_name, cache_version="faithfulness-2025-11-29"):
    # Verify token activation reasoning via LLM.
    headers = {"Authorization": f"Bearer {api_config['api_key']}", "Content-Type": "application/json"}
    
    interpretation = feature_data.get('interpretation', 'N/A')
    top_tokens = [act['token'] for act in feature_data.get('top_activations', [])[:5]]
    
    try:
        layer_index = int(layer_name.split('_')[1])
        if layer_index <= 10: layer_pos = "early"
        elif 11 <= layer_index <= 21: layer_pos = "middle"
        else: layer_pos = "late"
    except (IndexError, ValueError):
        layer_pos = "unknown"

    feature_evidence = f"""
- **Feature Interpretation:** "{interpretation}"
- **Top Activating Tokens:** {top_tokens}
- **Layer Position:** {layer_pos} ({layer_name})
"""

    prompt = f"""{tr('semantic_verification_prompt_token_reasoning_header')}

{tr('semantic_verification_prompt_token_reasoning_rules_header')}
{tr('semantic_verification_prompt_token_reasoning_rule_1')}
{tr('semantic_verification_prompt_token_reasoning_rule_2')}

{tr('semantic_verification_prompt_token_reasoning_evidence_header')}
{feature_evidence}

{tr('semantic_verification_prompt_token_reasoning_claimed_explanation_header')}
"{claimed_explanation}"

{tr('semantic_verification_prompt_task_header')}
{tr('semantic_verification_prompt_task_instruction')}
"""
    
    data = {"model": "qwen2.5-vl-72b-instruct", "messages": [{"role": "user", "content": prompt}], "temperature": 0.0, "response_format": {"type": "json_object"}}
    
    response = requests.post(f"{api_config['api_endpoint']}/chat/completions", headers=headers, json=data, timeout=60)
    response.raise_for_status()
    
    try:
        return json.loads(response.json()["choices"][0]["message"]["content"])
    except (json.JSONDecodeError, KeyError):
        return {"is_verified": False, "reasoning": "Could not parse semantic verification result."}

@st.cache_data(persist=True)
def _cached_verify_causal_reasoning(api_config, claimed_explanation, source_interpretations, target_interpretations, central_feature_info=None, cache_version="faithfulness-2025-11-29"):
    # Verify causal reasoning via LLM.
    headers = {"Authorization": f"Bearer {api_config['api_key']}", "Content-Type": "application/json"}
    
    central_feature_context = ""
    if central_feature_info:
        central_feature_context = f"\n- **Central Feature Context:** {central_feature_info}"

    causal_evidence = f"""
- **Source Feature(s) Interpretations:** {source_interpretations}
- **Target Feature(s) Interpretations:** {target_interpretations}{central_feature_context}
"""

    prompt = f"""{tr('semantic_verification_prompt_causal_reasoning_header')}

{tr('semantic_verification_prompt_causal_reasoning_rules_header')}
{tr('semantic_verification_prompt_causal_reasoning_rule_1')}
{tr('semantic_verification_prompt_causal_reasoning_rule_2')}

{tr('semantic_verification_prompt_causal_reasoning_evidence_header')}
{causal_evidence}

{tr('semantic_verification_prompt_causal_reasoning_claimed_explanation_header')}
"{claimed_explanation}"

{tr('semantic_verification_prompt_task_header')}
{tr('semantic_verification_prompt_task_instruction')}
"""
    
    data = {"model": "qwen2.5-vl-72b-instruct", "messages": [{"role": "user", "content": prompt}], "temperature": 0.0, "response_format": {"type": "json_object"}}
    
    response = requests.post(f"{api_config['api_endpoint']}/chat/completions", headers=headers, json=data, timeout=60)
    response.raise_for_status()
    
    try:
        return json.loads(response.json()["choices"][0]["message"]["content"])
    except (json.JSONDecodeError, KeyError):
        return {"is_verified": False, "reasoning": "Could not parse semantic verification result."}

# --- End Faithfulness Verification ---


def format_tokens_for_display(tokens):
    # Converts tokenizer-style tokens into a human-readable comma-separated list.
    display_tokens = []
    for token in tokens or []:
        if token is None:
            continue

        cleaned = str(token)
        cleaned = cleaned.replace("Ġ", " ")
        cleaned = cleaned.replace("Ċ", "\n")
        cleaned = cleaned.replace("▁", " ")
        cleaned = cleaned.replace("\u0120", " ")
        cleaned = cleaned.replace("\u010a", "\n")

        if "\n" in cleaned:
            cleaned = cleaned.replace("\n", "\\n")

        cleaned = " ".join(cleaned.split())
        cleaned = cleaned.strip()

        if not cleaned:
            continue

        display_tokens.append(cleaned)

    if not display_tokens:
        return ""

    return ", ".join(f'"{tok}"' for tok in display_tokens)


def _normalize_token_core(token):
    if token is None:
        return ""
    text = str(token)
    replacements = [
        ("Ġ", " "),
        ("\u0120", " "),
        ("▁", " "),
        ("Ċ", "\n"),
        ("\u010a", "\n"),
    ]
    for old, new in replacements:
        text = text.replace(old, new)
    text = text.replace("\n", " ")
    
    # Be careful not to strip the token away if it IS a quote
    cleaned = text.strip(" \"'")
    if not cleaned and text.strip():
        # The token consists entirely of chars being stripped (e.g. ' or " or " "')
        # but isn't just whitespace. Preserve it.
        text = text.strip()
    else:
        text = cleaned
        
    text = " ".join(text.split())
    return text.lower()


def _normalize_actual_tokens(tokens):
    normalized = set()
    for token in tokens or []:
        base = _normalize_token_core(token)
        if not base:
            continue
        normalized.add(base)
        condensed = base.replace(" ", "")
        if condensed:
            normalized.add(condensed)
            normalized.add(f"g{condensed}")
        normalized.add(f"g{base}")

        for char in base:
            if not char.strip():
                continue
            normalized.add(char.lower())

    return normalized


_INTERPRETATION_STOPWORDS = {
    "a", "an", "the", "another", "additional", "extra", "other", "more",
    "feature", "features"
}


def _clean_interpretation_text(text):
    if not text:
        return ""
    cleaned = str(text).strip()
    if cleaned.lower().startswith("identifying "):
        cleaned = cleaned[12:]
    return cleaned.strip()


def _normalize_interpretation_text(text):
    if text is None:
        return ""
    cleaned = _clean_interpretation_text(text)
    cleaned = cleaned.replace("-", " ")
    cleaned = re.sub(r'[\"“”\'‘’]', '', cleaned.lower())
    cleaned = re.sub(r'\b(' + "|".join(_INTERPRETATION_STOPWORDS) + r')\b', ' ', cleaned)
    cleaned = re.sub(r'\s+', ' ', cleaned).strip()
    return cleaned


def _prepare_feature_interpretations(features):
    formatted = []
    fuzzy_candidates = []
    normalized_variants = set()

    seen_formatted = set()
    seen_candidates = set()

    for feature in features or []:
        layer = feature.get('layer', 'N/A')
        feature_name = feature.get('feature_name', 'Unknown feature')
        interpretation_raw = feature.get('interpretation', 'N/A')
        interpretation_clean = _clean_interpretation_text(interpretation_raw) or interpretation_raw

        formatted_str = f"L{layer}: {feature_name} ('{interpretation_clean}')"
        if formatted_str not in seen_formatted:
            formatted.append(formatted_str)
            seen_formatted.add(formatted_str)

        candidate_texts = {
            interpretation_raw,
            interpretation_clean,
            feature_name,
            feature_name.replace('_', ' ') if feature_name else "",
            f"{feature_name} {interpretation_clean}" if feature_name and interpretation_clean else ""
        }

        for candidate in candidate_texts:
            if not candidate:
                continue
            candidate = candidate.strip()
            if not candidate:
                continue
            if candidate not in seen_candidates:
                fuzzy_candidates.append(candidate)
                seen_candidates.add(candidate)
            normalized = _normalize_interpretation_text(candidate)
            if normalized:
                normalized_variants.add(normalized)

    return formatted, fuzzy_candidates, normalized_variants


def _token_variants_for_match(token):
    variants = set()
    base = _normalize_token_core(token)
    if not base:
        return variants
    variants.add(base)
    condensed = base.replace(" ", "")
    if condensed and condensed != base:
        variants.add(condensed)
    if base.startswith("g") and len(base) > 1:
        variants.add(base[1:])
    return variants


def _token_matches_actual(token, normalized_set, normalized_list):
    variants = _token_variants_for_match(token)
    for var in variants:
        if var in normalized_set:
            return True
        for actual in normalized_list:
            if not actual:
                continue
            if var == actual:
                return True
            if var in actual or actual in var:
                return True
    return False


def get_circuit_explanation(api_config, fig, analysis_data, visualization_type="circuit_graph"):
    # Prepares data and calls the cached explanation function.
    try:
        # Convert the Plotly figure to an image.
        img_bytes = fig.to_image(format="png", width=1200, height=800)
        img_base64 = base64.b64encode(img_bytes).decode()
        
        # Get the current language from the session state.
        lang = st.session_state.get('lang', 'en')

        # Prepare context and instructions based on the visualization type.
        if visualization_type == "circuit_graph":
            context = prepare_circuit_graph_context(analysis_data)
            instruction = (
                f"{tr('circuit_graph_instruction_header')}\n\n"
                f"{tr('circuit_graph_instruction_intro')}\n\n"
                f"{tr('circuit_graph_instruction_early')}\n\n"
                f"{tr('circuit_graph_instruction_middle')}\n\n"
                f"{tr('circuit_graph_instruction_late')}\n\n"
                f"{tr('circuit_graph_instruction_insight')}\n\n"
                f"{tr('circuit_graph_instruction_footer')}"
            )
            max_tokens_for_request = 1200
        elif visualization_type == "feature_explorer":
            context = prepare_feature_explorer_context(analysis_data)
            instruction = (
                f"{tr('feature_explorer_instruction_header')}\n\n"
                f"{tr('feature_explorer_instruction_role')}\n"
                f"{tr('feature_explorer_instruction_activations')}\n"
                f"{tr('feature_explorer_instruction_insight')}\n\n"
                f"{tr('feature_explorer_instruction_footer')}"
            )
            max_tokens_for_request = 400
        elif visualization_type == "subnetwork_graph":
            context = prepare_subnetwork_context(analysis_data)
            instruction = (
                f"{tr('subnetwork_graph_instruction_header')}\n\n"
                f"{tr('subnetwork_graph_instruction_role')}\n"
                f"{tr('subnetwork_graph_instruction_upstream')}\n"
                f"{tr('subnetwork_graph_instruction_downstream')}\n"
                f"{tr('subnetwork_graph_instruction_purpose')}\n\n"
                f"{tr('subnetwork_graph_instruction_footer')}"
            )
            max_tokens_for_request = 500
        else:
            context = tr('context_unspecified_viz')
            instruction = tr('instruction_unspecified_viz')
            max_tokens_for_request = 750
        
        structured_prompt = f"""{tr('explanation_prompt_header')}

{tr('explanation_prompt_context_header')}
{context}

{tr('explanation_prompt_instructions_header')}
{instruction}
"""
        
        explanation = explain_circuit_visualization(
            api_config,
            img_base64,
            structured_prompt,
            max_tokens_for_request
        )

        if "API request failed" in explanation or "Error generating explanation" in explanation:
            st.error(explanation)
            return "Unable to generate explanation."
        
        return explanation
        
    except Exception as e:
        st.error(f"Error preparing for explanation: {str(e)}")
        return "Unable to generate explanation."


def prepare_circuit_graph_context(analysis_data):
    # Prepares the context for the circuit graph explanation.
    prompt = analysis_data.get('prompt', 'Unknown prompt')
    input_tokens = analysis_data.get('input_tokens', [])
    layer_summaries = analysis_data.get('layer_summaries', {})
    
    lang = st.session_state.get('lang', 'en')

    # Prepare the layer summary context.
    summary_context = ""
    if layer_summaries:
        early_summary = "\n".join([
            tr('circuit_graph_context_feature_line').format(
                layer=f['layer'],
                interpretation=f['interpretation'],
                activation=f['activation']
            ) for f in layer_summaries.get('early', [])
        ])
        middle_summary = "\n".join([
            tr('circuit_graph_context_feature_line').format(
                layer=f['layer'],
                interpretation=f['interpretation'],
                activation=f['activation']
            ) for f in layer_summaries.get('middle', [])
        ])
        late_summary = "\n".join([
            tr('circuit_graph_context_feature_line').format(
                layer=f['layer'],
                interpretation=f['interpretation'],
                activation=f['activation']
            ) for f in layer_summaries.get('late', [])
        ])

        summary_context = f"""
{tr('circuit_graph_context_summary_header')}

{tr('circuit_graph_context_early_header')}
{early_summary if early_summary else tr('circuit_graph_context_no_features')}

{tr('circuit_graph_context_middle_header')}
{middle_summary if middle_summary else tr('circuit_graph_context_no_features')}

{tr('circuit_graph_context_late_header')}
{late_summary if late_summary else tr('circuit_graph_context_no_features')}
"""

    tokens_display = format_tokens_for_display(input_tokens) or ' '.join(input_tokens)

    return f"""
{tr('circuit_graph_context_header').format(prompt=prompt)}
{tr('circuit_graph_context_tokens').format(tokens=tokens_display)}
{summary_context}
"""

def prepare_subnetwork_context(analysis_data):
    # Prepares the context for the subnetwork graph explanation.
    prompt = analysis_data.get('prompt', 'Unknown prompt')
    central_feature_info = analysis_data.get('central_feature_info', {})
    subgraph_stats = analysis_data.get('subgraph_stats', {})
    subgraph_neighbors = analysis_data.get('subgraph_neighbors', {})
    
    lang = st.session_state.get('lang', 'en')

    # Prepare the context for neighboring features.
    upstream_features = subgraph_neighbors.get('upstream', [])
    downstream_features = subgraph_neighbors.get('downstream', [])
    
    upstream_context = ""
    if upstream_features:
        header = tr('subnetwork_context_upstream_header')
        feature_lines = [
            tr('subnetwork_context_feature_line').format(
                layer=feat.get('layer'),
                feature_name=feat.get('feature_name'),
                interpretation=feat.get('interpretation', 'N/A')
            ) for feat in upstream_features[:5]
        ]
        upstream_context = header + "\n" + "\n".join(feature_lines)

    downstream_context = ""
    if downstream_features:
        header = tr('subnetwork_context_downstream_header')
        feature_lines = [
            tr('subnetwork_context_feature_line').format(
                layer=feat.get('layer'),
                feature_name=feat.get('feature_name'),
                interpretation=feat.get('interpretation', 'N/A')
            ) for feat in downstream_features[:5]
        ]
        downstream_context = header + "\n" + "\n".join(feature_lines)

    central_interpretation = central_feature_info.get('interpretation')
    if not central_interpretation:
        central_interpretation = tr('subnetwork_context_no_interpretation')

    return f"""
{tr('subnetwork_context_header').format(prompt=prompt)}

{tr('subnetwork_context_centered_on')}
{tr('subnetwork_context_feature').format(name=central_feature_info.get('name', 'Unknown'))}
{tr('subnetwork_context_layer').format(layer=central_feature_info.get('layer', 'Unknown'))}
{tr('subnetwork_context_interpretation').format(interpretation=central_interpretation)}

{upstream_context}
{downstream_context}

{tr('subnetwork_context_depth').format(depth=analysis_data.get('depth', 'N/A'))}

{tr('subnetwork_context_stats_header')}
{tr('subnetwork_context_stats_nodes').format(nodes=subgraph_stats.get('nodes', 0))}
{tr('subnetwork_context_stats_edges').format(edges=subgraph_stats.get('edges', 0))}

{tr('subnetwork_context_viz_header')}
{tr('subnetwork_context_viz_central')}
{tr('subnetwork_context_viz_nodes')}
{tr('subnetwork_context_viz_lilac')}
{tr('subnetwork_context_viz_other')}
{tr('subnetwork_context_viz_edges')}
"""

def prepare_feature_explorer_context(analysis_data):
    # Prepares the context for the feature explorer explanation.
    prompt = analysis_data.get('prompt', 'Unknown prompt')
    input_tokens = analysis_data.get('input_tokens', [])
    selected_layer_str = analysis_data.get('selected_layer', 'layer_unknown')
    selected_feature = analysis_data.get('selected_feature', 'Unknown Feature')

    lang = st.session_state.get('lang', 'en')

    try:
        layer_index = int(selected_layer_str.split('_')[1])
        if layer_index <= 10:
            layer_position_desc = tr('feature_explorer_context_position_early')
        elif 11 <= layer_index <= 21:
            layer_position_desc = tr('feature_explorer_context_position_middle')
        else:
            layer_position_desc = tr('feature_explorer_context_position_late')
        layer_context = tr('feature_explorer_context_analyzing_feature').format(
            feature=selected_feature,
            layer=layer_index,
            position=layer_position_desc
        )
    except (IndexError, ValueError):
        layer_context = tr('feature_explorer_context_analyzing_feature_no_pos').format(
            feature=selected_feature,
            layer=selected_layer_str
        )

    # Safely get the feature data.
    feature_data = analysis_data.get('feature_visualizations', {}).get(selected_layer_str, {}).get(selected_feature, {})
    interpretation = feature_data.get('interpretation', tr('feature_explorer_context_no_interpretation'))
    
    tokens_display = format_tokens_for_display(input_tokens) or ' '.join(input_tokens)

    return f"""
{tr('feature_explorer_context_header').format(prompt=prompt)}

{tr('feature_explorer_context_model_header')}

{layer_context}

{tr('feature_explorer_context_tokens').format(tokens=tokens_display)}
{tr('feature_explorer_context_interpretation').format(interpretation=interpretation)}

{tr('feature_explorer_context_footer')}
"""




@st.cache_data
def load_attribution_results(lang='en'):
    # Load attribution results for selected language.
    # Get file path by language.
    if lang == 'de':
        file_path = Path(__file__).parent / 'results/attribution_graphs_results_de.json'
    else:
        file_path = Path(__file__).parent / 'results/attribution_graphs_results.json'

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        st.error(tr('no_results_warning'))
        # Specific error message.
        if lang == 'de':
            st.info("Bitte führen Sie zuerst die deutsche Analyse aus: `python3 circuit_analysis/attribution_graphs_olmo_de.py --prompt-index 0 --force-retrain-clt`")
        else:
            st.info(tr('run_analysis_info'))
        return None
    except json.JSONDecodeError:
        st.error(f"Error: Invalid JSON in {file_path}. The file might be corrupted or empty.")
        return None

def create_interactive_feature_explorer(analysis, prompt_idx, enable_explanations=False, qwen_api_config=None):
    # Interactive feature explorer.
    st.subheader(tr('feature_explorer_title').format(prompt=analysis['prompt']))
    
    # Layer selection.
    available_layers = list(analysis['feature_visualizations'].keys())
    if not available_layers:
        st.warning(tr('no_feature_viz_warning'))
        return None
    
    selected_layer = st.selectbox(
        tr('select_layer_label'),
        available_layers,
        format_func=lambda x: tr('layer_label_format').format(layer_num=x.split('_')[1])
    )
    
    layer_features = analysis['feature_visualizations'][selected_layer]
    
    if not layer_features:
        st.warning(tr('no_features_in_layer_warning').format(selected_layer=selected_layer))
        return None
    
    # Layout columns.
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.write(tr('active_features_label'))
        feature_options = list(layer_features.keys())
        selected_feature = st.selectbox(tr('choose_feature_label'), feature_options)
        
        if selected_feature:
            feat_data = layer_features[selected_feature]
            
            # Feature statistics.
            st.metric(tr('max_activation_label'), f"{feat_data['max_activation']:.3f}")
            st.metric(tr('mean_activation_label'), f"{feat_data['mean_activation']:.3f}")
            st.metric(tr('sparsity_label'), f"{feat_data['sparsity']:.3f}")
            
            # Feature interpretation.
            interpretation = feat_data.get('interpretation', 'N/A')
            if interpretation.startswith("Identifying "):
                interpretation = interpretation[12:]
            
            st.info(f"**{tr('interpretation_label')}:** {interpretation}")
    
    with col2:
        if selected_feature:
            feat_data = layer_features[selected_feature]
            
            # Activation pattern chart.
            if 'top_activations' in feat_data:
                activation_data = []
                for item in feat_data['top_activations']:
                    activation_data.append({
                        'token': item['token'],
                        'position': item['position'],
                        'activation': item['activation']
                    })
                
                if activation_data:
                    df = pd.DataFrame(activation_data)
                    
                    fig = px.bar(
                        df, 
                        x='token', 
                        y='activation',
                        title=tr('top_activating_tokens_title').format(selected_feature=selected_feature),
                        hover_data=['position'],
                        color='activation',
                        color_continuous_scale='viridis'
                    )
                    fig.update_layout(
                        xaxis_title=tr('xaxis_token_label'),
                        yaxis_title=tr('yaxis_activation_label'),
                        showlegend=False
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Add an AI explanation if enabled.
                    if enable_explanations and qwen_api_config is not None:
                        cache_key = f"explanation_feature_explorer_{prompt_idx}_{selected_layer}_{selected_feature}"
                        
                        if cache_key not in st.session_state:
                            with st.spinner(tr('generating_feature_explanation_spinner')):
                                try:
                                    # Explanation context.
                                    analysis_with_context = analysis.copy()
                                    analysis_with_context['selected_layer'] = selected_layer
                                    analysis_with_context['selected_feature'] = selected_feature

                                    explanation = get_circuit_explanation(
                                        qwen_api_config, 
                                        fig, 
                                        analysis_with_context, 
                                        visualization_type="feature_explorer"
                                    )
                                    
                                    # Format explanation.
                                    processed_explanation = explanation.replace("- **", "\n- **")
                                    if not processed_explanation.strip().startswith('-'):
                                        processed_explanation = f"- {processed_explanation.strip()}"
                                    
                                    st.session_state[cache_key] = processed_explanation
                                    
                                except Exception as e:
                                    st.error(tr('feature_explanation_error').format(e=str(e)))
                                    st.session_state[cache_key] = "Error: Could not generate explanation."
                        
                        if st.session_state.get(cache_key) and "Error:" not in st.session_state[cache_key] and "Unable to generate" not in st.session_state[cache_key]:
                            st.markdown(tr('ai_feature_analysis_header'))
                            # Convert markdown to HTML.
                            html_explanation = markdown.markdown(st.session_state[cache_key])
                            st.markdown(f"""
                            <div style="background-color: #2b2b2b; color: #ffffff; padding: 1.2rem; border-radius: 10px; border-left: 4px solid #dcae36; font-size: 0.9rem; margin-bottom: 1rem;">
                                {html_explanation}
                            </div>
                            """, unsafe_allow_html=True)
                            
                            # Faithfulness Check.
                            with st.expander(tr('faithfulness_check_expander')):
                                st.markdown(tr('faithfulness_explanation_feature_explorer_html'), unsafe_allow_html=True)
                                with st.spinner(tr('running_faithfulness_check_spinner')):
                                    analysis_with_context = analysis.copy()
                                    analysis_with_context['selected_layer'] = selected_layer
                                    analysis_with_context['selected_feature'] = selected_feature
                                    
                                    claims = _cached_extract_circuit_claims(qwen_api_config, st.session_state[cache_key], "feature_explorer", cache_version="faithfulness-2025-11-29")
                                    verification_results = verify_circuit_claims(claims, analysis_with_context, "feature_explorer")
                                    
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
                    
                    return fig
    
    return None


def render_dataset_faithfulness_summary(summary: dict):
    if not summary:
        return
    
    table_rows = []
    for label, key in [
        ('Targeted', 'targeted'),
        ('Random baseline', 'random_baseline'),
        ('Path', 'path'),
        ('Random path baseline', 'random_path_baseline')
    ]:
        stats = summary.get(key, {}) or {}
        table_rows.append({
            'Type': label,
            'Count': stats.get('count', 0),
            'Avg |Δp|': stats.get('avg_abs_probability_change', 0.0),
            'Flip rate': stats.get('flip_rate', 0.0),
            'Avg |Δlogit|': stats.get('avg_abs_logit_change', 0.0)
        })
    
    df = pd.DataFrame(table_rows).set_index('Type')
    st.table(df.style.format({'Avg |Δp|': '{:.4f}', 'Flip rate': '{:.2%}', 'Avg |Δlogit|': '{:.4f}'}))
    
    diff_abs = summary.get('target_minus_random_abs_probability_change', 0.0)
    diff_flip = summary.get('target_flip_rate_minus_random', 0.0)
    path_diff_abs = summary.get('path_minus_random_abs_probability_change', 0.0)
    path_diff_flip = summary.get('path_flip_rate_minus_random', 0.0)
    st.caption(f"|Δp| difference (targeted − random): {diff_abs:.4f}")
    st.caption(f"Flip-rate difference (targeted − random): {diff_flip:.4f}")
    st.caption(f"|Δp| difference (path − random path): {path_diff_abs:.4f}")
    st.caption(f"Flip-rate difference (path − random path): {path_diff_flip:.4f}")


def render_faithfulness_metrics(analysis: dict, prompt_idx: int):
    summary_stats = analysis.get('summary_statistics')
    if not summary_stats:
        return
    
    targeted_summary = summary_stats.get('targeted', {}) or {}
    random_summary = summary_stats.get('random_baseline', {}) or {}
    path_summary = summary_stats.get('path', {}) or {}
    random_path_summary = summary_stats.get('random_path_baseline', {}) or {}
    
    summary_df = pd.DataFrame(
        [
            {
                'Type': 'Targeted',
                'Count': targeted_summary.get('count', 0),
                'Avg |Δp|': targeted_summary.get('avg_abs_probability_change', 0.0),
                'Flip rate': targeted_summary.get('flip_rate', 0.0),
                'Avg |Δlogit|': targeted_summary.get('avg_abs_logit_change', 0.0)
            },
            {
                'Type': 'Random baseline',
                'Count': random_summary.get('count', 0),
                'Avg |Δp|': random_summary.get('avg_abs_probability_change', 0.0),
                'Flip rate': random_summary.get('flip_rate', 0.0),
                'Avg |Δlogit|': random_summary.get('avg_abs_logit_change', 0.0)
            },
            {
                'Type': 'Path',
                'Count': path_summary.get('count', 0),
                'Avg |Δp|': path_summary.get('avg_abs_probability_change', 0.0),
                'Flip rate': path_summary.get('flip_rate', 0.0),
                'Avg |Δlogit|': path_summary.get('avg_abs_logit_change', 0.0)
            },
            {
                'Type': 'Random path baseline',
                'Count': random_path_summary.get('count', 0),
                'Avg |Δp|': random_path_summary.get('avg_abs_probability_change', 0.0),
                'Flip rate': random_path_summary.get('flip_rate', 0.0),
                'Avg |Δlogit|': random_path_summary.get('avg_abs_logit_change', 0.0)
            }
        ]
    ).set_index('Type')
    
    diff_abs = summary_stats.get('target_minus_random_abs_probability_change', 0.0)
    diff_flip = summary_stats.get('target_flip_rate_minus_random', 0.0)
    path_diff_abs = summary_stats.get('path_minus_random_abs_probability_change', 0.0)
    path_diff_flip = summary_stats.get('path_flip_rate_minus_random', 0.0)
    
    with st.expander("Faithfulness metrics", expanded=False):
        st.table(summary_df.style.format({'Avg |Δp|': '{:.4f}', 'Flip rate': '{:.2%}', 'Avg |Δlogit|': '{:.4f}'}))
        st.caption(f"|Δp| difference (targeted − random): {diff_abs:.4f}")
        st.caption(f"Flip-rate difference (targeted − random): {diff_flip:.4f}")
        st.caption(f"|Δp| difference (path − random path): {path_diff_abs:.4f}")
        st.caption(f"Flip-rate difference (path − random path): {path_diff_flip:.4f}")
        
        targeted_results = analysis.get('perturbation_experiments', []) or []
        random_results = analysis.get('random_baseline_experiments', []) or []
        comparison_rows = []
        if targeted_results:
            for exp in targeted_results:
                feature_set = exp.get('feature_set', []) or []
                feature_label = exp.get('feature_name')
                if not feature_label and feature_set:
                    feature_label = ", ".join(f"L{item.get('layer')}F{item.get('feature')}" for item in feature_set[:3])
                comparison_rows.append({
                    'Label': feature_label,
                    'Type': 'Targeted',
                    'Δp': exp.get('probability_change', 0.0),
                    '|Δp|': abs(exp.get('probability_change', 0.0)),
                    'Δlogit': exp.get('logit_change', 0.0),
                    '|Δlogit|': abs(exp.get('logit_change', 0.0)),
                    'Flips top': exp.get('ablation_flips_top_prediction', False),
                    'Interpretation': exp.get('feature_interpretation')
                })
        if random_results:
            for exp in random_results:
                comparison_rows.append({
                    'Label': f"Random {exp.get('trial_index')}",
                    'Type': 'Random',
                    'Δp': exp.get('probability_change', 0.0),
                    '|Δp|': abs(exp.get('probability_change', 0.0)),
                    'Δlogit': exp.get('logit_change', 0.0),
                    '|Δlogit|': abs(exp.get('logit_change', 0.0)),
                    'Flips top': exp.get('ablation_flips_top_prediction', False),
                    'Interpretation': None
                })
        if comparison_rows:
            comparison_df = pd.DataFrame(comparison_rows)
            numeric_cols = ['Δp', '|Δp|', 'Δlogit', '|Δlogit|']
            comparison_df[numeric_cols] = comparison_df[numeric_cols].apply(
                lambda col: col.map(lambda x: round(float(x), 4))
            )
            top_targeted = comparison_df[comparison_df['Type'] == 'Targeted'].nlargest(5, '|Δp|')
            display_df = pd.concat([top_targeted, comparison_df[comparison_df['Type'] == 'Random']], ignore_index=True)
            display_df = display_df.sort_values(['Type', '|Δp|'], ascending=[True, True])
            st.markdown("**Targeted vs Random feature ablations (|Δp|)**")
            st.plotly_chart(
                px.bar(
                    display_df,
                    x='|Δp|',
                    y='Label',
                    color='Type',
                    orientation='h',
                    hover_data=['Δp', 'Δlogit', 'Flips top', 'Interpretation'],
                    labels={'|Δp|': '|Δp|', 'Label': 'Feature / Trial'},
                    title=None
                ),
                use_container_width=True
            )

            path_results = analysis.get('path_ablation_experiments', []) or []
            if path_results:
                path_rows = []
                def _top_token(entries):
                    if not entries:
                        return None
                    first = entries[0]
                    if isinstance(first, (list, tuple)) and first:
                        return str(first[0]).strip()
                    return str(first).strip()
                for exp in path_results:
                    feature_set = exp.get('feature_set', []) or []
                    feature_label = ", ".join(f"L{item.get('layer')}F{item.get('feature')}" for item in feature_set)
                    flipped = exp.get('ablation_flips_top_prediction', False)
                    flip_detail = None
                    if flipped:
                        baseline_top = _top_token(exp.get('baseline_top_tokens', []))
                        ablated_top = _top_token(exp.get('ablated_top_tokens', []))
                        if baseline_top and ablated_top and baseline_top != ablated_top:
                            flip_detail = f"{baseline_top} → {ablated_top}"
                    path_rows.append({
                        'Description': exp.get('path_description', 'Path'),
                        'Features': feature_label,
                        'Δp': exp.get('probability_change', 0.0),
                        '|Δp|': abs(exp.get('probability_change', 0.0)),
                        'Δlogit': exp.get('logit_change', 0.0),
                        '|Δlogit|': abs(exp.get('logit_change', 0.0)),
                        'Flips top': flipped,
                        'Flip detail': flip_detail
                    })
                path_df = pd.DataFrame(path_rows)
                if not path_df.empty:
                    numeric_cols = ['Δp', '|Δp|', 'Δlogit', '|Δlogit|']
                    path_df[numeric_cols] = path_df[numeric_cols].apply(
                        lambda col: col.map(lambda x: round(float(x), 4))
                    )
                    st.markdown("**Path ablations (|Δp|)**")
                    avg_abs = path_df['|Δp|'].mean()
                    max_abs = path_df['|Δp|'].max()
                    flip_rate = path_df['Flips top'].mean()
                    st.caption(
                        f"Traced circuits ablated end-to-end. Average |Δp| = {avg_abs:.4f}, "
                        f"max |Δp| = {max_abs:.4f}, flip rate = {flip_rate:.2%}."
                    )
                    chip_style = (
                        "display:inline-flex; align-items:center; padding:0.25rem 0.55rem;"
                        "border-radius:999px; background-color:#2f2f2f; font-size:0.85rem;"
                        "font-weight:600; color:#f8fafc;"
                    )
                    arrow_html = "<span style='color:#94a3b8; margin:0 0.4rem;'>→</span>"
                    path_list = path_df.sort_values('|Δp|', ascending=False)
                    for _, row in path_list.iterrows():
                        accent = '#F97316' if row['Flips top'] else '#38BDF8'
                        description = str(row['Description']) if row['Description'] else 'Path'
                        segments = [
                            seg.strip() for seg in description.replace('->', '→').split('→') if seg.strip()
                        ]
                        if not segments:
                            segments = [description]
                        path_nodes_html = ""
                        for idx, segment in enumerate(segments):
                            path_nodes_html += f"<span style=\"{chip_style}\">{segment}</span>"
                            if idx < len(segments) - 1:
                                path_nodes_html += arrow_html
                        delta_p = row['|Δp|']
                        delta_logit = row['Δlogit']
                        flip_text = 'Prediction flipped' if row['Flips top'] else 'Prediction unchanged'
                        flip_detail = row.get('Flip detail')
                        feature_text = row['Features'] if row['Features'] else 'No internal features recorded'
                        st.markdown(
                            f"""
                            <div style="
                                padding:0.75rem 0.9rem;
                                border-radius:8px;
                                background-color:#1f2024;
                                border:1px solid {accent};
                                margin-bottom:0.6rem;
                            ">
                                <div style="display:flex; flex-wrap:wrap; align-items:center; gap:0.35rem;">
                                    {path_nodes_html}
                                </div>
                                <div style="margin-top:0.45rem; font-size:0.9rem;">
                                    <span style="font-weight:600; color:#f8fafc;">|Δp|</span>
                                    <span style="color:#f8fafc;">= {delta_p:.4f}</span>
                                    &nbsp;|&nbsp;
                                    <span style="font-weight:600; color:#f8fafc;">Δlogit</span>
                                    <span style="color:#f8fafc;">= {delta_logit:.4f}</span>
                                    &nbsp;|&nbsp;
                                    <span style="color:{accent}; font-weight:600;">{flip_text}</span>
                                    {"&nbsp;|&nbsp;<span style='color:#f8fafc;'>"+flip_detail+"</span>" if flip_detail else ""}
                                </div>
                                <div style="margin-top:0.25rem; font-size:0.8rem; color:#cbd5f5;">
                                    {feature_text}
                                </div>
                            </div>
                            """,
                            unsafe_allow_html=True
                        )

            random_path_results = analysis.get('random_path_baseline_experiments', []) or []
            if random_path_results:
                random_path_rows = []
                for exp in random_path_results:
                    feature_set = exp.get('feature_set', []) or []
                    feature_label = ", ".join(f"L{item.get('layer')}F{item.get('feature')}" for item in feature_set)
                    random_path_rows.append({
                        'Trial': exp.get('trial_index'),
                        'Sampled features': feature_label,
                        'Δp': exp.get('probability_change', 0.0),
                        '|Δp|': abs(exp.get('probability_change', 0.0)),
                        'Δlogit': exp.get('logit_change', 0.0),
                        '|Δlogit|': abs(exp.get('logit_change', 0.0)),
                        'Flips top': exp.get('ablation_flips_top_prediction', False)
                    })
                random_path_df = pd.DataFrame(random_path_rows)
                if not random_path_df.empty:
                    numeric_cols = ['Δp', '|Δp|', 'Δlogit', '|Δlogit|']
                    random_path_df[numeric_cols] = random_path_df[numeric_cols].apply(
                        lambda col: col.map(lambda x: round(float(x), 4))
                    )
                    st.markdown("<div style='height:0.75rem;'></div>", unsafe_allow_html=True)
                    st.markdown("**Random path baselines (|Δp|)**")
                    avg_abs = random_path_df['|Δp|'].mean()
                    max_abs = random_path_df['|Δp|'].max()
                    flip_rate = random_path_df['Flips top'].mean()
                    st.caption(
                        f"Randomly sampled paths from the same layer span. "
                        f"Average |Δp| = {avg_abs:.4f}, max |Δp| = {max_abs:.4f}, "
                        f"flip rate = {flip_rate:.2%}."
                    )
                    chip_style = (
                        "display:inline-flex; align-items:center; padding:0.25rem 0.55rem;"
                        "border-radius:999px; background-color:#2f2f2f; font-size:0.85rem;"
                        "font-weight:600; color:#f8fafc;"
                    )
                    arrow_html = "<span style='color:#94a3b8; margin:0 0.4rem;'>→</span>"
                    baseline_cards = random_path_df.sort_values('|Δp|', ascending=False)
                    for _, row in baseline_cards.iterrows():
                        accent = '#F97316' if row['Flips top'] else '#38BDF8'
                        feature_text = row['Sampled features'] if row['Sampled features'] else 'Randomly sampled features'
                        feature_tokens = [tok.strip() for tok in feature_text.split(',') if tok.strip()]
                        feature_nodes_html = ""
                        for idx, token in enumerate(feature_tokens):
                            feature_nodes_html += f"<span style=\"{chip_style}\">{token}</span>"
                            if idx < len(feature_tokens) - 1:
                                feature_nodes_html += arrow_html
                        delta_p = row['|Δp|']
                        delta_logit = row['Δlogit']
                        flip_text = 'Prediction flipped' if row['Flips top'] else 'Prediction unchanged'
                        st.markdown(
                            f"""
                            <div style="
                                padding:0.75rem 0.9rem;
                                border-radius:8px;
                                background-color:#1f2024;
                                border:1px solid {accent};
                                margin-bottom:0.6rem;
                            ">
                                <div style="display:flex; flex-wrap:wrap; align-items:center; gap:0.35rem;">
                                    {feature_nodes_html if feature_nodes_html else '<span style="color:#64748b;">No feature IDs logged</span>'}
                                </div>
                                <div style="margin-top:0.45rem; font-size:0.9rem;">
                                    <span style="font-weight:600; color:#f8fafc;">Random trial</span>
                                    <span style="color:#f8fafc;">#{int(row['Trial']) if pd.notnull(row['Trial']) else '-'}</span>
                                    &nbsp;|&nbsp;
                                    <span style="font-weight:600; color:#f8fafc;">|Δp|</span>
                                    <span style="color:#f8fafc;">= {delta_p:.4f}</span>
                                    &nbsp;|&nbsp;
                                    <span style="font-weight:600; color:#f8fafc;">Δlogit</span>
                                    <span style="color:#f8fafc;">= {delta_logit:.4f}</span>
                                    &nbsp;|&nbsp;
                                    <span style="color:{accent}; font-weight:600;">{flip_text}</span>
                                </div>
                            </div>
                            """,
                            unsafe_allow_html=True
                        )
def create_interactive_attribution_graph(analysis, prompt_idx, enable_explanations=False, qwen_api_config=None):
    # Create interactive attribution graph.
    
    # UI controls.
    col1, col2 = st.columns(2)
    
    with col1:
        node_size_factor = st.slider(tr('node_size_label'), 0.5, 3.0, 1.0, 0.1)
    with col2:
        edge_threshold = st.slider(tr('edge_threshold_label'), 0.0, 1.0, 0.1, 0.05)

    layer_spacing = 200

    # Simplified graph for visualization.
    G = nx.DiGraph()
    
    # Add nodes.
    node_positions = {}
    node_info = {}
    
    # Embedding nodes.
    tokens = analysis['input_tokens']
    num_tokens = len(tokens)
    for i, token in enumerate(tokens):
        node_id = f"emb_{i}_{token}"
        G.add_node(node_id)
        # Display tokens from top to bottom.
        node_positions[node_id] = (0, (num_tokens - 1 - i) * 50)
        node_info[node_id] = {
            'type': 'embedding',
            'token': token,
            'layer': -1,
            'activation': 1.0
        }
    
    # Add nodes.
    feature_count = 0
    max_layer = -1
    layer_feature_counts = {}
    
    for layer_name, layer_features in analysis['feature_visualizations'].items():
        layer_idx = int(layer_name.split('_')[1])
        max_layer = max(max_layer, layer_idx)
        layer_feature_counts[layer_idx] = 0
        
        for feat_idx, (feat_name, feat_data) in enumerate(layer_features.items()):
            if feat_data['max_activation'] > edge_threshold:
                node_id = f"feat_{layer_idx}_{feat_name}"
                G.add_node(node_id)
                
                # Position nodes.
                y_pos = feat_idx * 80 - len(layer_features) * 40
                node_positions[node_id] = ((layer_idx + 1) * layer_spacing, y_pos)
                
                # Feature interpretation.
                interpretation = feat_data.get('interpretation', 'N/A')
                if interpretation.startswith("Identifying "):
                    interpretation = interpretation[12:]
                
                node_info[node_id] = {
                    'type': 'feature',
                    'layer': layer_idx,
                    'feature_name': feat_name,
                    'activation': feat_data['max_activation'],
                    'interpretation': interpretation,
                    'sparsity': feat_data['sparsity']
                }
                feature_count += 1
                layer_feature_counts[layer_idx] += 1
    
    # Add weighted edges.
    # Organize nodes by layer.
    nodes_by_layer = {}
    embedding_nodes = []
    
    for node_id, info in node_info.items():
        if info['type'] == 'embedding':
            embedding_nodes.append(node_id)
        else:
            layer = info['layer']
            if layer not in nodes_by_layer:
                nodes_by_layer[layer] = []
            nodes_by_layer[layer].append(node_id)
    
    # Connect embeddings to first layer.
    if 0 in nodes_by_layer:
        for emb_node in embedding_nodes:
            for feat_node in nodes_by_layer[0]:
                # Connection strength.
                weight = node_info[feat_node]['activation'] * 0.7
                if weight > edge_threshold:
                    G.add_edge(emb_node, feat_node, weight=weight)
    
    # Connect features across layers.
    sorted_layers = sorted(nodes_by_layer.keys())
    for i in range(len(sorted_layers) - 1):
        current_layer = sorted_layers[i]
        next_layer = sorted_layers[i + 1]
        
        for source_node in nodes_by_layer[current_layer]:
            source_info = node_info[source_node]
            
            for target_node in nodes_by_layer[next_layer]:
                target_info = node_info[target_node]
                
                # Calculate weight.
                source_activation = source_info['activation']
                target_activation = target_info['activation']
                
                # Weight based on activation and similarity.
                base_weight = min(source_activation, target_activation) * 0.5
                
                # Sparsity similarity bonus.
                sparsity_similarity = 1.0 - abs(source_info['sparsity'] - target_info['sparsity'])
                similarity_bonus = sparsity_similarity * 0.2
                
                final_weight = base_weight + similarity_bonus
                
                # Add edge if threshold met.
                if final_weight > edge_threshold:
                    G.add_edge(source_node, target_node, weight=final_weight)
    
    # Create edge traces.
    edge_traces = []
    
    # Normalize edge thickness.
    all_weights = [G.get_edge_data(u, v).get('weight', 0.1) for u, v in G.edges()]
    min_weight = min(all_weights) if all_weights else 0.1
    max_weight = max(all_weights) if all_weights else 1.0

    for edge in G.edges():
        x0, y0 = node_positions[edge[0]]
        x1, y1 = node_positions[edge[1]]
        
        weight = G[edge[0]][edge[1]].get('weight', 0.1)
        
        # Normalize weight to thickness 0.5-4px.
        if max_weight > min_weight:
            thickness = 0.5 + 3.5 * (weight - min_weight) / (max_weight - min_weight)
        else:
            thickness = 2
        
        edge_trace = go.Scatter(
            x=[x0, x1], y=[y0, y1],
            line=dict(width=thickness, color='gray'),
            hoverinfo='text',
            hovertext=f"Connection Weight: {weight:.3f}<br>Thickness: {thickness:.1f}px",
            mode='lines',
            showlegend=False,
            opacity=0.7
        )
        edge_traces.append(edge_trace)
    
    # Collect activations for scaling.
    all_activations = [info['activation'] for info in node_info.values() if info['type'] == 'feature']
    if all_activations:
        max_activation = max(all_activations)
        min_activation = min(all_activations)
        # Square root scaling to compress large values.
        def scale_activation(act):
            if max_activation == min_activation:
                return 1.0
            # Normalize to [0, 1].
            normalized = (act - min_activation) / (max_activation - min_activation)
            # Compress.
            scaled = np.sqrt(normalized)
            return scaled
    else:
        def scale_activation(act):
            return 1.0
    
    # Create node traces.
    node_traces = {}
    
    for node in G.nodes():
        info = node_info[node]
        node_type = info['type']
        
        if node_type not in node_traces:
            node_traces[node_type] = {
                'x': [], 'y': [], 'text': [], 'hovertext': [],
                'size': [], 'color': []
            }
        
        x, y = node_positions[node]
        node_traces[node_type]['x'].append(x)
        node_traces[node_type]['y'].append(y)
        
        # Node label.
        if info['type'] == 'embedding':
            label = info['token']
        else:
            label = ""
        
        node_traces[node_type]['text'].append(label)
        
        # Hover info.
        if info['type'] == 'embedding':
            hover_text = f"Token: {info['token']}<br>Type: Embedding"
        else:
            hover_text = (f"Feature: {info['feature_name']}<br>"
                         f"Layer: {info['layer']}<br>"
                         f"Activation: {info['activation']:.3f}<br>"
                         f"Sparsity: {info['sparsity']:.3f}<br>"
                         f"Interpretation: {info['interpretation']}")
        
        node_traces[node_type]['hovertext'].append(hover_text)
        
        # Node size and color.
        if info['type'] == 'feature':
            # Scale activation.
            scaled_act = scale_activation(info['activation'])
            # Scaled size, original color.
            size = scaled_act * 75 * node_size_factor
        else:
            # Embeddings use original activation.
            size = info['activation'] * 20 * node_size_factor
        node_traces[node_type]['size'].append(max(size, 5))
        base_size = max(size, 5)
        node_traces[node_type]['color'].append(info['activation'])
        
    def map_path_node_id(raw_id: str) -> Optional[str]:
        if raw_id in node_positions:
            return raw_id
        if raw_id.startswith("feat_"):
            parts = raw_id.split('_')
            if len(parts) >= 4:
                layer_part = parts[1]
                feature_part = parts[3]
                try:
                    layer_idx = int(layer_part[1:]) if layer_part.startswith('L') else int(layer_part)
                    feature_idx = int(feature_part[1:]) if feature_part.startswith('F') else int(feature_part)
                    candidate = f"feat_{layer_idx}_feature_{feature_idx}"
                    if candidate in node_positions:
                        return candidate
                except ValueError:
                    return None
        return None

    # Highlight ablated paths.
    show_paths = st.toggle("Highlight ablated paths", value=False)
    path_highlight_traces: List[go.Scatter] = []
    path_results = analysis.get('path_ablation_experiments', []) or []
    if show_paths and path_results:
        for idx, exp in enumerate(path_results[:5]):
            raw_nodes = exp.get('path_nodes', []) or []
            mapped_sequence: List[str] = []
            for node_id in raw_nodes:
                mapped = map_path_node_id(node_id)
                if mapped:
                    mapped_sequence.append(mapped)
            if not mapped_sequence:
                feature_set = exp.get('feature_set', []) or []
                for feature in feature_set:
                    layer = feature.get('layer')
                    feat_idx = feature.get('feature')
                    if layer is None or feat_idx is None:
                        continue
                    candidate = f"feat_{int(layer)}_feature_{int(feat_idx)}"
                    if candidate in node_positions:
                        mapped_sequence.append(candidate)
            # Unique order.
            seen_nodes = set()
            ordered_coords = []
            for node_id in mapped_sequence:
                if node_id in seen_nodes or node_id not in node_positions:
                    continue
                ordered_coords.append(node_positions[node_id])
                seen_nodes.add(node_id)
            if len(ordered_coords) >= 2:
                x_vals = [coord[0] for coord in ordered_coords]
                y_vals = [coord[1] for coord in ordered_coords]
                try:
                    label = exp.get('path_description', f"Path {idx + 1}")
                    label = (label[:60] + "…") if label and len(label) > 60 else label
                except Exception:
                    label = f"Path {idx + 1}"
                try:
                    path_label_prefix = tr('path_highlight_label')
                except Exception:
                    path_label_prefix = "Circuit path"
                path_trace = go.Scatter(
                    x=x_vals,
                    y=y_vals,
                    mode='lines+markers',
                    line=dict(width=2, color='#FFD166', dash='dash'),
                    marker=dict(
                        size=10,
                        color='#FFD166',
                        line=dict(width=1, color='#1F2933'),
                        symbol='circle-open'
                    ),
                    opacity=0.95,
                    name=f"{path_label_prefix} {idx + 1}",
                    hoverinfo='text',
                    hovertext=label or f"Path {idx + 1}"
                )
                path_highlight_traces.append(path_trace)

    # Create layer labels.
    layer_annotations = []
    original_layers = analysis['feature_visualizations']

    # Calculate stable y-position.
    global_max_y = 200
    if original_layers:
        max_y_per_layer = []
        for layer_features in original_layers.values():
            num_features = len(layer_features)
            if num_features > 0:
                # Calculate max y for layer.
                max_y_for_layer = (num_features * 40) - 80
                max_y_per_layer.append(max_y_for_layer)
        if max_y_per_layer:
            global_max_y = max(max_y_per_layer)

    stable_label_y = global_max_y + 60

    # Sort layers.
    sorted_layer_names = sorted(original_layers.keys(), key=lambda x: int(x.split('_')[1]))

    for layer_name in sorted_layer_names:
        layer_idx = int(layer_name.split('_')[1])
        layer_x = (layer_idx + 1) * layer_spacing
        
        layer_annotations.append(dict(
            x=layer_x,
            y=stable_label_y,
            text=f"<b>L{layer_idx}</b>",
            showarrow=False,
            font=dict(size=14, color="#dcae36"),
            xanchor="center",
            yanchor="bottom"
        ))
    
    # Create final traces.
    traces = edge_traces
    traces.extend(path_highlight_traces)
    
    colors = {'embedding': 'lightblue', 'feature': 'lightgreen', 'output': 'orange'}
    
    for node_type, data in node_traces.items():
        if data['x']:
            trace = go.Scatter(
                x=data['x'], y=data['y'],
                mode='markers+text',
                hoverinfo='text',
                hovertext=data['hovertext'],
                text=data['text'],
                textposition="middle center",
                marker=dict(
                    size=data['size'],
                    color=data['color'],
                    colorscale='viridis',
                    showscale=True if node_type == 'feature' else False,
                    colorbar=dict(
                        title=tr('colorbar_title'),
                        x=0.97,
                        xanchor="left",
                        thickness=15,
                        len=0.7
                    ) if node_type == 'feature' else None,
                    line=dict(width=2, color='black')
                ),
                name=node_type.title()
            )
            # Manually translate legend names.
            if node_type == 'embedding':
                trace.name = tr('embedding_legend')
            elif node_type == 'feature':
                trace.name = tr('feature_legend')
                
            traces.append(trace)
    
    # Calculate total width.
    total_width = (max_layer + 2) * layer_spacing + 200
    
    # Create figure.
    fig = go.Figure(data=traces)
    
    # Combine annotations.
    all_annotations = [
        dict(
            text=tr('tip_scroll_horizontally'),
            showarrow=False,
            xref="paper", yref="paper",
            x=0.005, y=-0.002,
            xanchor='left', yanchor='bottom',
            font=dict(color="gray", size=12)
        )
    ] + layer_annotations
    
    fig.update_layout(
        showlegend=True,
        hovermode='closest',
        legend=dict(
            orientation='h',
            yanchor='bottom',
            y=1.07,
            xanchor='left',
            x=0,
            bgcolor='rgba(0,0,0,0)'
        ),
        margin=dict(b=20, l=5, r=120, t=110),
        annotations=all_annotations,
        xaxis=dict(
            showgrid=True, 
            zeroline=False, 
            showticklabels=True,
            # Enable horizontal scrolling.
            range=[0, min(1000, total_width)],
            # Add scrollbar.
            rangeslider=dict(
                visible=True,
                thickness=0.05
            ),
            autorange=True
        ),
        yaxis=dict(showgrid=True, zeroline=False, showticklabels=False, autorange=True),
        height=700,
        # Set specific width.
        width=None,
        autosize=True
    )
    
    # Get unique layers with features.
    layers_with_features = set()
    for node in G.nodes():
        info = node_info.get(node)
        if info and info['type'] == 'feature':
            layers_with_features.add(info['layer'])
    
    # Add container with horizontal scrollbar.
    st.markdown("""
    <div style="
        background-color: #2b2b2b; 
        padding: 10px; 
        border-radius: 5px; 
        margin-bottom: 10px;
        border-left: 4px solid #dcae36;
    ">
        <h4 style="margin: 0; color: #dcae36;">{layer_nav_header}</h4>
        <p style="margin: 5px 0 0 0; color: #ffffff;">
            {layer_nav_desc}
        </p>
    </div>
    """.format(
        layer_nav_header=tr('layer_nav_header'),
        layer_nav_desc=tr('layer_nav_desc').format(num_layers=len(layers_with_features))
    ), unsafe_allow_html=True)
    
    # Display plot.
    st.plotly_chart(fig, use_container_width=True, config={
        'scrollZoom': True,
        'displayModeBar': True,
        'modeBarButtonsToRemove': ['zoom2d', 'zoomIn2d', 'zoomOut2d'],
        'modeBarButtonsToAdd': ['pan2d', 'autoScale2d', 'select2d', 'lasso2d'],
        'modeBarButtons': [['pan2d', 'autoScale2d', 'select2d', 'lasso2d', 'resetScale2d']],
        'doubleClick': 'autosize',
        'dragmode': 'pan',
        'staticPlot': False,
    })
    
    # Display faithfulness metrics.
    st.markdown("---")
    render_faithfulness_metrics(analysis, prompt_idx)
    
    # Add AI explanation if enabled.
    if enable_explanations and qwen_api_config is not None:
        cache_key = f"explanation_circuit_graph_{prompt_idx}"

        # Prepare layer summaries.
        all_features = [info for info in node_info.values() if info['type'] == 'feature']
        early_layers = (0, 10)
        middle_layers = (11, 21)
        late_layers = (22, 31)
        early_feats = sorted([f for f in all_features if early_layers[0] <= f['layer'] <= early_layers[1]], key=lambda x: x['activation'], reverse=True)
        middle_feats = sorted([f for f in all_features if middle_layers[0] <= f['layer'] <= middle_layers[1]], key=lambda x: x['activation'], reverse=True)
        late_feats = sorted([f for f in all_features if late_layers[0] <= f['layer'] <= late_layers[1]], key=lambda x: x['activation'], reverse=True)
        
        analysis_with_context = analysis.copy()
        analysis_with_context['layer_summaries'] = {
            'early': early_feats[:5],
            'middle': middle_feats[:5],
            'late': late_feats[:5]
        }
        
        if cache_key not in st.session_state:
            with st.spinner(tr('generating_circuit_explanation_spinner')):
                try:
                    explanation = get_circuit_explanation(
                        qwen_api_config, 
                        fig, 
                        analysis_with_context, 
                        visualization_type="circuit_graph"
                    )
                    st.session_state[cache_key] = explanation
                except Exception as e:
                    st.error(tr('circuit_explanation_error').format(e=str(e)))
                    st.session_state[cache_key] = "Error: Could not generate explanation."
        
        if st.session_state.get(cache_key) and "Error:" not in st.session_state[cache_key] and "Unable to generate" not in st.session_state[cache_key]:
            
            explanation_text = st.session_state.get(cache_key, "")
            
            # Split explanation.
            parts = re.split(r'(?=\n####\s)', explanation_text.strip())
            parts = [p.strip() for p in parts if p.strip()]

            box_style = "background-color: #2b2b2b; color: #ffffff; padding: 1.2rem; border-radius: 10px; font-size: 0.9rem; margin-bottom: 1rem;"

            if not parts:
                # Handle malformed explanation.
                st.markdown(f'<div style="{box_style} border-left: 4px solid #dcae36;">{markdown.markdown(explanation_text)}</div>', unsafe_allow_html=True)
            else:
                intro_part = ""
                layers_part = ""
                insight_part = ""

                # Find insight part.
                insight_keywords = ["Primary Insight", "Zentrale Erkenntnis"]
                insight_idx = -1
                for i, p in enumerate(parts):
                    first_line = p.split('\n')[0]
                    if any(keyword in first_line for keyword in insight_keywords):
                        insight_idx = i
                        break
                
                intro_part = parts[0]

                if insight_idx != -1:
                    # Insight section found.
                    insight_part = parts[insight_idx]
                    # Everything else is the layers section.
                    if insight_idx > 1:
                        layers_part = "\n\n".join(parts[1:insight_idx])
                else:
                    # No insight section was found.
                    if len(parts) > 1:
                        layers_part = "\n\n".join(parts[1:])

                # Display the parts in colored boxes.
                if intro_part:
                    st.markdown(f'<div style="{box_style} border-left: 4px solid #dcae36;">{markdown.markdown(intro_part)}</div>', unsafe_allow_html=True)

                if layers_part:
                    st.markdown(f'<div style="{box_style} border-left: 4px solid #A78BFA;">{markdown.markdown(layers_part)}</div>', unsafe_allow_html=True)
                
                if insight_part:
                    st.markdown(f'<div style="{box_style} border-left: 4px solid #6EE7B7;">{markdown.markdown(insight_part)}</div>', unsafe_allow_html=True)

            # Faithfulness Check for the entire circuit graph explanation
            with st.expander(tr('faithfulness_check_expander')):
                st.markdown(tr('faithfulness_explanation_circuit_graph_html'), unsafe_allow_html=True)
                
                faithfulness_cache_key = f"faithfulness_circuit_graph_{prompt_idx}"
                if faithfulness_cache_key in st.session_state and st.session_state[faithfulness_cache_key] is not None:
                    verification_results = st.session_state[faithfulness_cache_key]
                else:
                    with st.spinner(tr('running_faithfulness_check_spinner')):
                        # Filter explanation to only include layer-specific sections
                        explanation_parts = re.split(r'(?=####\s)', explanation_text.strip())
                        
                        # Get keywords for layer sections to make it localization-friendly
                        def _heading_keyword(loc_key):
                            text = tr(loc_key)
                            heading_line = text.split('\n')[0].strip()
                            heading_line = heading_line.replace('####', '').strip()
                            return heading_line.split('(')[0].strip()

                        early_keyword = _heading_keyword('circuit_graph_instruction_early')
                        middle_keyword = _heading_keyword('circuit_graph_instruction_middle')
                        late_keyword = _heading_keyword('circuit_graph_instruction_late')

                        layer_keywords = [early_keyword, middle_keyword, late_keyword]
                        
                        relevant_parts = []
                        for part in explanation_parts:
                            if not part.strip().startswith('####'):
                                continue
                            
                            heading = part.split('\n')[0].replace('####', '').strip()
                            if any(keyword in heading for keyword in layer_keywords):
                                relevant_parts.append(part)
                        
                        relevant_text = "".join(relevant_parts) if relevant_parts else explanation_text
                        
                        claims = _cached_extract_circuit_claims(qwen_api_config, relevant_text, "circuit_graph", cache_version="faithfulness-2025-11-29")
                        verification_results = verify_circuit_claims(claims, analysis_with_context, "circuit_graph")
                        st.session_state[faithfulness_cache_key] = verification_results
                
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

    return fig, G, node_info, node_positions

def create_subnetwork_visualization(analysis, G, node_info, node_positions, enable_explanations=False, qwen_api_config=None):
    # Interactive subnetwork visualization.
    st.markdown(f"### {tr('subnetwork_explorer_title')}")
    st.write(tr('subnetwork_explorer_desc'))

    if G.number_of_nodes() == 0:
        st.info(tr('subnetwork_graph_empty_info'))
        return

    # --- UI Controls ---
    col1, col2, col3 = st.columns(3)

    with col1:
        # Layer selection.
        layers_in_graph = sorted(list(set(
            info['layer'] for node, info in node_info.items() if info['type'] == 'feature'
        )))
        
        if not layers_in_graph:
            st.warning(tr('no_features_in_graph_warning'))
            return
            
        layer_options = [f"layer_{l}" for l in layers_in_graph]
        selected_layer = st.selectbox(
            tr('select_layer_label_subnetwork'),
            layer_options,
            format_func=lambda x: tr('layer_label_format').format(layer_num=x.split('_')[1]),
            key="subnetwork_layer_selector"
        )

    with col2:
        # Feature selection.
        layer_idx = int(selected_layer.split('_')[1])
        feature_options = sorted([
            info['feature_name'] 
            for node, info in node_info.items() 
            if info['type'] == 'feature' and info['layer'] == layer_idx
        ])
        
        if not feature_options:
            st.warning(tr('no_features_in_layer_subnetwork_warning').format(selected_layer=selected_layer))
            return
            
        selected_feature = st.selectbox(
            tr('select_feature_label_subnetwork'),
            feature_options,
            key="subnetwork_feature_selector"
        )

    with col3:
        # Depth slider.
        traversal_depth = st.slider(tr('traversal_depth_label'), 1, 5, 2, key="subnetwork_depth_slider")

    # --- Subgraph Generation ---
    if selected_feature:
        selected_node_id = next((
            node_id for node_id, info in node_info.items() 
            if info.get('feature_name') == selected_feature and info.get('layer') == layer_idx
        ), None)
        
        if selected_node_id:
            # Find reachable nodes.
            downstream_nodes = set(nx.dfs_preorder_nodes(G, source=selected_node_id, depth_limit=traversal_depth))
            upstream_nodes = set(nx.dfs_preorder_nodes(G.reverse(), source=selected_node_id, depth_limit=traversal_depth))
            
            subgraph_nodes = sorted(list(upstream_nodes.union(downstream_nodes)))
            subgraph = G.subgraph(subgraph_nodes)

            # --- Layout ---
            viz_col, analysis_col = st.columns([2, 1])

            with viz_col:
                # --- Visualization ---
                if subgraph.number_of_nodes() > 0:
                    edge_traces = []
                    
                    # Edge traces.
                    all_weights = [subgraph.get_edge_data(u, v).get('weight', 0.1) for u, v in subgraph.edges()]
                    min_w, max_w = (min(all_weights), max(all_weights)) if all_weights else (0.1, 1.0)
    
                    for u, v, data in sorted(subgraph.edges(data=True)):
                        x0, y0 = node_positions[u]
                        x1, y1 = node_positions[v]
                        weight = data.get('weight', 0.1)
                        thickness = 0.5 + 3.5 * (weight - min_w) / (max_w - min_w) if max_w > min_w else 2
                        
                        edge_traces.append(go.Scatter(
                            x=[x0, x1], y=[y0, y1], line=dict(width=thickness, color='gray'),
                            hoverinfo='text', hovertext=f"Weight: {weight:.3f}", mode='lines', showlegend=False, opacity=0.7
                        ))
    
                    # Activations for scaling.
                    subgraph_activations = [node_info[n]['activation'] for n in subgraph.nodes() if node_info[n]['type'] == 'feature']
                    if subgraph_activations:
                        subgraph_max = max(subgraph_activations)
                        subgraph_min = min(subgraph_activations)
                        def scale_subgraph_activation(act):
                            if subgraph_max == subgraph_min:
                                return 1.0
                            normalized = (act - subgraph_min) / (subgraph_max - subgraph_min)
                            return np.sqrt(normalized)
                    else:
                        def scale_subgraph_activation(act):
                            return 1.0
                    
                    # Node data.
                    embedding_data = {'x': [], 'y': [], 'text': [], 'hovertext': [], 'size': [], 'color': [], 'ids': []}
                    feature_data = {'x': [], 'y': [], 'text': [], 'hovertext': [], 'size': [], 'ids': []}
    
                    for node in sorted(subgraph.nodes()):
                        info = node_info[node]
                        x, y = node_positions[node]
                        
                        if info['type'] == 'embedding':
                            target_data = embedding_data
                            label = info['token']
                            hover = f"Token: {info['token']}<br>Type: Embedding"
                            target_data['color'].append(info['activation'])
                            # Embeddings use original activation
                            node_size = info['activation'] * 30 + 5
                        else:  # feature
                            target_data = feature_data
                            label = f"F{info['feature_name'].split('_')[-1]}"
                            hover = f"Feature: {info['feature_name']}<br>Layer: {info['layer']}<br>Activation: {info['activation']:.3f}<br>Sparsity: {info.get('sparsity', 0.0):.3f}<br>Interpretation: {info.get('interpretation', 'N/A')}"
                            # Scale feature activations
                            scaled_act = scale_subgraph_activation(info['activation'])
                            node_size = scaled_act * 30 + 5
    
                        target_data['x'].append(x)
                        target_data['y'].append(y)
                        target_data['text'].append(label)
                        target_data['hovertext'].append(hover)
                        target_data['size'].append(node_size)
                        target_data['ids'].append(node)
    
                    # Assemble figure.
                    final_traces = edge_traces
                    
                    # Feature nodes trace.
                    if feature_data['x']:
                        final_traces.append(go.Scatter(
                            x=feature_data['x'], y=feature_data['y'], mode='markers+text', hoverinfo='text',
                            hovertext=feature_data['hovertext'], text=feature_data['text'], textposition="middle center",
                            marker=dict(
                                size=feature_data['size'],
                                color='purple',
                                showscale=False,  # Keep sub-graph clean
                                line=dict(width=3, color=['crimson' if nid == selected_node_id else 'black' for nid in feature_data['ids']])
                            ),
                            name=tr('feature_legend')
                        ))
                    
                    # Embedding nodes trace.
                    if embedding_data['x']:
                        final_traces.append(go.Scatter(
                            x=embedding_data['x'], y=embedding_data['y'], mode='markers+text', hoverinfo='text',
                            hovertext=embedding_data['hovertext'], text=embedding_data['text'], textposition="middle center",
                            marker=dict(
                                size=embedding_data['size'],
                                color=embedding_data['color'],
                                colorscale='viridis',
                                showscale=False,
                                line=dict(width=3, color=['crimson' if nid == selected_node_id else 'black' for nid in embedding_data['ids']])
                            ),
                            name=tr('embedding_legend')
                        ))
    
                    fig_sub = go.Figure(data=final_traces)
                    
                    # Layer annotations.
                    layer_annotations = []
                    layers_in_subgraph = sorted(list(set(
                        node_info[node]['layer'] 
                        for node in subgraph.nodes() 
                        if node_info[node]['type'] == 'feature'
                    )))
                    
                    if layers_in_subgraph:
                        max_y_in_subgraph = max(node_positions[node][1] for node in subgraph.nodes())
                        label_y_pos = max_y_in_subgraph + 25
                        
                        for layer_idx in layers_in_subgraph:
                            node_in_layer = next((node for node in subgraph.nodes() if node_info[node].get('layer') == layer_idx), None)
                            if node_in_layer:
                                layer_x = node_positions[node_in_layer][0]
                                layer_annotations.append(dict(
                                    x=layer_x,
                                    y=label_y_pos,
                                    text=f"<b>L{layer_idx}</b>",
                                    showarrow=False,
                                    font=dict(size=14, color="#dcae36"),
                                    xanchor="center",
                                    yanchor="bottom"
                                ))
    
                    fig_sub.update_layout(
                        title=tr('subnetwork_graph_title').format(feature=selected_feature),
                        showlegend=True, hovermode='closest', margin=dict(b=20, l=5, r=5, t=80),
                        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False, autorange=True),
                        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False, autorange=True),
                        height=600, autosize=True,
                        annotations=layer_annotations
                    )
                    st.plotly_chart(fig_sub, use_container_width=True)
    
                    # Subnetwork AI explanation.
                    if enable_explanations and qwen_api_config:
                        cache_key = f"explanation_subnetwork_{analysis['prompt']}_{selected_layer}_{selected_feature}_{traversal_depth}"
    
                        # Context for explanation.
                        central_info = node_info.get(selected_node_id, {})
                        context_data = analysis.copy()
                        context_data['central_feature_info'] = {
                            "name": central_info.get('feature_name', 'N/A'),
                            "layer": central_info.get('layer', 'N/A'),
                            "interpretation": central_info.get('interpretation', 'N/A'),
                        }
                        
                        # Neighbor details.
                        subgraph_feature_nodes = [
                            nid for nid in subgraph.nodes() 
                            if node_info[nid]['type'] == 'feature' and nid != selected_node_id
                        ]
                        central_layer = central_info.get('layer', -1)
                        
                        # Upstream tokens.
                        preds = subgraph.predecessors(selected_node_id)
                        upstream_tokens = [node_info[p]['token'] for p in preds if node_info[p]['type'] == 'embedding']
    
                        context_data['subgraph_neighbors'] = {
                            'upstream': sorted(
                                [node_info[nid] for nid in subgraph_feature_nodes if node_info[nid].get('layer', -2) < central_layer],
                                key=lambda x: x.get('activation', 0), reverse=True
                            ),
                            'downstream': sorted(
                                [node_info[nid] for nid in subgraph_feature_nodes if node_info[nid].get('layer', -2) > central_layer],
                                key=lambda x: x.get('activation', 0), reverse=True
                            ),
                            'upstream_tokens': upstream_tokens
                        }
    
                        context_data['subgraph_stats'] = {
                            "nodes": subgraph.number_of_nodes(),
                            "edges": subgraph.number_of_edges(),
                        }
                        context_data['depth'] = traversal_depth
    
                        if cache_key not in st.session_state:
                            with st.spinner(tr('generating_subnetwork_explanation_spinner')):
                                try:
                                    explanation = get_circuit_explanation(
                                        qwen_api_config, fig_sub, context_data, "subnetwork_graph"
                                    )
                                    st.session_state[cache_key] = explanation
                                except Exception as e:
                                    st.error(f"Error generating subnetwork explanation: {str(e)}")
                                    st.session_state[cache_key] = "Error: Could not generate explanation."
                        
                        explanation = st.session_state.get(cache_key)
                        if explanation and "Error:" not in explanation:
                            st.markdown(tr('ai_subnetwork_analysis_header'))
                            html_explanation = markdown.markdown(explanation)
                            st.markdown(f"""
                            <div style="background-color: #2b2b2b; color: #ffffff; padding: 1.2rem; border-radius: 10px; border-left: 4px solid #A78BFA; font-size: 0.9rem; margin-bottom: 1rem;">
                                {html_explanation}
                            </div>
                            """, unsafe_allow_html=True)
    
                            # Faithfulness Check.
                            with st.expander(tr('faithfulness_check_expander')):
                                st.markdown(tr('faithfulness_explanation_subnetwork_graph_html'), unsafe_allow_html=True)
                                with st.spinner(tr('running_faithfulness_check_spinner')):
                                    claims = _cached_extract_circuit_claims(qwen_api_config, explanation, "subnetwork_graph", cache_version="faithfulness-2025-11-29")
                                    verification_results = verify_circuit_claims(claims, context_data, "subnetwork_graph")
                                    
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
                    st.info(tr('subnetwork_no_connections_info'))
            
            with analysis_col:
                st.markdown(f"#### {tr('subnetwork_analysis_title')}")

                # Find all features in the subnetwork.
                subgraph_features = [
                    (node, node_info[node])
                    for node in subgraph.nodes()
                    if node_info[node]['type'] == 'feature'
                ]

                if not subgraph_features:
                    st.info(tr('subnetwork_no_features_info'))
                else:
                    # Aggregate top activating tokens from all features.
                    all_top_activations = []
                    for _, feat_info in subgraph_features:
                        layer_name = f"layer_{feat_info['layer']}"
                        feat_name = feat_info['feature_name']
                        
                        # Get token activations from the original analysis data.
                        viz_data = analysis.get('feature_visualizations', {}).get(layer_name, {}).get(feat_name, {})
                        if 'top_activations' in viz_data:
                            for act_item in viz_data['top_activations']:
                                all_top_activations.append({
                                    'token': act_item['token'],
                                    'activation': act_item['activation'],
                                    'feature': feat_name
                                })
                    
                    if not all_top_activations:
                        st.info(tr('subnetwork_no_token_info'))
                    else:
                        # Find the top 10 unique tokens by max activation.
                        df_activations = pd.DataFrame(all_top_activations)
                        top_tokens = df_activations.loc[df_activations.groupby('token')['activation'].idxmax()]
                        top_tokens = top_tokens.nlargest(10, 'activation')
                        
                        st.write(tr('subnetwork_top_tokens_desc'))

                        for _, row in top_tokens.iterrows():
                            # Normalize activation for the bar display.
                            normalized_act = (row['activation'] - df_activations['activation'].min()) / \
                                             (df_activations['activation'].max() - df_activations['activation'].min())
                            bar_length = int(normalized_act * 10)
                            bar = '█' * bar_length + ' ' * (10 - bar_length)

                            st.markdown(f"`{row['token']}`: <span style='font-family: monospace; color: #A78BFA;'>[{bar}]</span> ({row['activation']:.2f})", unsafe_allow_html=True)

                        st.info(tr('subnetwork_token_interpretation_info'))

def show_circuit_trace_page():
    # Circuit Trace page.
    # Bootstrap icons.
    st.markdown('<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/bootstrap-icons@1.10.5/font/bootstrap-icons.css">', unsafe_allow_html=True)
    
    # Header.
    st.markdown(f"<h1><i class='bi bi-diagram-3-fill'></i> {tr('circuit_trace_page_title')}</h1>", unsafe_allow_html=True)
    st.markdown(f"<p>{tr('circuit_trace_page_desc')}</p>", unsafe_allow_html=True)
    
    display_feature_explanation()
    display_circuit_trace_explanation()
    
    # Load results.
    lang = st.session_state.get('lang', 'en')
    
    results = load_attribution_results(lang)
    if results is None:
        st.warning(tr('no_results_warning'))
        st.info(tr('run_analysis_info'))
        return
    
    # Config details.
    with st.expander(f"{tr('config_header')}"):
        config = results['config']
        
        col1, col2 = st.columns(2)
        with col1:
            st.write(f"**{tr('model_label')}** {config['model_path']}")
            st.write(f"**{tr('device_label')}** {config['device']}")
            st.write(f"**{tr('features_per_layer_label')}** {config['n_features_per_layer']}")
        with col2:
            st.write(f"**{tr('training_steps_label')}** {config['training_steps']}")
            st.write(f"**{tr('batch_size_label')}** {config['batch_size']}")
            st.write(f"**{tr('learning_rate_label')}** {config['learning_rate']}")
    
    aggregate_summary = results.get('aggregate_summary')
    if aggregate_summary:
        st.markdown("### Dataset Faithfulness Summary")
        render_dataset_faithfulness_summary(aggregate_summary)
    
    # Interactive analysis.
    st.markdown("---")
    st.markdown(f"<h2><i class='bi bi-magic'></i> {tr('interactive_analysis_header')}</h2>", unsafe_allow_html=True)
    
    # Prompt selector.
    prompt_map = {analysis['prompt']: key for key, analysis in results['analyses'].items()}

    selected_prompt_text = st.selectbox(
        tr('select_prompt_label'),
        list(prompt_map.keys()),
        key="prompt_selector",
        help=tr('select_prompt_help')
    )
    
    # AI explanations checkbox.
    enable_explanations = st.checkbox(
        tr('enable_ai_explanations_circuit'),
        value=True,
        help=tr('enable_ai_explanations_circuit_help')
    )

    # Initialize API.
    qwen_api_config = None
    if enable_explanations:
        qwen_api_config = init_qwen_api()

    # Get selected prompt key.
    selected_prompt_key = prompt_map[selected_prompt_text]
    
    analysis = results['analyses'][selected_prompt_key]
    prompt_idx = int(selected_prompt_key.split('_')[-1])
    
    # Load cached data.
    cached_circuit_data = {}
    cache_file = Path(__file__).parent.parent / "cache" / "cached_circuit_trace_results.json"
    if cache_file.exists():
        with open(cache_file, 'r', encoding='utf-8') as f:
            all_cached_data = json.load(f)
            if analysis['prompt'] in all_cached_data:
                cached_circuit_data = all_cached_data[analysis['prompt']]

    # Pass cached data into session state for the UI components to use
    if 'circuit_graph' in cached_circuit_data:
        st.session_state[f"explanation_circuit_graph_{prompt_idx}"] = cached_circuit_data['circuit_graph'].get('explanation')
        st.session_state[f"faithfulness_circuit_graph_{prompt_idx}"] = cached_circuit_data['circuit_graph'].get('faithfulness')
    
    # Create the interactive circuit graph.
    circuit_fig, G, node_info, node_positions = create_interactive_attribution_graph(
        analysis, prompt_idx, enable_explanations, qwen_api_config
    )
    
    # Graph statistics.
    st.markdown("---")

    # Subnetwork analysis.
    create_subnetwork_visualization(
        analysis, G, node_info, node_positions, enable_explanations, qwen_api_config
    )
    
    st.markdown("---")

    # Feature explorer and token analysis.
    st.markdown(f"### <i class='bi bi-search'></i> {tr('feature_explorer_header')}", unsafe_allow_html=True)
    feature_fig = create_interactive_feature_explorer(analysis, prompt_idx, enable_explanations, qwen_api_config)
    
    # Token breakdown.
    st.subheader(tr('token_analysis_header'))
    tokens = analysis['input_tokens']
    st.write(f"**{tr('input_tokens_label')}**", " | ".join([f"`{token}`" for token in tokens]))

    # Display the feedback survey in the sidebar.
   # if 'analyses' in results:
    #    display_circuit_trace_feedback()
    

def display_feature_explanation():
    # Feature explanation.
    st.markdown(f"<h2> {tr('what_is_a_feature_header')}</h2>", unsafe_allow_html=True)
    st.html(f"""
        <div style="background-color: #2b2b2b; border-radius: 10px; padding: 1.5rem; margin: 1rem 0; border-left: 4px solid #6EE7B7;">
            <div style="display: flex; align-items: center; gap: 1.5rem;">
                <i class="bi bi-lightbulb-fill" style="font-size: 3rem; color: #6EE7B7;"></i>
                <div>
                    <h5 style="color: #6EE7B7; margin-top: 0;">{tr('what_is_a_feature_title')}</h5>
                    <p style="font-size: 0.9rem; margin-bottom: 0;">{tr('what_is_a_feature_desc')}</p>
                </div>
            </div>
        </div>
    """)

def display_circuit_trace_explanation():
    # Methodology explanation.
    st.markdown(f"<h2>{tr('how_circuit_tracing_works_header')}</h2>", unsafe_allow_html=True)
    st.html(f"""
        <div style="background-color: #2b2b2b; border-radius: 10px; padding: 1.5rem; margin: 1rem 0; border-left: 4px solid #A78BFA;">
            <p style="font-size: 1rem;">{tr('how_circuit_tracing_works_desc')}</p>
            <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 2rem; padding: 1rem 0;">
                
                <!-- Step 1 -->
                <div style="text-align: center;">
                    <i class="bi bi-box-seam" style="font-size: 3rem; color: #A78BFA;"></i>
                    <h5 style="color: #A78BFA; margin-top: 1rem;">{tr('circuit_tracing_step1_title')}</h5>
                    <p style="font-size: 0.9rem;">{tr('circuit_tracing_step1_desc')}</p>
                </div>

                <!-- Step 2 -->
                <div style="text-align: center;">
                    <i class="bi bi-activity" style="font-size: 3rem; color: #A78BFA;"></i>
                    <h5 style="color: #A78BFA; margin-top: 1rem;">{tr('circuit_tracing_step2_title')}</h5>
                    <p style="font-size: 0.9rem;">{tr('circuit_tracing_step2_desc')}</p>
                </div>

                <!-- Step 3 -->
                <div style="text-align: center;">
                    <i class="bi bi-diagram-3" style="font-size: 3rem; color: #A78BFA;"></i>
                    <h5 style="color: #A78BFA; margin-top: 1rem;">{tr('circuit_tracing_step3_title')}</h5>
                    <p style="font-size: 0.9rem;">{tr('circuit_tracing_step3_desc')}</p>
                </div>

            </div>
        </div>
    """)

def _compute_layer_max_activations(analysis_data):
    layer_max = {}
    feature_visualizations = analysis_data.get('feature_visualizations', {})
    for layer_name, features in feature_visualizations.items():
        try:
            layer_idx = int(layer_name.split('_')[1])
        except (IndexError, ValueError):
            continue
        max_activation = max((feat.get('max_activation', 0.0) for feat in features.values()), default=None)
        if max_activation is not None:
            layer_max[layer_idx] = max_activation
    return layer_max

def _check_activation_numbers(claim_text, layer_max_activations, layer_section=None, tolerance=0.05):
    # Validate activation numbers.
    claim_lower = claim_text.lower()
    if 'activation' not in claim_lower and 'value' not in claim_lower and 'reach' not in claim_lower:
        return True, ""

    pattern = re.compile(r'Layer\s*(\d+)([^\d]{0,120}?)([0-9]+(?:\.[0-9]+)?)', re.IGNORECASE | re.DOTALL)
    matches = []
    for match in pattern.finditer(claim_text):
        layer_idx = int(match.group(1))
        context = match.group(2).lower()
        number = float(match.group(3))
        if any(keyword in context for keyword in ['activation', 'value', 'reach', 'level']):
            matches.append((layer_idx, number))

    section_ranges = {
        'early': range(0, 11),
        'middle': range(11, 22),
        'late': range(22, 32),
    }

    if not matches:
        generic_numbers = re.findall(r'([0-9]+(?:\.[0-9]+)?)', claim_text)
        numbers = []
        for num_str in generic_numbers:
            number = float(num_str)
            idx = claim_text.find(num_str)
            if idx == -1:
                continue
            context_window = claim_lower[max(0, idx - 80): idx + 80]
            before_window = claim_lower[max(0, idx - 10): idx]
            before_window_extended = claim_lower[max(0, idx - 25): idx]
            
            is_integer = '.' not in num_str
            
            # Case 1: "Layer 12" or "Layers 12"
            if ('layer' in before_window or 'layers' in before_window) and is_integer:
                continue
                
            # Case 2: "Layers 12 and 20" or "Layers 12, 15, 20"
            if is_integer and ('layer' in before_window_extended or 'layers' in before_window_extended):
                if 'and' in before_window or ',' in before_window:
                    continue

            if idx > 0:
                preceding_char = claim_text[idx - 1]
                if preceding_char.lower() == 'l':
                    continue
            if any(keyword in context_window for keyword in ['activation', 'value', 'reach', 'level', 'high', 'higher', 'increase']):
                numbers.append(number)

        if numbers and layer_section in section_ranges:
            ranges = section_ranges[layer_section]
            available = [layer_max_activations.get(i) for i in ranges if layer_max_activations.get(i) is not None]
            if available:
                section_max = max(available)
                matches = [(None, num, section_max) for num in numbers]
        elif numbers:
            overall_max = max(layer_max_activations.values()) if layer_max_activations else None
            if overall_max is not None:
                matches = [(None, num, overall_max) for num in numbers]

    if not matches:
        return True, ""

    evidences = []
    all_verified = True
    for item in matches:
        if len(item) == 3:
            layer_idx = None
            claimed_value, actual_value = item[1], item[2]
        else:
            layer_idx, claimed_value = item
            actual_value = layer_max_activations.get(layer_idx)

        if actual_value is None:
            target = f"Layer {layer_idx}" if layer_idx is not None else (f"{layer_section or 'overall'} section")
            evidences.append(f"No activation data available for {target} to verify claimed {claimed_value:.2f}.")
            all_verified = False
            continue

        if actual_value + tolerance < claimed_value:
            evidences.append(f"Claimed activation {claimed_value:.2f} for Layer {layer_idx} exceeds actual max {actual_value:.2f}.")
            all_verified = False
        else:
            if layer_idx is None:
                evidences.append(f"Section max activation {actual_value:.2f} supports claimed {claimed_value:.2f}.")
            else:
                evidences.append(f"Layer {layer_idx} max activation {actual_value:.2f} supports claimed {claimed_value:.2f}.")

    return all_verified, " ".join(evidences)

@st.cache_data(persist=True)
def verify_circuit_claims(claims, analysis_data, context):
    # Verify circuit trace claims.
    verification_results = []
    
    layer_max_activations = _compute_layer_max_activations(analysis_data) if context == "circuit_graph" else {}
    
    for claim in claims:
        is_verified = False
        evidence = "Could not be verified."
        details = claim.get('details', {})
        layer_section_for_activation = None
        
        try:
            claim_type = claim.get('claim_type')
            
            if context == "feature_explorer":
                if claim_type == 'top_token_activation_claim':
                    tokens_claimed = details.get('tokens', [])
                    feature_data = analysis_data.get('feature_visualizations', {}).get(analysis_data['selected_layer'], {}).get(analysis_data['selected_feature'], {})
                    top_activations = feature_data.get('top_activations', [])
                    top_tokens = [act['token'] for act in top_activations]
                    
                    if not tokens_claimed:
                        evidence = "Claim did not specify any tokens to verify."
                    else:
                        # 1. Verify token presence.
                        normalized_actual = _normalize_actual_tokens(top_tokens)
                        normalized_actual_list = [_normalize_token_core(t) for t in top_tokens or []]
                        unverified_tokens = []
                        tokens_present = True
                        for token in tokens_claimed:
                            if not _token_matches_actual(token, normalized_actual, normalized_actual_list):
                                tokens_present = False
                                unverified_tokens.append(f"'{token}'")

                        # 2. Verify reasoning.
                        api_config = init_qwen_api()
                        if not api_config:
                            reasoning_verified = False
                            reasoning_evidence = "API key not configured for semantic verification."
                        else:
                            verification = _cached_verify_token_reasoning(
                                api_config,
                                _stringify_summary(claim.get('claim_text', '')),
                                feature_data,
                                analysis_data.get('selected_layer'),
                                cache_version="faithfulness-2025-11-29"
                            )
                            reasoning_verified = verification.get('is_verified', False)
                            reasoning_evidence = verification.get('reasoning', "Failed to get semantic reasoning.")
                        
                        # Combine.
                        is_verified = tokens_present and reasoning_verified
                        
                        evidence_parts = []
                        if tokens_present:
                            evidence_parts.append(f"Verified: All claimed tokens ({', '.join(tokens_claimed)}) were found in the top activators.")
                        else:
                            evidence_parts.append(f"Contradicted: The following tokens were not found: {', '.join(unverified_tokens)}.")
                        
                        evidence_parts.append(f"Reasoning check: {reasoning_evidence}")
                        evidence = " ".join(evidence_parts)

                elif claim_type == 'feature_interpretation_claim':
                    interpretation_summaries = details.get('interpretation_summaries', [])
                    feature_data = analysis_data.get('feature_visualizations', {}).get(analysis_data['selected_layer'], {}).get(analysis_data['selected_feature'], {})

                    api_config = init_qwen_api()
                    if not api_config:
                        evidence = "API key not configured for semantic verification."
                    else:
                        results = []
                        # Semantic check.
                        base_claim = _stringify_summary(claim.get('claim_text', ''))
                        claim_variants = [base_claim]
                        if interpretation_summaries:
                            for summary in interpretation_summaries:
                                summary_text = _stringify_summary(summary)
                                if not summary_text:
                                    continue
                                if summary_text.lower() not in base_claim.lower():
                                    claim_variants.append(f"{base_claim} (Summary: {summary_text})")

                        for claimed_role in claim_variants:
                            verification = _cached_verify_feature_role_claim(
                                api_config,
                                claimed_role,
                                feature_data,
                                analysis_data.get('selected_layer'),
                                cache_version="faithfulness-2025-11-29"
                            )
                            results.append(verification)
                            if verification.get('is_verified'):
                                break

                        final_verification = next((res for res in results if res.get('is_verified')), results[-1] if results else {})
                        is_verified = final_verification.get('is_verified', False)
                        evidence = final_verification.get('reasoning', "Failed to get semantic reasoning.")

            elif context == "circuit_graph":
                claim_type = claim.get('claim_type')

                if claim_type == 'layer_role_claim':
                    claim_details = details
                    if not isinstance(claim_details, list):
                        claim_details = [claim_details]

                    if not claim_details:
                        is_verified = False
                        evidence = "Claim missing details for layer roles."
                    else:
                        all_verified = True
                        evidence_parts = []
                        api_config = init_qwen_api()

                        if not api_config:
                            is_verified = False
                            evidence = "API key not configured for semantic verification."
                        else:
                            for detail_item in claim_details:
                                layer_section = detail_item.get('layer_section')
                                role_summary = detail_item.get('role_summary')

                                if not role_summary or not layer_section:
                                    all_verified = False
                                    evidence_parts.append("Skipped a detail item because it was missing 'layer_section' or 'role_summary'.")
                                    continue
                                
                                start, end = {"early": (0, 10), "middle": (11, 21), "late": (22, 31)}.get(layer_section, (None, None))
                                if start is None:
                                    all_verified = False
                                    evidence_parts.append(f"Invalid layer section ('{layer_section}').")
                                    continue
                                
                                actual_interpretations = []
                                all_visualizations = analysis_data.get('feature_visualizations', {})
                                for i in range(start, end + 1):
                                    layer_name = f"layer_{i}"
                                    if layer_name in all_visualizations:
                                        features_in_layer = all_visualizations[layer_name]
                                        actual_interpretations.extend([data.get('interpretation', '') for data in features_in_layer.values() if data.get('interpretation')])
                                
                                # Context cues.
                                if layer_section == "early":
                                    actual_interpretations.append("This layer family focuses on dissecting the input into foundational components such as surface structure, grammar, and basic patterns.")
                                elif layer_section == "middle":
                                    actual_interpretations.append("This layer family links earlier low-level recognitions into broader constructs, weaving them into coherent themes and contextual meaning.")
                                    actual_interpretations.append("This layer family refines understanding toward more nuanced, higher-level abstractions built from the earlier components.")
                                elif layer_section == "late":
                                    actual_interpretations.append("This layer family synthesizes accumulated abstractions to finalize consolidated, coherent outputs ready for downstream use.")
                                
                                verification = _cached_verify_semantic_summary(api_config, role_summary, actual_interpretations, layer_section, cache_version="faithfulness-2025-11-29")
                                if not verification.get('is_verified', False):
                                    all_verified = False
                                evidence_parts.append(f"Summary '{role_summary}': {verification.get('reasoning', 'Failed')}")
                            
                            is_verified = all_verified
                            evidence = " ".join(evidence_parts)

                elif claim_type == 'feature_interpretation_claim':
                    # List of details for layers.
                    claim_details = details
                    if not isinstance(claim_details, list):
                        claim_details = [claim_details]
                    
                    if not claim_details:
                        is_verified = False
                        evidence = "Claim did not contain details to verify."
                    else:
                        all_verified = True
                        evidence_parts = []
                        api_config = None
                        section_interpretations = {"early": set(), "middle": set(), "late": set()}
                        
                        for detail_item in claim_details:
                            layer = detail_item.get('layer')
                            interpretation_summary = detail_item.get('interpretation_summary')

                            if interpretation_summary is None or layer is None:
                                all_verified = False
                                evidence_parts.append("Skipped a detail item because it was missing 'layer' or 'interpretation_summary'.")
                                continue

                            try:
                                layer_idx = int(layer)
                            except (TypeError, ValueError):
                                layer_idx = None

                            if layer_idx is not None:
                                if layer_idx <= 10:
                                    layer_section = "early"
                                elif 11 <= layer_idx <= 21:
                                    layer_section = "middle"
                                else:
                                    layer_section = "late"
                            else:
                                layer_section = "early"

                            layer_name = f"layer_{layer}"
                            features_in_layer = analysis_data.get('feature_visualizations', {}).get(layer_name, {})
                            actual_interpretations = [data.get('interpretation', '') for data in features_in_layer.values() if data.get('interpretation')]

                            filtered_actual = [interp for interp in actual_interpretations if interp]
                            if filtered_actual:
                                section_interpretations[layer_section].update(filtered_actual)
                            
                            if not actual_interpretations:
                                all_verified = False
                                evidence_parts.append(f"For Layer {layer}, no active features with interpretations were found.")
                            else:
                                match, score = process.extractOne(interpretation_summary, actual_interpretations)
                                if match and score > 85:
                                    evidence_parts.append(f"Verified: '{interpretation_summary}' in L{layer} matched '{match}'.")
                                else:
                                    if api_config is None:
                                        api_config = init_qwen_api()
                                    if not api_config:
                                        all_verified = False
                                        evidence_parts.append("API key not configured for semantic verification.")
                                    else:
                                        augmented_interpretations = list(filtered_actual) if filtered_actual else list(actual_interpretations)
                                        if layer_section == "early":
                                            augmented_interpretations.append("This layer family focuses on foundational grammar, syntax, and basic sentence structure.")
                                        elif layer_section == "middle":
                                            augmented_interpretations.append("This layer family integrates context and develops thematic meaning from the earlier components.")
                                        else:
                                            augmented_interpretations.append("This layer family synthesizes accumulated information to finalize the model's answer.")

                                        verification = _cached_verify_semantic_summary(
                                            api_config,
                                            interpretation_summary,
                                            augmented_interpretations,
                                            layer_section,
                                            cache_version="faithfulness-2025-11-29"
                                        )
                                        if verification.get('is_verified', False):
                                            evidence_parts.append(f"Verified via semantic reasoning: {verification.get('reasoning', 'Consistent with layer behavior.')}")
                                        else:
                                            all_verified = False
                                            evidence_parts.append(f"Contradicted: {verification.get('reasoning', 'Summary not supported.')}")

                        claim_text_full = _stringify_summary(claim.get('claim_text', ''))
                        if claim_text_full:
                            if api_config is None:
                                api_config = init_qwen_api()
                            if api_config:
                                section_guidance = {
                                    "early": "This layer family focuses on foundational grammar, syntax, and basic sentence structure.",
                                    "middle": "This layer family integrates context and develops thematic meaning from the earlier components.",
                                    "late": "This layer family synthesizes accumulated information to finalize the model's answer."
                                }
                                for section, interpretations in section_interpretations.items():
                                    if not interpretations:
                                        continue
                                    augmented = list(interpretations)
                                    augmented.append(section_guidance.get(section, ""))
                                    verification = _cached_verify_semantic_summary(
                                        api_config,
                                        claim_text_full,
                                        augmented,
                                        section,
                                        cache_version="faithfulness-2025-11-29"
                                    )
                                    if verification.get('is_verified', False):
                                        evidence_parts.append(f"Semantic reasoning for {section} layers: {verification.get('reasoning', 'Consistent with the broader explanation.')}")

                        is_verified = all_verified
                        evidence = " ".join(evidence_parts)

            elif context == "subnetwork_graph":
                claim_type = claim.get('claim_type')

                if claim_type == 'feature_interpretation_claim':
                    interpretation_summaries = details.get('interpretation_summaries', [])
                    central_feature_info = analysis_data.get('central_feature_info', {})
                    
                    if not interpretation_summaries:
                        evidence = "Claim missing interpretation summaries."
                    elif not central_feature_info:
                        evidence = "Central feature information not available in data."
                    else:
                        api_config = init_qwen_api()
                        if not api_config:
                            evidence = "API key not configured for semantic verification."
                        else:
                            # Neighbor info for context.
                            neighbors = analysis_data.get('subgraph_neighbors', {})
                            upstream_formatted, _, _ = _prepare_feature_interpretations(neighbors.get('upstream', [])[:5])
                            downstream_formatted, _, _ = _prepare_feature_interpretations(neighbors.get('downstream', [])[:5])
                            neighbor_info = {
                                'upstream': upstream_formatted,
                                'downstream': downstream_formatted
                            }
                            
                            # Full data for central feature.
                            central_feature_name = central_feature_info.get('name')
                            central_layer_idx = central_feature_info.get('layer')
                            central_layer_name = f"layer_{central_layer_idx}"
                            full_central_feature_data = analysis_data.get('feature_visualizations', {}).get(central_layer_name, {}).get(central_feature_name, {})
                            full_central_feature_data.update(central_feature_info)

                            verification = _cached_verify_feature_role_claim(
                                api_config,
                                claim.get('claim_text', ''),
                                full_central_feature_data,
                                f"layer_{central_feature_info.get('layer')}",
                                neighbor_info,
                                cache_version="faithfulness-2025-11-29"
                            )
                            is_verified = verification.get('is_verified', False)
                            evidence = verification.get('reasoning', "Failed to get semantic reasoning.")
                
                elif claim_type == 'token_influence_claim':
                    tokens_claimed = details.get('tokens', [])
                    neighbors = analysis_data.get('subgraph_neighbors', {})
                    actual_upstream_tokens = neighbors.get('upstream_tokens', [])
                    
                    normalized_actual_tokens = _normalize_actual_tokens(actual_upstream_tokens)
                    normalized_actual_list = [_normalize_token_core(t) for t in actual_upstream_tokens or []]
                    unverified_tokens = []
                    for token in tokens_claimed:
                        if not _token_matches_actual(token, normalized_actual_tokens, normalized_actual_list):
                            # Heuristic: Ignore descriptive text (contains spaces and > 2 words)
                            if " " in token.strip() and len(token.strip().split()) > 2:
                                continue
                            unverified_tokens.append(token)
                    
                    is_verified = not unverified_tokens
                    actual_tokens_display = format_tokens_for_display(actual_upstream_tokens) or " ".join(actual_upstream_tokens)
                    if is_verified:
                        evidence = f"Verified. All claimed tokens {tokens_claimed} were found within the actual upstream token sequence: {actual_tokens_display}."
                    else:
                        evidence = f"The following claimed tokens were not found as direct upstream influences: {unverified_tokens}. Actual upstream tokens: {actual_tokens_display}."

                elif claim_type == 'causal_claim':
                    source_interps = details.get('source_feature_interpretations', [])
                    target_interps = details.get('target_feature_interpretations', [])
                    relationship = details.get('relationship')
                    
                    neighbors = analysis_data.get('subgraph_neighbors', {})
                    central_info_dict = analysis_data.get('central_feature_info', {}) or {}
                    central_interp_raw = central_info_dict.get('interpretation', '')
                    central_interp = _clean_interpretation_text(central_interp_raw) or central_interp_raw
                    central_feature_str = None
                    if central_info_dict:
                        central_feature_str = (
                            f"L{central_info_dict.get('layer', 'N/A')}: "
                            f"{central_info_dict.get('name', 'Unknown feature')} "
                            f"('{central_interp or 'N/A'}')"
                        )
                    
                    if relationship == 'upstream':
                        upstream_features = neighbors.get('upstream', [])
                        formatted_interpretations, fuzzy_candidates, normalized_actual = _prepare_feature_interpretations(upstream_features)
                        fuzzy_pool = list(dict.fromkeys(fuzzy_candidates + formatted_interpretations))
                        
                        all_verified = True
                        verified_sources = []
                        contradicted_sources = []
                        evidence_parts = []
                        
                        if not source_interps:
                            evidence_parts.append("Claim did not include explicit upstream feature interpretations; using semantic reasoning only.")
                        else:
                            for source_interp in source_interps:
                                if not source_interp:
                                    continue
                                claim_variants = {source_interp, _clean_interpretation_text(source_interp)}
                                matched = False
                                for variant in claim_variants:
                                    normalized_variant = _normalize_interpretation_text(variant)
                                    if normalized_variant and normalized_variant in normalized_actual:
                                        matched = True
                                        break
                                if not matched and fuzzy_pool:
                                    variant_for_fuzzy = _clean_interpretation_text(source_interp) or source_interp
                                    match = process.extractOne(variant_for_fuzzy, fuzzy_pool)
                                    if match and match[1] > 85:
                                        matched = True

                                if matched:
                                    verified_sources.append(f"'{source_interp}'")
                                else:
                                    if re.search(r'feature_\\d+', source_interp, re.IGNORECASE):
                                        all_verified = False
                                        contradicted_sources.append(f"'{source_interp}'")
                            
                            if all_verified:
                                evidence_parts.append(f"Verified all upstream influences: {', '.join(verified_sources)}.")
                            else:
                                if verified_sources:
                                    evidence_parts.append(f"Verified: {', '.join(verified_sources)}.")
                                if contradicted_sources:
                                    evidence_parts.append(f"Contradicted: {', '.join(contradicted_sources)} not found.")
                        
                        evidence = " ".join(evidence_parts)
                        
                        # Semantic check.
                        api_config = init_qwen_api()
                        if not api_config:
                            reasoning_verified = False
                            reasoning_evidence = "API key not configured."
                        else:
                            verification = _cached_verify_causal_reasoning(
                                api_config,
                                claim.get('claim_text', ''),
                                formatted_interpretations,
                                [central_interp],
                                central_feature_info=central_feature_str,
                                cache_version="faithfulness-2025-11-29"
                            )
                            reasoning_verified = verification.get('is_verified', False)
                            reasoning_evidence = verification.get('reasoning', 'Failed')
                        
                        is_verified = all_verified and reasoning_verified
                        evidence += f" Reasoning check: {reasoning_evidence}"

                    elif relationship == 'downstream':
                        downstream_features = neighbors.get('downstream', [])
                        formatted_interpretations, fuzzy_candidates, normalized_actual = _prepare_feature_interpretations(downstream_features)
                        fuzzy_pool = list(dict.fromkeys(fuzzy_candidates + formatted_interpretations))
                        
                        all_verified = True
                        verified_targets = []
                        contradicted_targets = []
                        evidence_parts = []

                        if not target_interps:
                            evidence_parts.append("Claim did not include explicit downstream feature interpretations; using semantic reasoning only.")
                        else:
                            for target_interp in target_interps:
                                if not target_interp:
                                    continue
                                claim_variants = {target_interp, _clean_interpretation_text(target_interp)}
                                matched = False
                                for variant in claim_variants:
                                    normalized_variant = _normalize_interpretation_text(variant)
                                    if normalized_variant and normalized_variant in normalized_actual:
                                        matched = True
                                        break
                                if not matched and fuzzy_pool:
                                    variant_for_fuzzy = _clean_interpretation_text(target_interp) or target_interp
                                    match = process.extractOne(variant_for_fuzzy, fuzzy_pool)
                                    if match and match[1] > 85:
                                        matched = True
                                
                                if matched:
                                    verified_targets.append(f"'{target_interp}'")
                                else:
                                    if re.search(r'feature_\\d+', target_interp, re.IGNORECASE):
                                        all_verified = False
                                        contradicted_targets.append(f"'{target_interp}'")
                            
                            if all_verified:
                                evidence_parts.append(f"Verified all downstream influences: {', '.join(verified_targets)}.")
                            else:
                                if verified_targets:
                                    evidence_parts.append(f"Verified: {', '.join(verified_targets)}.")
                                if contradicted_targets:
                                    evidence_parts.append(f"Contradicted: {', '.join(contradicted_targets)} not found.")

                        evidence = " ".join(evidence_parts)
                        
                        # Semantic check.
                        api_config = init_qwen_api()
                        if not api_config:
                            reasoning_verified = False
                            reasoning_evidence = "API key not configured."
                        else:
                            verification = _cached_verify_causal_reasoning(
                                api_config,
                                claim.get('claim_text', ''),
                                [central_interp],
                                formatted_interpretations,
                                central_feature_info=central_feature_str,
                                cache_version="faithfulness-2025-11-29"
                            )
                            reasoning_verified = verification.get('is_verified', False)
                            reasoning_evidence = verification.get('reasoning', 'Failed')

                        is_verified = all_verified and reasoning_verified
                        evidence += f" Reasoning check: {reasoning_evidence}"
                
                elif claim_type == 'subnetwork_purpose_claim':
                    purpose_summary = details.get('purpose_summary')
                    if not purpose_summary:
                        evidence = "Claim was missing a 'purpose_summary'."
                    else:
                        # Collect interpretations.
                        neighbors = analysis_data.get('subgraph_neighbors', {})
                        central_info = analysis_data.get('central_feature_info', {})
                        
                        subnetwork_interpretations = [central_info.get('interpretation', '')]
                        subnetwork_interpretations.extend([f.get('interpretation', '') for f in neighbors.get('upstream', [])])
                        subnetwork_interpretations.extend([f.get('interpretation', '') for f in neighbors.get('downstream', [])])
                        
                        api_config = init_qwen_api()
                        if not api_config:
                            evidence = "API key not configured for semantic verification."
                        else:
                            verification = _cached_verify_subnetwork_purpose(api_config, purpose_summary, subnetwork_interpretations, cache_version="faithfulness-2025-11-29")
                            is_verified = verification.get('is_verified', False)
                            evidence = verification.get('reasoning', "Failed to get semantic reasoning.")

            # Fallback/activation number check.
            if context == "circuit_graph" and layer_max_activations:
                activation_ok, activation_evidence = _check_activation_numbers(claim.get('claim_text', ''), layer_max_activations, layer_section_for_activation)
                if activation_ok:
                    if activation_evidence:
                        if not is_verified or evidence.startswith("Could not be verified") or evidence.lower().startswith("contradicted"):
                            evidence = activation_evidence.strip()
                            is_verified = True
                        else:
                            evidence = "{} {}".format(evidence, activation_evidence).strip()
                    else:
                        evidence = evidence.strip()
                else:
                    is_verified = False
                    evidence = activation_evidence if activation_evidence else evidence

        except Exception as e:
            evidence = f"An error occurred: {str(e)}"

        verification_results.append({'claim_text': claim.get('claim_text', 'N/A'), 'verified': is_verified, 'evidence': evidence})
        
    return verification_results

if __name__ == "__main__":
    # Standalone testing.
    st.set_page_config(
        page_title="Circuit Trace Explorer",
        page_icon="🔬",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    # Localization.
    from utilities.localization import initialize_localization
    initialize_localization()
    show_circuit_trace_page()