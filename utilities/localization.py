import streamlit as st
import json
from pathlib import Path
import os

# Set path to locales directory.
LOCALE_DIR = Path(__file__).parent.parent / "locales"

def load_language(lang_code):
    # Load all JSON language files for a given language.
    lang_dir = LOCALE_DIR / lang_code
    translations = {}
    
    if lang_dir.is_dir():
        for file_path in lang_dir.glob("*.json"):
            with open(file_path, "r", encoding="utf-8") as f:
                translations.update(json.load(f))
    
    # Fallback to English if no translations are found.
    if not translations:
        en_dir = LOCALE_DIR / "en"
        if en_dir.is_dir():
            for file_path in en_dir.glob("*.json"):
                with open(file_path, "r", encoding="utf-8") as f:
                    translations.update(json.load(f))
                    
    return translations

def initialize_localization():
    # Set up the session state for localization.
    if 'lang' not in st.session_state:
        st.session_state.lang = "en"
    
    if 'translations' not in st.session_state or st.session_state.get('lang_changed', False):
        st.session_state.translations = load_language(st.session_state.lang)
        st.session_state.lang_changed = False

def tr(key):
    # Translate a key using the loaded language file.
    return st.session_state.translations.get(key, key)

def language_selector():
    # Show a dropdown to select the language.
    languages = {"English": "en", "Deutsch": "de"}
    display_to_code = {name: code for name, code in languages.items()}
    code_to_display = {code: name for name, code in languages.items()}
    
    def on_change():
        selected_display_name = st.session_state.language_selector_key
        st.session_state.lang = display_to_code[selected_display_name]
        st.session_state.lang_changed = True

    current_lang_code = st.session_state.get('lang', 'en')
    
    try:
        current_index = list(languages.keys()).index(code_to_display[current_lang_code])
    except (KeyError, ValueError):
        current_index = 0

    st.selectbox(
        label="Language",
        options=list(languages.keys()),
        key="language_selector_key",
        on_change=on_change,
        format_func=lambda lang_name: f"🌐 {lang_name}",
        label_visibility="visible",
        index=current_index
    ) 