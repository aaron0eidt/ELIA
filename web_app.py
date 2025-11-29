import streamlit as st
from streamlit_option_menu import option_menu
import os
import sys
import base64
from pathlib import Path

# Import page modules.
from attribution_analysis.attribution_analysis_page import show_attribution_analysis
from function_vectors.function_vectors_page import show_function_vectors_page
from circuit_analysis.circuit_trace_page import show_circuit_trace_page
from utilities.welcome_page import show_welcome_page
from utilities.utils import set_seed
from utilities.localization import initialize_localization, tr, language_selector
from utilities.feedback_survey import get_next_participant_id

# Import cached functions.
from attribution_analysis.attribution_analysis_page import (
    get_influential_docs, 
    _cached_explain_heatmap as attr_explain_heatmap, 
    generate_all_attribution_analyses
)
from circuit_analysis.circuit_trace_page import explain_circuit_visualization
from function_vectors.function_vectors_page import (
    _perform_analysis as fv_perform_analysis, 
    _explain_with_llm as fv_explain_llm
)


# Page configuration.
st.set_page_config(
    page_title="LLM Analysis Suite",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Disable tokenizers parallelism.
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Suppress macOS error.
os.environ["OBJC_DISABLE_INITIALIZE_FORK_SAFETY"] = "YES"


# Custom CSS.
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #2f3f70;
        text-align: center;
        margin-bottom: 2rem;
    }
    .stButton > button {
        background-color: #2f3f70;
        color: #f5f7fb;
        border-radius: 20px;
        border: none;
        padding: 0.5rem 2rem;
        font-weight: bold;
        box-shadow: 0 10px 20px rgba(47, 63, 112, 0.25);
    }
    .stButton > button:hover {
        background-color: #3a4c86;
        color: #ffffff;
    }
    .stTextArea > div > div > textarea {
        border-radius: 10px;
    }
    .attribution-info {
        background-color: rgba(47, 63, 112, 0.82);
        color: #f5f7fb;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
        border-left: 4px solid #dcae36;
    }
</style>
""", unsafe_allow_html=True)

def main():
    # Main app function.
    set_seed()
    initialize_localization()

    # Language selector is on welcome page.

    # Check welcome form.
    if "user_info" not in st.session_state or not st.session_state.user_info.get("form_submitted"):
        show_welcome_page()
    else:
        # Show main app.
        
        # Participant ID.
        if 'participant_id' not in st.session_state:
            st.session_state.participant_id = get_next_participant_id()
            
        # Feedback session state.
        if 'attr_feedback_submitted' not in st.session_state:
            st.session_state.attr_feedback_submitted = False
        if 'fv_feedback_submitted' not in st.session_state:
            st.session_state.fv_feedback_submitted = False

        st.set_page_config(
            page_title="LLM Analysis Suite",
            page_icon="🧠",
            layout="wide",
            initial_sidebar_state="expanded"
        )

        logo_path = Path(__file__).parent / "LOGO" / "Logo.png"
        if logo_path.exists():
            with open(logo_path, "rb") as logo_file:
                logo_base64 = base64.b64encode(logo_file.read()).decode("utf-8")
            st.markdown(
                f"""
                <div style="text-align: center; margin-bottom: 2rem;">
                    <img src="data:image/png;base64,{logo_base64}" alt="{tr('llm_analysis_suite')}" style="max-width: 320px; width: 60%; min-width: 200px;" />
                </div>
                """,
                unsafe_allow_html=True
            )
        else:
            st.markdown(f"<h1 class='main-header'>{tr('llm_analysis_suite')}</h1>", unsafe_allow_html=True)
        
        with st.sidebar:
            selected_page = option_menu(
                menu_title=tr('main_menu'),
                options=[tr('attribution_analysis'), tr('function_vectors'), tr('circuit_tracing')],
                icons=['search', 'cpu', 'diagram-3'],
                menu_icon='cast',
                default_index=0
            )

            

        if selected_page == tr('attribution_analysis'):
            show_attribution_analysis()
        elif selected_page == tr('function_vectors'):
            show_function_vectors_page()
        elif selected_page == tr('circuit_tracing'):
            show_circuit_trace_page()

if __name__ == "__main__":
    main() 