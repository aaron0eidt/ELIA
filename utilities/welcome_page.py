import base64
from pathlib import Path
from datetime import datetime
import uuid

import streamlit as st

from utilities.localization import tr, language_selector

def show_welcome_page():
    # Renders the welcome page with a user info form.
    
    # Language selection dropdown.
    #language_selector()
    st.markdown("---")

    # Header and introduction.
    logo_path = Path(__file__).resolve().parent.parent / "LOGO" / "Logo.png"
    logo_html = ""
    if logo_path.exists():
        with open(logo_path, "rb") as logo_file:
            logo_base64 = base64.b64encode(logo_file.read()).decode("utf-8")
        logo_html = (
            f"<img src='data:image/png;base64,{logo_base64}' alt='{tr('llm_analysis_suite')}' "
            "style='max-width: 280px; width: 55%; min-width: 180px; margin-bottom: 1.5rem;'/>"
        )

    st.markdown(
        f"""
        <div style="background: radial-gradient(circle at top, #1c2947, #0b1321); color: #f9fafb; padding: 2.75rem; border-radius: 14px; text-align: center; border: 1px solid rgba(220,174,54,0.25); box-shadow: 0 24px 45px rgba(8, 13, 24, 0.55);">
            {logo_html}
            <h1 style="color: #f9fafb; font-size: 2.2rem; margin-bottom: 0.75rem;">{tr('welcome_to_llm_analysis_suite')}</h1>
            <p style="font-size: 1.15rem; margin: 0 auto; max-width: 640px; color: #dce3f2;">{tr('research_tool_intro')}</p>
        </div>
        """,
        unsafe_allow_html=True
    )

    # st.markdown(f"""
    # ### {tr('about_this_tool')}
    # {tr('research_study_info')}

    # **{tr('your_role')}**
    # - {tr('role_1')}
    # - {tr('role_2')}
    # - {tr('role_3')}

    # **{tr('data_privacy')}**
    # - {tr('privacy_1')}
    # - {tr('privacy_2')}
    # - {tr('privacy_3')}

    # ---
    # """, unsafe_allow_html=True)
    
    # Don't show the form if it has already been submitted.
    if 'user_info' in st.session_state and st.session_state.user_info.get("form_submitted"):
        st.success(tr('thank_you_proceed'))
        return

    # st.header(tr('tell_us_about_yourself'))
    
    with st.form("user_data_form"):
        # age_options = ["under_18", "18_24", "25_34", "35_44", "45_54", "55_64", "65_or_over", "prefer_not_to_say"]
        # age_group = st.selectbox(
        #     tr('what_is_your_age_group'),
        #     options=age_options,
        #     format_func=lambda x: tr(x),
        #     index=2
        # )
        
        # expertise_options = ["novice", "intermediate", "expert"]
        # expertise = st.selectbox(
        #     tr('rate_your_expertise'),
        #     options=expertise_options,
        #     format_func=lambda x: tr(x),
        #     index=1
        # )
        
        # Set default values
        age_group = "25_34"
        expertise = "intermediate"
        
        submitted = st.form_submit_button(tr('start_analysis_button'), use_container_width=True)
        
        if submitted:
            # Generate a unique session ID.
            session_id = str(uuid.uuid4())
            
            # Store user data in the session.
            st.session_state.user_info = {
                "id": session_id,
                "age": age_group,
                "llm_experience": expertise,
                "form_submitted": True,
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
            
            st.success(tr('thank_you_main_suite'))
            st.rerun()

if __name__ == "__main__":
    show_welcome_page() 