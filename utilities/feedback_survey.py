import streamlit as st
from utilities.localization import tr
from datetime import datetime
import pandas as pd
import os
import random

def get_next_participant_id():
    # Reads a counter file to get a unique ID for each participant.
    id_file = "user_study/data/participant_counter.txt"
    if not os.path.exists(id_file):
        with open(id_file, "w") as f:
            f.write("2")
        return 1
    else:
        with open(id_file, "r+") as f:
            current_id = int(f.read().strip())
            next_id = current_id + 1
            f.seek(0)
            f.truncate()
            f.write(str(next_id))
            return current_id

def save_feedback_to_session(page_key, data):
    # Saves feedback data to the current session.
    if 'session_feedback' not in st.session_state:
        st.session_state.session_feedback = {}
    st.session_state.session_feedback[page_key] = data

def write_session_feedback_to_csv():
    # Writes all feedback from the session to a CSV file.
    if 'session_feedback' in st.session_state:
        final_data = st.session_state.user_info.copy()
        
        # Handle old 'expertise' key for compatibility.
        if 'expertise' in final_data and 'llm_experience' not in final_data:
            final_data['llm_experience'] = final_data.pop('expertise')
        
        final_data["feedback_timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        final_data["participant_id"] = st.session_state.participant_id
        final_data["language"] = st.session_state.get("lang", "en")
        
        for page_data in st.session_state.session_feedback.values():
            final_data.update(page_data)
            
        df = pd.DataFrame([final_data])
        file_path = "user_study/data/user_data.csv"
        
        # Define all possible columns for the CSV.
        all_columns = [
            'participant_id', 'feedback_timestamp', 'language',
            'age', 'llm_experience',
            'attr_q_visual_clarity', 'attr_q_cognitive_load', 'attr_q_influencer_plausibility',
            'attr_s1_correct', 'attr_s2_correct', 'attr_s3_correct',
            'fv_q_pca_clarity', 'fv_q_type_attribution_clarity', 'fv_q_layer_evolution_plausibility',
            'fv_q1_correct', 'fv_q2_correct', 'fv_q3_correct',
            'ct_q_main_graph_clarity', 'ct_q_feature_explorer_usefulness', 'ct_q_subnetwork_clarity',
            'ct_q1_correct', 'ct_q2_correct', 'ct_q3_correct',
        ]
        
        # Make sure the dataframe has all columns.
        df = df.reindex(columns=all_columns)
        
        df.to_csv(file_path, mode='a', header=not os.path.exists(file_path), index=False)
        
        # Clear feedback after saving.
        if 'session_feedback' in st.session_state:
            del st.session_state.session_feedback


def display_attribution_feedback():
    # Shows the feedback form for the Attribution Analysis page.
    with st.sidebar:
        st.markdown("---")
        if st.session_state.get('attr_feedback_submitted', False):
            st.success(tr('feedback_success_message'))
            return

        with st.form("attribution_feedback_form"):
            st.header(tr('feedback_survey_header'))
            st.markdown(tr('feedback_survey_desc'))
            
            st.subheader(tr('ux_feedback_subheader'))
            visual_clarity = st.slider(tr('q_visual_clarity'), 1, 5, 3, help=tr('q_visual_clarity_help'))
            cognitive_load = st.slider(tr('q_cognitive_load'), 1, 5, 3, help=tr('q_cognitive_load_help'))
            influencer_plausibility = st.slider(tr('q_influential_docs_plausibility'), 1, 5, 3, help=tr('q_influential_docs_plausibility_help'))

            st.markdown("---")
            st.subheader(tr('comprehension_qs_subheader'))
            st.markdown(tr('comprehension_qs_desc'))
            
            # Questions for the feedback form.
            q_options = [tr('q_options_ig'), tr('q_options_occlusion'), tr('q_options_saliency')]
            questions_to_randomize = [
                {'key': 's1', 'text_key': 'q_s1', 'options': q_options.copy(), 'correct_answer': 'Saliency'},
                {'key': 's2', 'text_key': 'q_s2', 'options': q_options.copy(), 'correct_answer': 'Occlusion'},
                {'key': 's3', 'text_key': 'q_s3', 'options': q_options.copy(), 'correct_answer': 'Integrated Gradients'}
            ]

            # Shuffle questions and options once.
            if 'attr_question_order' not in st.session_state:
                question_order = questions_to_randomize.copy()
                for q in question_order:
                    random.shuffle(q['options'])
                st.session_state.attr_question_order = question_order

            responses = {}
            for q_info in st.session_state.attr_question_order:
                responses[q_info['key']] = st.radio(tr(q_info['text_key']), q_info['options'], key=q_info['key'], index=None)

            if st.form_submit_button(tr('submit_feedback_button')):
                if not all(responses.values()):
                    st.warning(tr('feedback_please_answer_all_qs'))
                else:
                    q_map = {
                        tr('q_options_ig'): "Integrated Gradients", 
                        tr('q_options_occlusion'): "Occlusion", 
                        tr('q_options_saliency'): "Saliency"
                    }
                    
                    data = {
                        'attr_q_visual_clarity': visual_clarity,
                        'attr_q_cognitive_load': cognitive_load,
                        'attr_q_influencer_plausibility': influencer_plausibility,
                    }

                    for q_key, response_text in responses.items():
                        original_q = next((q for q in st.session_state.attr_question_order if q['key'] == q_key), None)
                        if original_q:
                            response_mapped = q_map.get(response_text)
                            data[f'attr_{q_key}_correct'] = (response_mapped == original_q['correct_answer'])

                    save_feedback_to_session('attribution', data)
                    st.session_state.attr_feedback_submitted = True
                    del st.session_state.attr_question_order
                    st.rerun()

def display_circuit_trace_feedback():
    # Shows the feedback form for the Circuit Trace page.
    with st.sidebar:
        st.markdown("---")
        if st.session_state.get('ct_feedback_submitted', False):
            st.success(tr('feedback_success_message'))
            return

        with st.form("circuit_trace_feedback_form"):
            st.header(tr('feedback_survey_header'))
            st.markdown(tr('feedback_survey_desc'))
            
            st.subheader(tr('ux_feedback_subheader'))
            q_main_graph_clarity = st.slider(tr('ct_q_main_graph_clarity'), 1, 5, 3, help=tr('likert_scale_meaning'))
            q_feature_explorer_usefulness = st.slider(tr('ct_q_feature_explorer_usefulness'), 1, 5, 3, help=tr('likert_scale_meaning'))
            q_subnetwork_clarity = st.slider(tr('ct_q_subnetwork_clarity'), 1, 5, 3, help=tr('likert_scale_meaning'))

            st.markdown("---")
            st.subheader(tr('comprehension_qs_subheader'))
            st.markdown(tr('comprehension_qs_desc'))

            questions_to_randomize = [
                {'key': 'ct_q1', 'text_key': 'ct_q1', 'options': [tr('ct_q1_option_a'), tr('ct_q1_option_b'), tr('ct_q1_option_c')], 'correct_answer': tr('ct_q1_option_b')},
                {'key': 'ct_q2', 'text_key': 'ct_q2', 'options': [tr('ct_q2_option_a'), tr('ct_q2_option_b'), tr('ct_q2_option_c')], 'correct_answer': tr('ct_q2_option_b')},
                {'key': 'ct_q3', 'text_key': 'ct_q3', 'options': [tr('ct_q3_option_a'), tr('ct_q3_option_b'), tr('ct_q3_option_c')], 'correct_answer': tr('ct_q3_option_a')}
            ]
            
            if 'ct_question_order' not in st.session_state:
                # Shuffle questions and options once.
                question_order = questions_to_randomize.copy()
                for q in question_order:
                    random.shuffle(q['options'])
                st.session_state.ct_question_order = question_order

            responses = {}
            for q_info in st.session_state.ct_question_order:
                responses[q_info['key']] = st.radio(tr(q_info['text_key']), q_info['options'], key=q_info['key'], index=None)

            if st.form_submit_button(tr('submit_feedback_button')):
                if not all(responses.values()):
                    st.warning(tr('feedback_please_answer_all_qs'))
                else:
                    data = {
                        'ct_q_main_graph_clarity': q_main_graph_clarity,
                        'ct_q_feature_explorer_usefulness': q_feature_explorer_usefulness,
                        'ct_q_subnetwork_clarity': q_subnetwork_clarity,
                    }
                    
                    for q_key, response_text in responses.items():
                        original_q = next((q for q in st.session_state.ct_question_order if q['key'] == q_key), None)
                        if original_q:
                            data[f'{q_key}_correct'] = (response_text == original_q['correct_answer'])

                    save_feedback_to_session('circuit_trace', data)
                    st.session_state.ct_feedback_submitted = True
                    del st.session_state.ct_question_order
                    
                    # Check if all feedback has been submitted, then save.
                    if 'session_feedback' in st.session_state and len(st.session_state.session_feedback) >= 3:
                        write_session_feedback_to_csv()
                        
                    st.rerun()

def display_function_vector_feedback():
    # Shows the feedback form for the Function Vectors page.
    with st.sidebar:
        st.markdown("---")
        if st.session_state.get('fv_feedback_submitted', False):
            st.success(tr('feedback_success_message'))
            return

        with st.form("fv_feedback_form"):
            st.header(tr('feedback_survey_header'))
            st.markdown(tr('feedback_survey_desc'))
            
            st.subheader(tr('ux_feedback_subheader'))
            q1_pca_clarity = st.slider(tr('q1_pca_clarity'), 1, 5, 3, help=tr('likert_scale_meaning'))
            q2_type_attribution_clarity = st.slider(tr('q2_type_attribution_clarity'), 1, 5, 3, help=tr('likert_scale_meaning'))
            q_layer_evolution_plausibility = st.slider(tr('q_layer_evolution_plausibility'), 1, 5, 3, help=tr('likert_scale_meaning'))

            st.markdown("---")
            st.subheader(tr('comprehension_qs_subheader'))
            st.markdown(tr('comprehension_qs_desc'))

            questions_to_randomize = [
                {'key': 'fv_q1', 'text_key': 'fv_q1', 'options': [tr('fv_q1_option_c'), tr('fv_q1_option_b'), tr('fv_q1_option_a')], 'correct_answer': tr('fv_q1_option_c')},
                {'key': 'fv_q2', 'text_key': 'fv_q2', 'options': [tr('fv_q2_option_b'), tr('fv_q2_option_c'), tr('fv_q2_option_a')], 'correct_answer': tr('fv_q2_option_b')},
                {'key': 'fv_q3', 'text_key': 'fv_q3', 'options': [tr('fv_q3_option_c'), tr('fv_q3_option_a'), tr('fv_q3_option_d')], 'correct_answer': tr('fv_q3_option_c')}
            ]
            
            if 'fv_question_order' not in st.session_state:
                # Shuffle questions and options once.
                question_order = questions_to_randomize.copy()
                for q in question_order:
                    random.shuffle(q['options'])
                st.session_state.fv_question_order = question_order

            responses = {}
            for q_info in st.session_state.fv_question_order:
                # The options are already shuffled.
                responses[q_info['key']] = st.radio(tr(q_info['text_key']), q_info['options'], key=q_info['key'], index=None)

            if st.form_submit_button(tr('submit_feedback_button')):
                if not all(responses.values()):
                    st.warning(tr('feedback_please_answer_all_qs'))
                else:
                    data = {
                        'fv_q_pca_clarity': q1_pca_clarity,
                        'fv_q_type_attribution_clarity': q2_type_attribution_clarity,
                        'fv_q_layer_evolution_plausibility': q_layer_evolution_plausibility,
                    }
                    
                    for q_key, response_text in responses.items():
                        original_q = next((q for q in st.session_state.fv_question_order if q['key'] == q_key), None)
                        if original_q:
                            data[f'{q_key}_correct'] = (response_text == original_q['correct_answer'])

                    save_feedback_to_session('function_vector', data)
                    st.session_state.fv_feedback_submitted = True
                    del st.session_state.fv_question_order
                    
                    # If all forms are submitted, write to the CSV.
                    if 'session_feedback' in st.session_state and len(st.session_state.session_feedback) >= 3:
                        write_session_feedback_to_csv()
                        
                    st.rerun() 