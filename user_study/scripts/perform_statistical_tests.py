import pandas as pd
from scipy.stats import kruskal, mannwhitneyu, spearmanr
import os

def load_and_preprocess_data(filepath='user_study/data/user_data.csv'):
    # Loads and preprocesses the user study data.
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"The data file was not found at {filepath}")
    df = pd.read_csv(filepath)
    if 'attr_q_cognitive_load' in df.columns:
        df['attr_q_ease_of_use'] = 6 - df['attr_q_cognitive_load']
    return df

def run_ux_ratings_test(df):
    # Compares UX ratings across the three pages.
    print("\n--- 1. UX Ratings Comparison Across Pages ---")
    page_ratings = {
        'Attribution': df[['attr_q_visual_clarity', 'attr_q_ease_of_use', 'attr_q_influencer_plausibility']].mean(axis=1),
        'Function Vectors': df[['fv_q_pca_clarity', 'fv_q_type_attribution_clarity', 'fv_q_layer_evolution_plausibility']].mean(axis=1),
        'Circuit Trace': df[['ct_q_main_graph_clarity', 'ct_q_feature_explorer_usefulness', 'ct_q_subnetwork_clarity']].mean(axis=1)
    }
    
    attr_scores = page_ratings['Attribution'].dropna()
    fv_scores = page_ratings['Function Vectors'].dropna()
    ct_scores = page_ratings['Circuit Trace'].dropna()

    if len(attr_scores) > 0 and len(fv_scores) > 0 and len(ct_scores) > 0:
        stat, p = kruskal(attr_scores, fv_scores, ct_scores)
        print("Kruskal-Wallis test for overall UX ratings across the three pages:")
        print(f"H-statistic: {stat:.4f}, p-value: {p:.4f}")
        if p < 0.05:
            print("Result: There is a statistically significant difference in UX ratings between the pages.")
        else:
            print("Result: There is no statistically significant difference in UX ratings between the pages.")
    else:
        print("Could not perform Kruskal-Wallis test due to insufficient data.")

def run_language_comparison_test(df):
    # Compares ease of use for the Attribution page between English and German speakers.
    print("\n--- 2. Language Comparison for Attribution Page Ease of Use ---")
    en_df = df[df['language'] == 'en']
    de_df = df[df['language'] == 'de']
    
    en_scores = en_df['attr_q_ease_of_use'].dropna()
    de_scores = de_df['attr_q_ease_of_use'].dropna()

    if len(en_scores) > 0 and len(de_scores) > 0:
        stat, p = mannwhitneyu(en_scores, de_scores, alternative='two-sided')
        print("Mann-Whitney U test for 'Ease of Use' on Attribution page (English vs. German):")
        print(f"U-statistic: {stat:.4f}, p-value: {p:.4f}")
        if p < 0.05:
            print("Result: There is a statistically significant difference between the language groups.")
        else:
            print("Result: There is no statistically significant difference between the language groups.")
    else:
        print("Could not perform Mann-Whitney U test due to insufficient data.")

def run_experience_correctness_test(df):
    # Tests for a correlation between LLM experience and correctness.
    print("\n--- 3. Correlation between LLM Experience and Comprehension Correctness ---")
    
    experience_map = {'novice': 1, 'intermediate': 2, 'expert': 3}
    df['llm_experience_ordinal'] = df['llm_experience'].map(experience_map)
    
    correct_cols = [col for col in df.columns if 'correct' in col]
    df['overall_correctness'] = df[correct_cols].mean(axis=1)
    
    corr_df = df[['llm_experience_ordinal', 'overall_correctness']].dropna()
    if not corr_df.empty:
        corr, p = spearmanr(corr_df['llm_experience_ordinal'], corr_df['overall_correctness'])
        print("Spearman correlation between LLM experience and overall comprehension correctness:")
        print(f"Rho: {corr:.4f}, p-value: {p:.4f}")
        if p < 0.05:
            print("Result: There is a statistically significant correlation.")
        else:
            print("Result: There is no statistically significant correlation.")
    else:
        print("Could not perform Spearman correlation due to insufficient data.")

if __name__ == '__main__':
    try:
        data = load_and_preprocess_data('../../user_study/data/user_data.csv')
        run_ux_ratings_test(data)
        run_language_comparison_test(data)
        run_experience_correctness_test(data)
    except Exception as e:
        print(f"An error occurred: {e}") 