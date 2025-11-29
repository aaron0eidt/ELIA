import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import os

def load_and_preprocess_data(filepath='user_study/data/user_data.csv'):
    # Loads and preprocesses the user study data.
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"The data file was not found at {filepath}")
    
    df = pd.read_csv(filepath)
    
    # Invert Cognitive Load to make it more intuitive.
    for col in ['attr_q_cognitive_load']:
        if col in df.columns:
            df[col.replace('cognitive_load', 'ease_of_use')] = 6 - df[col]
            df.drop(columns=[col], inplace=True)
            
    return df

def plot_user_demographics(df, output_dir='writing/Simplifying_Outcomes_of_Language_Model_Component_Analyses/figures/results'):
    # Plots user demographic data.
    os.makedirs(output_dir, exist_ok=True)
    
    # Configure the plot style.
    sns.set_theme(style="ticks", palette="viridis")
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = 'Arial'
    plt.rcParams['axes.labelweight'] = 'normal'
    plt.rcParams['axes.titleweight'] = 'bold'
    plt.rcParams['figure.titleweight'] = 'bold'
    plt.rcParams['savefig.dpi'] = 300
    plt.rcParams['figure.facecolor'] = 'white'
    plt.rcParams['axes.facecolor'] = 'white'
    plt.rcParams['grid.alpha'] = 0.2
    plt.rcParams['axes.spines.top'] = False
    plt.rcParams['axes.spines.right'] = False

    # Plot age distribution.
    plt.figure(figsize=(8, 6))
    age_order = ['under_18', '18_24', '25_34', '35_44', '55_64']
    ax = sns.countplot(data=df, x='age', order=age_order, palette="colorblind", hue='age', legend=False)
    plt.xlabel('Age Group', fontsize=18)
    plt.ylabel('Number of Participants', fontsize=18)
    plt.xticks(rotation=45, fontsize=14)
    # Set y-axis ticks to integers.
    ax.set_yticks(np.arange(0, df['age'].value_counts().max() + 1, 1))
    ax.tick_params(axis='y', labelsize=12)
    sns.despine()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'user_demographics_age.png'))
    plt.close()

    # Plot LLM experience.
    plt.figure(figsize=(8, 6))
    exp_order = ['novice', 'intermediate', 'expert']
    ax = sns.countplot(data=df, x='llm_experience', order=exp_order, palette="colorblind", hue='llm_experience', legend=False)
    plt.xlabel('Experience Level', fontsize=16)
    plt.ylabel('Number of Participants', fontsize=16)
    plt.xticks(fontsize=14)
    # Set y-axis ticks to integers.
    ax.set_yticks(np.arange(0, df['llm_experience'].value_counts().max() + 1, 1))
    ax.tick_params(axis='y', labelsize=14)
    sns.despine()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'user_demographics_experience.png'))
    plt.close()

def plot_ux_ratings_by_page(df, language_filter='all'):
    # Plots UX ratings, with an option to filter by language.
    if language_filter != 'all':
        df = df[df['language'] == language_filter]
        
    if df.empty:
        print(f"No data for language filter: {language_filter}. Skipping plot.")
        return

    ux_metrics = {
        'Attribution': ['Visual Clarity', 'Ease of Use', 'Influencer Plausibility'],
        'Function Vectors': ['Pca Clarity', 'Type Attribution Clarity', 'Layer Evolution Plausibility'],
        'Circuit Trace': ['Main Graph Clarity', 'Feature Explorer Usefulness', 'Subnetwork Clarity']
    }

    # Map the clean metric names to the dataframe column names.
    column_mapping = {
        'Visual Clarity': 'attr_q_visual_clarity',
        'Ease of Use': 'attr_q_ease_of_use',
        'Influencer Plausibility': 'attr_q_influencer_plausibility',
        'Pca Clarity': 'fv_q_pca_clarity',
        'Type Attribution Clarity': 'fv_q_type_attribution_clarity',
        'Layer Evolution Plausibility': 'fv_q_layer_evolution_plausibility',
        'Main Graph Clarity': 'ct_q_main_graph_clarity',
        'Feature Explorer Usefulness': 'ct_q_feature_explorer_usefulness',
        'Subnetwork Clarity': 'ct_q_subnetwork_clarity'
    }
    
    df_melted = pd.DataFrame()
    for page, cols in ux_metrics.items():
        for col_clean_name in cols:
            col_original_name = column_mapping[col_clean_name]
            if col_original_name in df.columns:
                temp_df = df[[col_original_name]].copy()
                temp_df.rename(columns={col_original_name: 'Rating'}, inplace=True)
                temp_df['Page'] = page
                temp_df['Metric'] = col_clean_name
                df_melted = pd.concat([df_melted, temp_df], ignore_index=True)

    plt.figure(figsize=(14, 8))
    sns.boxplot(data=df_melted, x='Metric', y='Rating', hue='Page', palette='colorblind', fliersize=0)
    plt.xlabel('UX Metric', fontsize=14)
    plt.ylabel('Rating (1-5)', fontsize=14)
    plt.xticks(rotation=15, fontsize=12)
    plt.legend(title='Analysis Page', fontsize=12)
    plt.yticks(np.arange(1, 6, 1), fontsize=12)
    sns.despine()
    plt.tight_layout()
    
    # Save the figure with a language-specific name.
    output_path = os.path.join('writing/Simplifying_Outcomes_of_Language_Model_Component_Analyses/figures/results', f'ux_ratings_by_page_{language_filter}.png')
    plt.savefig(output_path)
    print(f"Saved UX ratings plot to {output_path}")
    plt.close()

def plot_correctness_by_experience(df, output_dir='writing/Simplifying_Outcomes_of_Language_Model_Component_Analyses/figures/results'):
    # Plots comprehension correctness by LLM experience.
    os.makedirs(output_dir, exist_ok=True)
    
    correct_cols = [col for col in df.columns if 'correct' in col]
    df_corr = df[['llm_experience'] + correct_cols].copy()
    
    df_melted = df_corr.melt(id_vars=['llm_experience'], var_name='Question', value_name='Is Correct')
    df_melted['Is Correct'] = df_melted['Is Correct'].astype(float)
    
    plt.figure(figsize=(10, 6))
    ax = sns.barplot(data=df_melted, x='llm_experience', y='Is Correct', order=['novice', 'intermediate', 'expert'], palette='colorblind', hue='llm_experience', legend=False, errorbar=None)
    plt.xlabel('Experience Level', fontsize=16)
    plt.ylabel('Proportion Correct', fontsize=16)
    plt.ylim(0, 1)
    plt.yticks(np.arange(0, 1.1, 0.1))
    ax.tick_params(axis='x', labelsize=14)
    ax.tick_params(axis='y', labelsize=14)
    sns.despine()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'correctness_by_experience.png'))
    plt.close()

def plot_correlation_heatmap(df, output_dir='writing/Simplifying_Outcomes_of_Language_Model_Component_Analyses/figures/results'):
    # Plots a correlation heatmap of all numerical data.
    os.makedirs(output_dir, exist_ok=True)
    
    quant_cols = df.select_dtypes(include=np.number).columns.tolist()
    # Remove participant ID from the correlation.
    if 'participant_id' in quant_cols:
        quant_cols.remove('participant_id')
        
    corr = df[quant_cols].corr()
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", linewidths=.5)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'correlation_heatmap.png'))
    plt.close()

if __name__ == '__main__':
    try:
        data = load_and_preprocess_data('../../user_study/data/user_data.csv')
        plot_user_demographics(data)
        
        # Generate all three versions of the UX ratings plot.
        plot_ux_ratings_by_page(data.copy(), language_filter='all')
        plot_ux_ratings_by_page(data.copy(), language_filter='en')
        plot_ux_ratings_by_page(data.copy(), language_filter='de')
        
        plot_correctness_by_experience(data)
        plot_correlation_heatmap(data)
        print("All plots generated successfully.")
    except Exception as e:
        print(f"An error occurred: {e}") 