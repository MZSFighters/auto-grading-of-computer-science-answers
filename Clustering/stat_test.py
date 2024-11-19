import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.stats.anova import anova_lm
from statsmodels.formula.api import ols
import warnings
warnings.filterwarnings('ignore')

# cleaning and preparing data for the statistical tests
def clean_data(data):
    data = data.dropna(subset=['variance', 'model_type', 'model_version'])
    
    # making sure all categorical variables are strings
    data['model_type'] = data['model_type'].astype(str)
    data['model_version'] = data['model_version'].astype(str)
    data['clustering_method'] = data['clustering_method'].astype(str)
    data['variance'] = pd.to_numeric(data['variance'], errors='coerce')
    data = data.dropna(subset=['variance'])
    return data

# t-test
def paired_test_models(df, model_name):
    try:
        base = df[(df['model_type'] == model_name) & 
                 (df['model_version'] == 'base')]['variance'].values
        finetuned = df[(df['model_type'] == model_name) & 
                      (df['model_version'] == 'finetuned')]['variance'].values
        
        # checking if we have equal number of samples
        min_len = min(len(base), len(finetuned))
        if min_len == 0:
            return np.nan, np.nan, 0
        
        # making them equal lengths if necessary
        base = base[:min_len]
        finetuned = finetuned[:min_len]
        
        # t-test and cohen's d
        t_stat, p_val = stats.ttest_rel(base, finetuned)
        d = (np.mean(finetuned) - np.mean(base)) / np.std(base)
        return t_stat, p_val, d
    except Exception as e:
        print(f"Error in paired test for {model_name}: {str(e)}")
        return np.nan, np.nan, 0

# anova testing
def perform_anova(data, clustering_method):
    try:
        # preparing data ...
        anova_data = data[data['clustering_method'] == clustering_method].copy()
        if len(anova_data) < 3:
            print(f"Insufficient data for ANOVA analysis in {clustering_method} clustering")
            return None

        # anova model and table
        model = ols('variance ~ C(model_type) + C(model_version) + C(model_type):C(model_version)', 
                   data=anova_data).fit()
        anova_table = anova_lm(model, typ=2)
        
        return anova_table
    except Exception as e:
        print(f"Error in ANOVA for {clustering_method}: {str(e)}")
        return None

# fucntion for statistical testing
def perform_statistical_analysis(data):
 
    print("Starting statistical analysis...")

    data = clean_data(data)
    
    # Descriptive Statistics
    print("\nDescriptive Statistics:")
    desc_stats = data.groupby(['clustering_method', 'model_type', 'model_version'])['variance'].agg([
        'count', 'mean', 'std', 'min', 'max'
    ]).round(3)
    print(desc_stats)
    
    # paired t-tests and effect sizes for each clustering method
    for clustering in data['clustering_method'].unique():
        cluster_data = data[data['clustering_method'] == clustering]
        
        print(f"\nResults for {clustering} clustering:")
        
        # testing BERT base vs fine-tuned
        t_bert, p_bert, d_bert = paired_test_models(cluster_data, 'BERT')
        print(f"BERT base vs fine-tuned: t={t_bert:.3f}, p={p_bert:.4f}, Cohen's d={d_bert:.3f}")
        
        # testing GPT-2 base vs fine-tuned
        t_gpt, p_gpt, d_gpt = paired_test_models(cluster_data, 'GPT-2')
        print(f"GPT-2 base vs fine-tuned: t={t_gpt:.3f}, p={p_gpt:.4f}, Cohen's d={d_gpt:.3f}")
        
        # adding Wilcoxon signed-rank test as a non-parametric alternative
        print("\nWilcoxon signed-rank test results:")
        try:
            bert_stat, bert_p = stats.wilcoxon(
                cluster_data[(cluster_data['model_type'] == 'BERT') & 
                           (cluster_data['model_version'] == 'base')]['variance'],
                cluster_data[(cluster_data['model_type'] == 'BERT') & 
                           (cluster_data['model_version'] == 'finetuned')]['variance']
            )
            print(f"BERT Wilcoxon test: statistic={bert_stat:.3f}, p={bert_p:.4f}")
        except Exception as e:
            print(f"Could not perform Wilcoxon test for BERT: {str(e)}")
        
        try:
            gpt_stat, gpt_p = stats.wilcoxon(
                cluster_data[(cluster_data['model_type'] == 'GPT-2') & 
                           (cluster_data['model_version'] == 'base')]['variance'],
                cluster_data[(cluster_data['model_type'] == 'GPT-2') & 
                           (cluster_data['model_version'] == 'finetuned')]['variance']
            )
            print(f"GPT-2 Wilcoxon test: statistic={gpt_stat:.3f}, p={gpt_p:.4f}")
        except Exception as e:
            print(f"Could not perform Wilcoxon test for GPT-2: {str(e)}")
        
        # 3. Two-way ANOVA
        anova_table = perform_anova(data, clustering)
        if anova_table is not None:
            print(f"\nTwo-way ANOVA results for {clustering} clustering:")
            print(anova_table)
    
    plt.figure(figsize=(12, 6))
    sns.boxplot(data=data, x='model_type', y='variance', hue='model_version')
    plt.title('Variance Distribution by Model Type and Version')
    plt.savefig('variance_distribution.png')
    plt.close()

if __name__ == "__main__":
    try:
        data = pd.read_csv('clustering_results.csv')
        perform_statistical_analysis(data)
    except Exception as e:
        print(f"Error reading data: {str(e)}")