import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from embeddings_retriever import EmbeddingsRetriever
from sklearn.preprocessing import Normalizer
from sklearn.cluster import KMeans
import math
import wandb
import os

def process_sheet(sheet_name, df, models, model_names, num_runs):
    dsa_texts = list(df[['response', 'marks']].itertuples(index=False, name=None))
    texts = [text for text, _ in dsa_texts]
    original_marks = [mark for _, mark in dsa_texts]
    manual_marks = {text: mark for text, mark in dsa_texts}
    
    # max clusters = all texts
    max_num_clusters = len(texts) + 1
    
    all_variances = {}
    all_run_variances = {}
    
    for model, model_name in zip(models, model_names):
        
        # Retrieving Embeddings from model
        embeddings_retriever = EmbeddingsRetriever(model)
        embeddings_dict = embeddings_retriever.get_multiple_embeddings(texts)

        embeddings = []
        valid_texts = []
        for text in texts:
            emb = embeddings_dict.get(text, np.array([]))
            if emb.size == 0:
                print(f"Empty embedding for text: {text}")
            else:
                embeddings.append(emb)
                valid_texts.append(text)

        embeddings = np.array(embeddings)
        labels = [manual_marks[text] for text in valid_texts]
        max_num_clusters = min(max_num_clusters, len(valid_texts) + 1)
        
        # cosine distance normalisation
        normalizer = Normalizer(norm='l2')
        cos_embeddings = normalizer.fit_transform(embeddings)

        all_run_results = {n_clusters: [] for n_clusters in range(2, max_num_clusters)}
        
        # 10 runs averaging due to randomness in centrods for k-means
        for run in range(num_runs):
            print(f"Running iteration {run + 1} for model {model_name} on {sheet_name}")
            
            for num_clusters in range(2, max_num_clusters):
                # k-means clustering
                clustering = KMeans(n_clusters=num_clusters, random_state=run)
                predicted_labels = clustering.fit_predict(cos_embeddings)
                
                clusters = {}
                for cluster_id in range(num_clusters):
                    cluster_indices = np.where(predicted_labels == cluster_id)[0]
                    cluster_marks = [labels[idx] for idx in cluster_indices]
                    clusters[cluster_id] = cluster_marks

                cluster_variances = {}
                for cluster_id, cluster_marks in clusters.items():
                    if len(cluster_marks) == 0:
                        continue
                    # normalising variance using standard deviation
                    variance = np.std(cluster_marks)
                    if variance is not None:
                        cluster_variances[cluster_id] = variance

                if cluster_variances:
                    avg_variance = sum(cluster_variances.values()) / len(cluster_variances)
                    all_run_results[num_clusters].append(avg_variance)

        mean_variances = []
        run_variances = []
        
        for num_clusters in range(2, max_num_clusters):
            run_results = all_run_results[num_clusters]
            mean_variance = np.mean(run_results)
            variance_across_runs = np.var(run_results)
            
            mean_variances.append(mean_variance)
            run_variances.append(math.sqrt(variance_across_runs))
        
        all_variances[model_name] = mean_variances
        all_run_variances[model_name] = run_variances
    
    return all_variances, all_run_variances, max_num_clusters

def main():
    # wandb
    wandb.init(
        project="clustering-pretrained_model-analysis",
        config={
            "num_runs": 10
        }
    )
    
    num_runs = 10
    
    # All sheets to process
    file_paths = {
        'TheoryTest1-2022-Q1': './Phase-2/IDSA_data_final.xlsx',
        'Exam2020-Q1': './Phase-2/IDSA_data_final.xlsx',
        'Exam2020-Q2': './Phase-2/IDSA_data_final.xlsx',
        'Exam2022-Q2': './Phase-2/IDSA_data_final.xlsx',
        'Exam2022-Q3': './Phase-2/IDSA_data_final.xlsx',
        'Quiz5-2021-Q2': './Phase-2/IDSA_data_final.xlsx',
        'Quiz4-2021-Q1': './Phase-2/IDSA_data_final.xlsx',
        'Quiz3-2020-Q2': './Phase-2/IDSA_data_final.xlsx',
        'Quiz4-2020-Q1': './Phase-2/IDSA_data_final.xlsx',
        'Quiz3-2021-Q1': './Phase-2/IDSA_data_final.xlsx',
        'Quiz3-2020-Q1': './Phase-2/IDSA_data_final.xlsx',
        'Quiz4-2021-Q2': './Phase-2/IDSA_data_final.xlsx'
    }
    
    # for all different models - BERT, GPT-2, BERT fine-tuned, GPT-2 fine-tuned
    models = ['gpt2', 'bert-base-uncased', './Phase-2/gpt2_epoch_50', './Phase-2/bert_epoch_48']
    model_names = ['GPT-2', 'BERT', 'GPT-2 Fine-Tuned', 'BERT Fine-Tuned']
    
    # For each sheet ...
    for sheet_name, file_path in file_paths.items():
        print(f"Processing sheet: {sheet_name}")
        try:
            df = pd.read_excel(file_path, sheet_name=sheet_name)
            df['marks'] = df['marks'].round(0).astype(int)
            
            all_variances, all_run_variances, max_num_clusters = process_sheet(
                sheet_name, df, models, model_names, num_runs
            )
            
             # Plotting and Saving the graph
            plt.figure(figsize=(12, 8))
            for model_name in model_names:
                variances = all_variances[model_name]
                run_vars = all_run_variances[model_name]
                plt.errorbar(
                    range(2, max_num_clusters), 
                    variances, 
                    yerr=run_vars, 
                    label=model_name, 
                    capsize=3
                )

            plt.title(f'Average Variance of Marks for {sheet_name} with k-means clustering\n(Error bars show variance across {num_runs} runs)')
            plt.xlabel('Number of Clusters')
            plt.ylabel('Average Variance of Marks')
            plt.ylim(0, 40)
            plt.legend(loc='upper right')
            plot_path = f'plots_dynamic/{sheet_name}_variances.png'
            os.makedirs('plots_dynamic', exist_ok=True)
            plt.savefig(plot_path)
            
            # Logging to wandb
            wandb.log({
                f"{sheet_name}_plot": wandb.Image(plot_path),
                f"{sheet_name}_variances": {
                    model_name: variances 
                    for model_name, variances in all_variances.items()
                },
                f"{sheet_name}_run_variances": {
                    model_name: run_vars 
                    for model_name, run_vars in all_run_variances.items()
                }
            })
            
            plt.close()
            
        except Exception as e:
            print(f"Error processing sheet {sheet_name}: {str(e)}")
            continue

if __name__ == "__main__":
    main()