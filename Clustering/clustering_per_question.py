import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from embeddings_retriever import EmbeddingsRetriever
from sklearn.preprocessing import Normalizer
from sklearn.cluster import KMeans, AgglomerativeClustering


# Maximum number of clusters for the experiment and Number of times I run each configuration for averaging
max_num_clusters = 50
num_runs = 10  

# different sheet names with file_paths
file_paths = {
    'TheoryTest1-2022-Q1': './Phase-2/IDSA_data_final.xlsx',
    #'TheoryTest1-2022-Q2': './Phase-2/IDSA_data_final.xlsx',
    'Exam2020-Q2': './Phase-2/IDSA_data_final.xlsx',
    #'Exam2020-Q1': './Phase-2/IDSA_data_final.xlsx',
    'Exam2022-Q1': './Phase-2/IDSA_data_final.xlsx',
    #'Exam2022-Q2': './Phase-2/IDSA_data_final.xlsx',
    #'Exam2022-Q3': './Phase-2/IDSA_data_final.xlsx',
    'Quiz5-2021-Q1': './Phase-2/IDSA_data_final.xlsx',
    #'Quiz5-2021-Q2': './Phase-2/IDSA_data_final.xlsx',
    'Quiz4-2021-Q1': './Phase-2/IDSA_data_final.xlsx',
    #'Quiz4-2021-Q2': './Phase-2/IDSA_data_final.xlsx',
    'Quiz3-2021-Q1': './Phase-2/IDSA_data_final.xlsx',
    'Quiz3-2020-Q1': './Phase-2/IDSA_data_final.xlsx',
    #'Quiz3-2020-Q2': './Phase-2/IDSA_data_final.xlsx',
    'Quiz4-2020-Q1': './Phase-2/IDSA_data_final.xlsx',
    #'Quiz4-2020-Q2': './Phase-2/IDSA_data_final.xlsx'
}

# Defining models and clustering algorithms
models = ['gpt2', 'bert-base-uncased', './Phase-2/gpt2_epoch_50', './Phase-2/bert_epoch_28']
model_names = ['GPT-2', 'BERT', 'GPT-2 Fine-Tuned', 'BERT Fine-Tuned']
clustering_algorithms = {
    'KMeans': lambda n: KMeans(n_clusters=n, random_state=None),        # undeterministic
    'Hierarchical': lambda n: AgglomerativeClustering(n_clusters=n)     # deterministic
}

# For each model, and for each clustering algorithm, ...
for model, model_name in zip(models, model_names):
    for clust_name, clust_func in clustering_algorithms.items():
        plt.figure(figsize=(15, 8))
        all_avg_variances = {}
        all_run_variances = {}
        
        # for each question in file_path, ...
        for sheet_name, file_path in file_paths.items():
            
            # read the sheet from excel file and round off the marks and store texts and marks
            df = pd.read_excel(file_path, sheet_name=sheet_name)
            df['marks'] = df['marks'].round(0).astype(int)
            dsa_texts = list(df[['response', 'marks']].itertuples(index=False, name=None))

            texts = [text for text, _ in dsa_texts]
            original_marks = [mark for _, mark in dsa_texts]
            manual_marks = {text: mark for text, mark in dsa_texts}


            # retrieve embeddings from model
            embeddings_retriever = EmbeddingsRetriever(model)
            embeddings_dict = embeddings_retriever.get_multiple_embeddings(texts)

            embeddings = []
            valid_texts = []
            for text in texts:
                emb = embeddings_dict.get(text, np.array([]))
                if emb.size > 0:
                    embeddings.append(emb)
                    valid_texts.append(text)

            embeddings = np.array(embeddings)
            labels = [manual_marks[text] for text in valid_texts]
            
            # cosine similarity normalisation
            normalizer = Normalizer(norm='l2')
            cos_embeddings = normalizer.fit_transform(embeddings)
            
            
            Average_variance = []
            Run_variance = []
            
            # for each num_cluster value from 3 to max_num_clusters, ... 
            for num_clusters in range(3, max_num_clusters):
                run_variances = []
                
                # for each run of the clustering, ...
                for run in range(num_runs):
                    
                    # use clustering algorithm
                    clustering = clust_func(num_clusters)
                    predicted_labels = clustering.fit_predict(cos_embeddings)
                    
                    clusters = {}
                    for cluster_id in range(num_clusters):
                        cluster_indices = np.where(predicted_labels == cluster_id)[0]
                        cluster_marks = [labels[idx] for idx in cluster_indices]
                        clusters[cluster_id] = cluster_marks

                    cluster_variances = {}
                    for cluster_id, cluster_marks in clusters.items():
                        if len(cluster_marks) > 0:
                            variance = np.std(cluster_marks)
                            if variance is not None:
                                cluster_variances[cluster_id] = variance

                    if cluster_variances:
                        total_variance = sum(cluster_variances.values())
                        avg_variance = total_variance / len(cluster_variances)
                        run_variances.append(avg_variance)
                
                # mean and standard deviation across runs
                mean_variance_for_num_cluster = np.mean(run_variances)
                variance_varinace_for_num_cluster = np.std(run_variances)
                
                Average_variance.append(mean_variance_for_num_cluster)
                Run_variance.append(variance_varinace_for_num_cluster)
            
            all_avg_variances[sheet_name] = Average_variance
            all_run_variances[sheet_name] = Run_variance

        # Plotting
        colors = plt.cm.tab20(np.linspace(0, 1, len(all_avg_variances)))
        for (sheet_name, variances), color in zip(all_avg_variances.items(), colors):
            error = all_run_variances[sheet_name]
            plt.errorbar(range(3, max_num_clusters), variances, yerr=error, 
                        label=sheet_name, color=color, capsize=3)

        plt.title(f'Average Variance per num_clusters for {model_name} using {clust_name}\n(Error bars show std dev across {num_runs} runs)')
        plt.xlabel('Number of Clusters')
        plt.ylabel('Average Variance')
        plt.ylim(0, 40)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='small')
        plt.tight_layout()
        plt.savefig(f'{model_name.lower().replace(" ", "_")}_{clust_name.lower()}_variances.png',
                    bbox_inches='tight', dpi=300)
        plt.close()