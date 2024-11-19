import numpy as np
import pandas as pd
from sklearn.preprocessing import Normalizer
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors
from scipy.stats import pearsonr
from sklearn.metrics import cohen_kappa_score
import matplotlib.pyplot as plt
from embeddings_retriever import EmbeddingsRetriever
import os
from tqdm import tqdm

def predict_marks(df, marked_indices, embeddings, knn, cluster_labels):

    predictions = np.zeros(len(df))
    true_marks = df['marks'].values
    
    # For marked submissions, use their actual marks
    for idx in marked_indices:
        predictions[idx] = true_marks[idx]
    
    # For unmarked submissions, predict based on cluster means
    unmarked = set(range(len(df))) - marked_indices
    for cluster in np.unique(cluster_labels):
        cluster_mask = cluster_labels == cluster
        cluster_marked = [i for i in marked_indices if cluster_labels[i] == cluster]
        
        if cluster_marked:
            # If cluster has marked submissions, use cluster mean
            cluster_mean = df.iloc[cluster_marked]['marks'].mean()
            for idx in unmarked:
                if cluster_mask[idx]:
                    predictions[idx] = cluster_mean
        else:
            # If no marked submissions in cluster, use kNN
            for idx in unmarked:
                if cluster_mask[idx]:
                    distances, indices = knn.kneighbors([embeddings[idx]])
                    marked_neighbors = [i for i in indices[0] if i in marked_indices]
                    if marked_neighbors:
                        weights = 1 / (distances[0][:len(marked_neighbors)] + 1e-6)
                        neighbor_marks = df.iloc[marked_neighbors]['marks'].values
                        predictions[idx] = np.average(neighbor_marks, weights=weights)
                    else:
                        predictions[idx] = df.iloc[list(marked_indices)]['marks'].mean()
    
    return predictions

# single run code
def run_single_experiment(df, embeddings, n_clusters):

    # normalise embeddings
    normalizer = Normalizer(norm='l2')
    embeddings = normalizer.fit_transform(embeddings)
    
    # KMeans and kNN
    kmeans = KMeans(n_clusters=n_clusters, random_state=None)
    cluster_labels = kmeans.fit_predict(embeddings)
    knn = NearestNeighbors(n_neighbors=3, metric='cosine')
    knn.fit(embeddings)
    
    marked_indices = set()
    num_submissions = len(df)
    
    pearson_scores = []
    quadratic_kappa_scores = []
    num_marked = []
    
    # incrementally marking submissions and calculating metrics
    for i in range(num_submissions):
        unmarked = list(set(range(num_submissions)) - marked_indices)
        if not unmarked or len(unmarked) <= 2:
            break
        
        selected_idx = np.random.choice(unmarked)
        marked_indices.add(selected_idx)
        
        # predicting the marks for unmarked submsisions
        predicted_marks = predict_marks(df, marked_indices, embeddings, knn, cluster_labels)
        
        # Calculating metrics
        unmarked_mask = [idx for idx in range(num_submissions) if idx not in marked_indices]
        true_marks_unmarked = df['marks'].iloc[unmarked_mask]
        predicted_marks_unmarked = predicted_marks[unmarked_mask]
        
        pearson = pearsonr(true_marks_unmarked, predicted_marks_unmarked)[0]
        quadratic_kappa = cohen_kappa_score(
            true_marks_unmarked.round(),
            predicted_marks_unmarked.round(),
            weights='quadratic'
        )
        
        pearson_scores.append(pearson)
        quadratic_kappa_scores.append(quadratic_kappa)
        num_marked.append(len(marked_indices))
    
    return np.array(num_marked), np.array(pearson_scores), np.array(quadratic_kappa_scores)

def run_multiple_experiments(sheet_name, embeddings_retriever, n_runs=10):
    cluster_numbers = [20, 30, 50, 80, 100]
    results = {n: {'pearson': [], 'kappa': [], 'num_marked': None} for n in cluster_numbers}
    
    # Load data from sheet
    df = pd.read_excel(file_path, sheet_name=sheet_name)
    df['marks'] = df['marks'].round(0).astype(int)
    
    # processing embeddings
    texts = list(df['response'])
    embeddings_dict = embeddings_retriever.get_multiple_embeddings(texts)
    
    embeddings = []
    valid_texts = []
    for text in texts:
        emb = embeddings_dict.get(text, np.array([]))
        if emb.size > 0:
            embeddings.append(emb)
            valid_texts.append(text)
    
    embeddings = np.array(embeddings)
    valid_df = df[df['response'].isin(valid_texts)].reset_index(drop=True)
    
    for n_clusters in tqdm(cluster_numbers, desc=f"Processing {sheet_name}"):
        # for 10 runs
        for run in range(n_runs):
            num_marked, pearson_scores, kappa_scores = run_single_experiment(
                valid_df, embeddings, n_clusters
            )
            
            results[n_clusters]['pearson'].append(pearson_scores)
            results[n_clusters]['kappa'].append(kappa_scores)
            if results[n_clusters]['num_marked'] is None:
                results[n_clusters]['num_marked'] = num_marked
    
    return results

def plot_and_save_results(results, sheet_name, output_dir='results_random'):
    os.makedirs(output_dir, exist_ok=True)
    
    # Plot Pearson correlation
    plt.figure(figsize=(12, 6))
    for n_clusters in results.keys():
        max_len = max(len(run) for run in results[n_clusters]['pearson'])
        
        padded_pearson = []
        for run in results[n_clusters]['pearson']:
            padded = np.pad(run, (0, max_len - len(run)), 'edge')
            padded_pearson.append(padded)
            
        pearson_scores = np.array(padded_pearson)
        num_marked = results[n_clusters]['num_marked']
        
        if len(num_marked) < max_len:
            num_marked = np.pad(num_marked, (0, max_len - len(num_marked)), 'edge')
        
        mean_scores = np.mean(pearson_scores, axis=0)
        std_scores = np.std(pearson_scores, axis=0)
        
        plt.plot(num_marked, mean_scores, label=f'{n_clusters} clusters')
        plt.fill_between(num_marked, 
                        mean_scores - std_scores, 
                        mean_scores + std_scores, 
                        alpha=0.2)
    
    plt.xlabel('Number of Manually Marked Submissions')
    plt.ylabel("Pearson's r")
    plt.title(f'Pearson Correlation vs Marked Submissions\n{sheet_name}')
    plt.legend()
    plt.grid(True)
    plt.ylim(-0.1, 1.1)
    plt.savefig(os.path.join(output_dir, f'{sheet_name}_pearson.png'))
    plt.close()
    
    # Plot Quadratic Kappa
    plt.figure(figsize=(12, 6))
    for n_clusters in results.keys():
        max_len = max(len(run) for run in results[n_clusters]['kappa'])
        
        padded_kappa = []
        for run in results[n_clusters]['kappa']:
            padded = np.pad(run, (0, max_len - len(run)), 'edge')
            padded_kappa.append(padded)
            
        kappa_scores = np.array(padded_kappa)
        num_marked = results[n_clusters]['num_marked']
        
        if len(num_marked) < max_len:
            num_marked = np.pad(num_marked, (0, max_len - len(num_marked)), 'edge')
        
        mean_scores = np.mean(kappa_scores, axis=0)
        std_scores = np.std(kappa_scores, axis=0)
        
        plt.plot(num_marked, mean_scores, label=f'{n_clusters} clusters')
        plt.fill_between(num_marked, 
                        mean_scores - std_scores, 
                        mean_scores + std_scores, 
                        alpha=0.2)
    
    plt.xlabel('Number of Manually Marked Submissions')
    plt.ylabel('Quadratic Weighted Kappa')
    plt.title(f'Quadratic Weighted Kappa vs Marked Submissions\n{sheet_name}')
    plt.legend()
    plt.grid(True)
    plt.ylim(-0.1, 1.1)
    plt.savefig(os.path.join(output_dir, f'{sheet_name}_kappa.png'))
    plt.close()
    
    # Saving results to CSV
    for n_clusters in results.keys():
        max_len = max(len(run) for run in results[n_clusters]['pearson'])
        
        padded_pearson = []
        padded_kappa = []
        for run_p, run_k in zip(results[n_clusters]['pearson'], results[n_clusters]['kappa']):
            padded_p = np.pad(run_p, (0, max_len - len(run_p)), 'edge')
            padded_k = np.pad(run_k, (0, max_len - len(run_k)), 'edge')
            padded_pearson.append(padded_p)
            padded_kappa.append(padded_k)
            
        pearson_scores = np.array(padded_pearson)
        kappa_scores = np.array(padded_kappa)
        num_marked = results[n_clusters]['num_marked']
        
        if len(num_marked) < max_len:
            num_marked = np.pad(num_marked, (0, max_len - len(num_marked)), 'edge')
        
        df_results = pd.DataFrame({
            'num_marked': num_marked,
            'pearson_mean': np.mean(pearson_scores, axis=0),
            'pearson_std': np.std(pearson_scores, axis=0),
            'kappa_mean': np.mean(kappa_scores, axis=0),
            'kappa_std': np.std(kappa_scores, axis=0)
        })
        
        df_results.to_csv(os.path.join(output_dir, 
            f'{sheet_name}_clusters_{n_clusters}_results.csv'), index=False)

if __name__ == "__main__":
    file_path = './IDSA_data_final.xlsx'
    sheet_names = ['Exam2020-Q1', 'TheoryTest1-2022-Q1', 'Quiz5-2021-Q2']
    
    # model and embeddings retriever
    model_name = './bert_epoch_48'
    embeddings_retriever = EmbeddingsRetriever(model_name)
    
    # Process each sheet and run experiments
    for sheet_name in sheet_names:
        print(f"\nProcessing {sheet_name}")
        results = run_multiple_experiments(sheet_name, embeddings_retriever)
        plot_and_save_results(results, sheet_name)