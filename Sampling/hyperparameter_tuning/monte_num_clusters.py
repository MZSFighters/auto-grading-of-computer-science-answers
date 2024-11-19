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

# initial marked subset sampling function
def initial_cluster_sampling(embeddings, n_clusters, samples_per_cluster):
    kmeans = KMeans(n_clusters=n_clusters, random_state=None)
    cluster_labels = kmeans.fit_predict(embeddings)
    
    # Sample points from each cluster, preferring points closer to centroids.
    selected_indices = []
    for cluster in range(n_clusters):
        cluster_mask = cluster_labels == cluster
        cluster_embeddings = embeddings[cluster_mask]
        cluster_indices = np.where(cluster_mask)[0]
        
        if len(cluster_indices) == 0:
            continue
            
        if len(cluster_indices) >= samples_per_cluster:
            random_indices = np.random.choice(cluster_indices, size=samples_per_cluster, replace=False)
        else:
            random_indices = cluster_indices
        
        selected_indices.extend(random_indices)
    return selected_indices, cluster_labels

def predict_marks(df, marked_indices, initial_indices, embeddings, knn, cluster_labels):
    #Predict marks using cluster means first, then KNN as fallback
    predictions = np.zeros(len(df))
    true_marks = df['marks'].values
        
    # assign true marks for submissions in initial set
    for idx in initial_indices:
        predictions[idx] = true_marks[idx]
    
    #For unmarked submissions, try cluster mean first, then KNN
    unmarked = set(range(len(df))) - set(initial_indices)
    for cluster in np.unique(cluster_labels):
        cluster_mask = cluster_labels == cluster
        cluster_marked = [i for i in marked_indices if cluster_labels[i] == cluster]
      
        cluster_unmarked = [i for i in unmarked if cluster_labels[i] == cluster]
        if cluster_marked:
            
            cluster_mean = df.iloc[cluster_marked]['marks'].mean()
            for idx in cluster_unmarked:
                predictions[idx] = cluster_mean
        else:
            for idx in cluster_unmarked:
                distances, indices = knn.kneighbors([embeddings[idx]])
                marked_neighbors = [i for i in indices[0] if i in marked_indices]
                if marked_neighbors:
                    weights = 1 / (distances[0][:len(marked_neighbors)] + 1e-6)
                    neighbor_marks = df.iloc[marked_neighbors]['marks'].values
                    predictions[idx] = np.average(neighbor_marks, weights=weights)
                else:
                    predictions[idx] = df.iloc[list(marked_indices)]['marks'].mean()
    return predictions

# function that runs single experiment
def run_experiment(df, embeddings, n_clusters, initial_samples, monte_carlo_ratio, n_iterations):
    
    # normalised embeddings
    normalizer = Normalizer(norm='l2')
    embeddings = normalizer.fit_transform(embeddings)
    
    # knn
    knn = NearestNeighbors(n_neighbors=3, metric='cosine')
    knn.fit(embeddings)
    
    samples_per_cluster = max(1, initial_samples // n_clusters)
    initial_indices, cluster_labels = initial_cluster_sampling(
        embeddings, n_clusters, samples_per_cluster)
    
    # Monte Carlo sampling
    n_monte_carlo_samples = int(len(initial_indices) * monte_carlo_ratio)
    all_predictions = np.zeros((len(df), n_iterations))
    
    marked_indices = set(initial_indices)
    pearson_scores = []
    quadratic_kappa_scores = []
    num_marked = []
    
    # Initial predictions
    for i in range(n_iterations):
        mc_indices = np.random.choice(
            initial_indices, n_monte_carlo_samples, replace=False)
        all_predictions[:, i] = predict_marks(
            df, mc_indices, initial_indices, embeddings, knn, cluster_labels)
    
    # Mean and Variance
    prediction_means = np.mean(all_predictions, axis=1)
    prediction_vars = np.std(all_predictions, axis=1)
    
    # Calculate initial metrics
    unmarked_mask = [idx for idx in range(len(df)) if idx not in marked_indices]
    true_marks_unmarked = df['marks'].iloc[unmarked_mask]
    predictions_unmarked = prediction_means[unmarked_mask]
    
    pearson_scores.append(pearsonr(true_marks_unmarked, predictions_unmarked)[0])
    quadratic_kappa_scores.append(cohen_kappa_score(
        true_marks_unmarked.round(),
        predictions_unmarked.round(),
        weights='quadratic'
    ))
    num_marked.append(len(marked_indices))
    
    # Greedy sampling process
    remaining_indices = set(range(len(df))) - marked_indices
    while remaining_indices and len(remaining_indices) > 2:
        variances = {idx: prediction_vars[idx] for idx in remaining_indices}
        next_idx = max(variances.items(), key=lambda x: x[1])[0]
        
        marked_indices.add(next_idx)
        remaining_indices.remove(next_idx)
        
        unmarked_mask = [idx for idx in range(len(df)) if idx not in marked_indices]
        true_marks_unmarked = df['marks'].iloc[unmarked_mask]
        predictions_unmarked = prediction_means[unmarked_mask]
        
        pearson_scores.append(pearsonr(true_marks_unmarked, predictions_unmarked)[0])
        quadratic_kappa_scores.append(cohen_kappa_score(
            true_marks_unmarked.round(),
            predictions_unmarked.round(),
            weights='quadratic'
        ))
        num_marked.append(len(marked_indices))
        
        # Update Monte Carlo predictions
        n_monte_carlo_samples = int(len(marked_indices) * monte_carlo_ratio)
        all_predictions = np.zeros((len(df), n_iterations))
        
        for i in range(n_iterations):
            mc_indices = np.random.choice(
                list(marked_indices), n_monte_carlo_samples, replace=False
            )
            all_predictions[:, i] = predict_marks(
                df, mc_indices, marked_indices, embeddings, knn, cluster_labels
            )
        
        prediction_means = np.mean(all_predictions, axis=1)
        prediction_vars = np.std(all_predictions, axis=1)
        
    return np.array(num_marked), np.array(pearson_scores), np.array(quadratic_kappa_scores)

def run_multiple_experiments(sheet_name, embeddings_retriever, n_runs=10):
    """Run multiple experiments for different cluster numbers and compute statistics."""
    cluster_numbers = [20, 30, 50, 80, 100]
    results = {n: {'pearson': [], 'kappa': [], 'num_marked': None} for n in cluster_numbers}
    
    # Load and preprocess data
    df = pd.read_excel(file_path, sheet_name=sheet_name)
    df['marks'] = df['marks'].round(0).astype(int)
    
    # embeddings processing
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
        initial_samples = 2 * n_clusters
        
        # for 10 runs
        for run in range(n_runs):
            num_marked, pearson_scores, kappa_scores = run_experiment(
                valid_df, embeddings, n_clusters, initial_samples, 
                monte_carlo_ratio=0.25, n_iterations=1000
            )
            
            results[n_clusters]['pearson'].append(pearson_scores)
            results[n_clusters]['kappa'].append(kappa_scores)
            if results[n_clusters]['num_marked'] is None:
                results[n_clusters]['num_marked'] = num_marked
    
    return results

def plot_and_save_results(results, sheet_name, output_dir='results_clustering'):
    os.makedirs(output_dir, exist_ok=True)
    cluster_numbers = list(results.keys())
    
    # Plot Pearson correlation
    plt.figure(figsize=(12, 6))
    for n_clusters in cluster_numbers:
        max_len = max(len(run) for run in results[n_clusters]['pearson'])
        padded_pearson = []
        for run in results[n_clusters]['pearson']:
            padded = np.pad(run, (0, max_len - len(run)), 'edge')  # Use edge padding
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
    for n_clusters in cluster_numbers:
        # Find the maximum length among all runs
        max_len = max(len(run) for run in results[n_clusters]['kappa'])
        
        # Pad each run to the maximum length
        padded_kappa = []
        for run in results[n_clusters]['kappa']:
            padded = np.pad(run, (0, max_len - len(run)), 'edge')  # Use edge padding
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
    
    # saving results to CSV
    for n_clusters in cluster_numbers:
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
    sheet_names = ['Exam2020-Q1', 'TheoryTest1-2022-Q1','Quiz5-2021-Q2']
    
    # model and embeddings
    model_name = './bert_epoch_48'
    embeddings_retriever = EmbeddingsRetriever(model_name)
    
    # Process each sheet
    for sheet_name in sheet_names:
        print(f"\nProcessing {sheet_name}")
        results = run_multiple_experiments(sheet_name, embeddings_retriever)
        plot_and_save_results(results, sheet_name)