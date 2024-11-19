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
def initial_cluster_sampling(embeddings, n_clusters=10, samples_per_cluster=5):
    kmeans = KMeans(n_clusters=n_clusters, random_state=None)
    # ahc = AgglomerativeClustering(n_clusters=n_clusters, metric='cosine', linkage='complete')
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


# This function nalyze how the maximum variance changes with increasing monte carlo iterations and returns the variance trajectories for multiple runs.
def analyze_monte_carlo_variance(df, embeddings, n_runs=10, max_iterations=1000):
    n_clusters = 25
    initial_samples = 2 * n_clusters
    monte_carlo_ratio = 0.25
    
    variance_trajectories = np.zeros((n_runs, max_iterations))
    
    # Process embeddings
    normalizer = Normalizer(norm='l2')
    embeddings = normalizer.fit_transform(embeddings)
    knn = NearestNeighbors(n_neighbors=3, metric='cosine')
    knn.fit(embeddings)
    
    for run in tqdm(range(n_runs), desc="Running Monte Carlo analysis"):
        # Initial clustering
        samples_per_cluster = max(1, initial_samples // n_clusters)
        initial_indices, cluster_labels = initial_cluster_sampling(
            embeddings, n_clusters, samples_per_cluster)
        
        # Monte Carlo sampling
        n_monte_carlo_samples = int(len(initial_indices) * monte_carlo_ratio)
        all_predictions = np.zeros((len(df), max_iterations))
        
        # Generate predictions for each iteration
        for i in range(max_iterations):
            mc_indices = np.random.choice(
                initial_indices, n_monte_carlo_samples, replace=False)
            all_predictions[:, i] = predict_marks(
                df, mc_indices, initial_indices, embeddings, knn, cluster_labels)
            
            # Calculate maximum variance up to current iteration
            current_predictions = all_predictions[:, :i+1]
            current_vars = np.std(current_predictions, axis=1)
            variance_trajectories[run, i] = np.max(current_vars)
    
    os.makedirs('results', exist_ok=True)
    variance_df = pd.DataFrame(variance_trajectories, columns=[f'Iteration_{i+1}' for i in range(max_iterations)])
    variance_df.to_csv(os.path.join('results', f'{sheet_name}_variance_trajectories.csv'), index=False)
    
    
    return variance_trajectories

#Plotting the convergence of maximum variance over iterations
def plot_variance_convergence(variance_trajectories, sheet_name, output_dir='results'):
    
    os.makedirs(output_dir, exist_ok=True)
    plt.figure(figsize=(12, 6))
    
    #individual runs with low opacity - shading
    for run in range(variance_trajectories.shape[0]):
        plt.plot(range(1, variance_trajectories.shape[1] + 1), 
                variance_trajectories[run], 
                color='blue', alpha=0.2)
    
    # mean and standard deviation
    mean_trajectory = np.mean(variance_trajectories, axis=0)
    std_trajectory = np.std(variance_trajectories, axis=0)
    
    plt.plot(range(1, len(mean_trajectory) + 1), mean_trajectory, 
             color='blue', linewidth=2, label='Mean')
    plt.fill_between(range(1, len(mean_trajectory) + 1),
                    mean_trajectory - std_trajectory,
                    mean_trajectory + std_trajectory,
                    color='blue', alpha=0.2)
    
    iterations_to_mark = [10, 50, 100, 200, 500, 1000]
    for iter_point in iterations_to_mark:
        if iter_point <= len(mean_trajectory):
            idx = iter_point - 1
            plt.annotate(f'{mean_trajectory[idx]:.3f} ± {std_trajectory[idx]:.3f}',
                        xy=(iter_point, mean_trajectory[idx]),
                        xytext=(10, 10), textcoords='offset points',
                        bbox=dict(facecolor='white', edgecolor='none', alpha=0.7))
    
    plt.xlabel('Number of Monte Carlo Iterations')
    plt.ylabel('Maximum Prediction Variance')
    plt.title(f'Monte Carlo Iteration Convergence\n{sheet_name}')
    plt.grid(True)
    plt.legend()
    
    # log scale subplot
    ax2 = plt.axes([0.65, 0.45, 0.2, 0.2])
    ax2.plot(range(1, len(mean_trajectory) + 1), mean_trajectory, color='blue')
    ax2.set_xscale('log')
    ax2.set_yscale('log')
    ax2.set_title('Log Scale View')
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{sheet_name}_variance_convergence.png'))
    plt.close()
    
    # saving to CSV
    df_results = pd.DataFrame({
        'iteration': range(1, len(mean_trajectory) + 1),
        'mean_variance': mean_trajectory,
        'std_variance': std_trajectory,
        'relative_std': (std_trajectory / mean_trajectory) * 100
    })
    
    df_results.to_csv(os.path.join(output_dir, 
        f'{sheet_name}_variance_convergence.csv'), index=False)
    
    print(f"\nConvergence Analysis for {sheet_name}:")
    
    # calculating relative change in variance
    relative_changes = np.diff(mean_trajectory) / mean_trajectory[:-1] * 100
    convergence_threshold = -0.1
    converged_iteration = np.where(relative_changes > convergence_threshold)[0][-1] + 2
    
    print(f"Variance appears to stabilize around iteration {converged_iteration}")
    print(f"Final maximum variance: {mean_trajectory[-1]:.3f} ± {std_trajectory[-1]:.3f}")
    
    return df_results

if __name__ == "__main__":
    file_path = './Phase-2/IDSA_data_final.xlsx'
    sheet_names = ['Exam2020-Q1', 'TheoryTest1-2022-Q1', 'Quiz3-2021-Q1']
    
    # model and embeddings retriever
    model_name = './Phase-2/bert_epoch_48'
    embeddings_retriever = EmbeddingsRetriever(model_name)
    
    # For each sheet, ...
    for sheet_name in sheet_names:
        print(f"\nAnalyzing variance convergence for {sheet_name}")
        
        # Load and preprocess data
        df = pd.read_excel(file_path, sheet_name=sheet_name)
        df['marks'] = df['marks'].round(0).astype(int)
        
        # Get embeddings
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
        
        variance_trajectories = analyze_monte_carlo_variance(valid_df, embeddings)
        plot_variance_convergence(variance_trajectories, sheet_name)