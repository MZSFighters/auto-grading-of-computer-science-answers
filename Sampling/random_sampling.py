
import numpy as np
import pandas as pd
from sklearn.preprocessing import Normalizer
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.neighbors import NearestNeighbors
from scipy.stats import pearsonr
from sklearn.metrics import cohen_kappa_score
import matplotlib.pyplot as plt
from embeddings_retriever import EmbeddingsRetriever

def incremental_random_sampling(df, model_name, n_clusters, k_neighbors):
    texts = list(df['response'])
    embeddings_retriever = EmbeddingsRetriever(model_name)
    embeddings_dict = embeddings_retriever.get_multiple_embeddings(texts)
    
    # process embeddings
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
    valid_df = df[df['response'].isin(valid_texts)].reset_index(drop=True)
    
    # normalising embeddings
    normalizer = Normalizer(norm='l2')
    embeddings = normalizer.fit_transform(embeddings)
    
    # Clustering
    kmeans = KMeans(n_clusters=n_clusters,random_state=None)
    predicted_labels = kmeans.fit_predict(embeddings)
    
    valid_df['cluster'] = predicted_labels
    
    # kNN model for backup after mean of cluster
    knn = NearestNeighbors(n_neighbors=k_neighbors, metric='cosine')
    knn.fit(embeddings)

    pearson_scores = []
    kappa_scores = []
    num_marked = []
    
    # marked submissions set
    marked_indices = set()
    num_submissions = len(valid_df)
    
    def predict_marks(df, marked_indices, embeddings):
        
        predictions = np.zeros(len(df))
        true_marks = df['marks'].values
        
        # For marked submissions, use their actual marks
        for idx in marked_indices:
            predictions[idx] = true_marks[idx]
        
        # For unmarked submissions, ...
        unmarked = set(range(len(df))) - marked_indices
        for cluster in df['cluster'].unique():
            cluster_mask = df['cluster'] == cluster
            cluster_marked = [i for i in marked_indices if df.iloc[i]['cluster'] == cluster]
            
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
                        # finding k nearest neighbors among marked submissions
                        distances, indices = knn.kneighbors([embeddings[idx]])
                        # filtering for only marked neighbors
                        marked_neighbors = [i for i in indices[0] if i in marked_indices]
                        if marked_neighbors:
                            #weighted average based on inverse distance
                            weights = 1 / (distances[0][:len(marked_neighbors)] + 1e-6)
                            neighbor_marks = df.iloc[marked_neighbors]['marks'].values
                            predictions[idx] = np.average(neighbor_marks, weights=weights)
                        else:
                            # if no marked neighbors found, use mean of all marked submissions
                            predictions[idx] = df.iloc[list(marked_indices)]['marks'].mean()
        
        return predictions
    
    # incrememntally marking submissions and calculating metrics
    for i in range(num_submissions):
        # Randomly select an unmarked submission
        unmarked = list(set(range(num_submissions)) - marked_indices)
        if not unmarked:
            break
            
        selected_idx = np.random.choice(unmarked)
        marked_indices.add(selected_idx)
        
        if len(marked_indices) >= 1:
            # Predict marks and calculating metrics for all submissions
            predicted_marks = predict_marks(valid_df, marked_indices, embeddings)
            pearson = pearsonr(valid_df['marks'], predicted_marks)[0]
            kappa = cohen_kappa_score(
                valid_df['marks'].round(),
                predicted_marks.round()
            )
            
            pearson_scores.append(pearson)
            kappa_scores.append(kappa)
            num_marked.append(len(marked_indices))
    
    # Plot and return results
    plt.figure(figsize=(12, 6))
    plt.plot(num_marked, pearson_scores, label="Pearson's r", marker='o', markersize=2)
    plt.plot(num_marked, kappa_scores, label="Cohen's Kappa", marker='s', markersize=2)
    plt.xlabel('Number of Manually Marked Submissions')
    plt.ylabel('Score')
    plt.title('Grading Metrics vs Number of Marked Submissions')
    plt.legend()
    plt.grid(True)
    plt.ylim(-0.1, 1.1)
    plt.tight_layout()
    plt.savefig('random_sampling_moc.png')
    plt.close()
    
    results_df = pd.DataFrame({
        'num_marked': num_marked,
        'pearson_r': pearson_scores,
        'cohen_kappa': kappa_scores
    })
    
    return results_df

def main():
    # Loading IDSA Dataset
    file_path = './Phase-2/IDSA_data_final.xlsx'
    df = pd.read_excel(file_path, sheet_name='TheoryTest1-2022-Q1')
    df['marks'] = df['marks'].round(0).astype(int)
    
    # Model configuration
    model_name = './Phase-2/bert_epoch_48'
    n_clusters = 20
    k_neighbors = 3
    
    # Run incremental sampling
    results = incremental_random_sampling(df, model_name, n_clusters, k_neighbors)
    print("\nFinal Results:")
    print(results)
    
    # Save results to CSV
    results.to_csv('random_sampling_moc_results.csv', index=False)
    print("\nResults saved to 'random_sampling_moc_results.csv' and plot saved to 'random_sampling_moc.png'")

if __name__ == "__main__":
    main()