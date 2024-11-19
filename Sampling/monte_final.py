import numpy as np
import pandas as pd
from sklearn.preprocessing import Normalizer
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors
from scipy.stats import pearsonr
from sklearn.metrics import cohen_kappa_score, root_mean_squared_error
import matplotlib.pyplot as plt
from embeddings_retriever import EmbeddingsRetriever

# plots the variance vs error graph
def plot_greedy_points(variances, errors):
    
    plt.figure(figsize=(10, 6))
    plt.scatter(variances, errors, alpha=0.6, color='blue')
    
    plt.xlabel('Prediction Variance')
    plt.ylabel('Absolute Error')
    plt.title('Variance vs Error for Greedily Selected Points')
    plt.grid(True, alpha=0.3)
    
    correlation = np.corrcoef(variances, errors)[0,1]
    plt.text(0.02, 0.98, f'Correlation: {correlation:.3f}', 
             transform=plt.gca().transAxes, 
             bbox=dict(facecolor='white', alpha=0.8))
    
    plt.show()

# plots the confidence vs manually marked submissions graph
def plot_greedy_confidence(num_marked, confidence):
    plt.figure(figsize=(10, 6))
    sorted_indices = np.argsort(num_marked)
    sorted_num_marked = np.array(num_marked)[sorted_indices]
    sorted_confidence = np.array(confidence)[sorted_indices]

    plt.plot(sorted_num_marked, sorted_confidence, color='green', linewidth=2)
    plt.ylim(50, 100)
    plt.xlabel('Number of Marked Submissions')
    plt.ylabel('Confidence in the Auto-Marked Submissions')
    plt.title('Confidence vs Number of Marked Submissions')
    plt.grid(True, alpha=0.3)
    plt.show()
    
# initial marked subset sampling function
def initial_cluster_sampling(embeddings, n_clusters, samples_per_cluster):
    
    # we run k-means to get the clusters
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
    
    # assign true marks for submissions in intitial set
    for idx in initial_indices:
        predictions[idx] = true_marks[idx]
    
    unmarked = set(range(len(df))) - set(initial_indices)
    
    # for each cluster, ...
    for cluster in np.unique(cluster_labels):
        cluster_marked = [i for i in marked_indices if cluster_labels[i] == cluster]
        
        cluster_unmarked = [i for i in unmarked if cluster_labels[i] == cluster]
        
        # if a submission is marked in the cluster then use Mean of Cluster
        if cluster_marked:
            cluster_mean = df.iloc[cluster_marked]['marks'].mean()
            for idx in cluster_unmarked:
                predictions[idx] = cluster_mean
        # otherwise knn
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


# loading IDSA dataset
file_path = './Phase-2/IDSA_data_final.xlsx'
df = pd.read_excel(file_path, sheet_name='Quiz3-2020-Q2')
df['marks'] = df['marks'].round(0).astype(int)

# Model configuration
model_name = './Phase-2/bert_epoch_48'
n_clusters = 25
initial_samples = 50
monte_carlo_ratio = 0.75
n_iterations = 300

texts = list(df['response'])
embeddings_retriever = EmbeddingsRetriever(model_name)
embeddings_dict = embeddings_retriever.get_multiple_embeddings(texts)

# processing embeddings
embeddings = []
valid_texts = []
for text in texts:
    emb = embeddings_dict.get(text, np.array([]))
    if emb.size > 0:
        embeddings.append(emb)
        valid_texts.append(text)

embeddings = np.array(embeddings)
valid_df = df[df['response'].isin(valid_texts)].reset_index(drop=True)

# Normalizng embeddings
normalizer = Normalizer(norm='l2')
embeddings = normalizer.fit_transform(embeddings)

# kNN model for backup after mean of cluster
knn = NearestNeighbors(n_neighbors=3, metric='cosine')
knn.fit(embeddings)

# Get initial samples from clusters and cluster labels
samples_per_cluster = max(1, initial_samples // n_clusters)
initial_indices, cluster_labels = initial_cluster_sampling(embeddings, n_clusters, samples_per_cluster)

unmarked = set(range(len(df))) - set(initial_indices)
for cluster in np.unique(cluster_labels):
    cluster_marked = [i for i in initial_indices if cluster_labels[i] == cluster]
        
    # Get unmarked submissions in this cluster
    cluster_unmarked = [i for i in unmarked if cluster_labels[i] == cluster]
    
# Monte Carlo sampling
n_monte_carlo_samples = int(len(initial_indices) * monte_carlo_ratio)
all_predictions = np.zeros((len(valid_df), n_iterations))
for i in range(n_iterations):
    # Randomly sample from  the initial set
    mc_indices = np.random.choice(
        initial_indices, n_monte_carlo_samples, replace=False)
    all_predictions[:, i] = predict_marks(
        valid_df, mc_indices, initial_indices, embeddings, knn, cluster_labels)

# Mean and Variance for each submission
prediction_means = np.mean(all_predictions, axis=1)
prediction_vars = np.std(all_predictions, axis=1)

pearson_scores = []
kappa_scores = []
weighted_kappa_scores = []
quadratic_kappa_scores = []
num_marked = []

# starting with initial marked set
marked_indices = set(initial_indices)

# evaluating correlation metrics for all unmarked submissions
unmarked_mask = [idx for idx in range(len(valid_df)) if idx not in marked_indices]
true_marks_unmarked = valid_df['marks'].iloc[unmarked_mask]
initial_predictions_unmarked = prediction_means[unmarked_mask]

pearson_scores.append(pearsonr(true_marks_unmarked, initial_predictions_unmarked)[0])
quadratic_kappa_scores.append(cohen_kappa_score(
    true_marks_unmarked.round(),
    initial_predictions_unmarked.round(), weights='quadratic'
))
num_marked.append(len(marked_indices))

greedy_variances = []
greedy_confidence = []
conf_unmarked = []
truth_marks = []
predicted_marks = []
greedy_errors = []

# Greedy sampling based on variance
remaining_indices = set(range(len(valid_df))) - marked_indices
while remaining_indices and len(remaining_indices) > 2:
    
    print("Submissions remaining: ", len(remaining_indices))
    
    # select submission with highest variance
    variances = {idx: prediction_vars[idx] for idx in remaining_indices}
    next_idx = max(variances.items(), key=lambda x: x[1])[0]
    
    true_mark = valid_df['marks'].iloc[next_idx]
    predicted_mark = prediction_means[next_idx]
    truth_marks.append(true_mark)
    predicted_marks.append(predicted_mark)
    
    greedy_variances.append(prediction_vars[next_idx])
    greedy_confidence.append(100-prediction_vars[next_idx])
    
    # Add to the marked set
    marked_indices.add(next_idx)
    remaining_indices.remove(next_idx)

    # add metrics values for evaluation
    unmarked_mask = [idx for idx in range(len(valid_df)) if idx not in marked_indices]
    true_marks_unmarked = valid_df['marks'].iloc[unmarked_mask]
    predictions_unmarked = prediction_means[unmarked_mask]

    greedy_errors.append(root_mean_squared_error(true_marks_unmarked, predictions_unmarked))
    
    pearson_scores.append(pearsonr(true_marks_unmarked, predictions_unmarked)[0])
    quadratic_kappa_scores.append(cohen_kappa_score(
        true_marks_unmarked.round(),
        predictions_unmarked.round()
    , weights='quadratic'))
    num_marked.append(len(marked_indices))
    conf_unmarked.append(len(marked_indices))

    # Run Monte Carlo simulation again after marking
    n_monte_carlo_samples = int(len(marked_indices) * monte_carlo_ratio)
    all_predictions = np.zeros((len(valid_df), n_iterations))

    for i in range(n_iterations):
        mc_indices = np.random.choice(
            list(marked_indices), n_monte_carlo_samples, replace=False
        )
        all_predictions[:, i] = predict_marks(
            valid_df, mc_indices, marked_indices, embeddings, knn, cluster_labels
        )

    # Recalculate mean and variance for each submission
    prediction_means = np.mean(all_predictions, axis=1)
    prediction_vars = np.std(all_predictions, axis=1)
    
# final plotting and results
plot_greedy_confidence(conf_unmarked, greedy_confidence)
plot_greedy_points(greedy_variances, greedy_errors)

plt.figure(figsize=(12, 6))
plt.plot(num_marked, pearson_scores, label="Pearson's r")
plt.plot(num_marked, quadratic_kappa_scores, label="Quadratic Kappa")
plt.xlabel('Number of Manually Marked Submissions')
plt.ylabel('Score')
plt.title('Grading Metrics vs Number of Marked Submissions')
plt.legend()
plt.grid(True)
plt.ylim(-0.1, 1.1)
plt.tight_layout()

results_df = pd.DataFrame({
    'num_marked': num_marked,
    'pearson_r': pearson_scores,
    'quadratic_kappa': quadratic_kappa_scores
})