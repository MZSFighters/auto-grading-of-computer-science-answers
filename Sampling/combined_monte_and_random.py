import numpy as np
import pandas as pd
from sklearn.preprocessing import Normalizer
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors
from scipy.stats import pearsonr
from sklearn.metrics import cohen_kappa_score, root_mean_squared_error
import matplotlib.pyplot as plt
from embeddings_retriever import EmbeddingsRetriever
import csv

# def plot_greedy_points(variances, errors):
    
#     plt.figure(figsize=(10, 6))
#     plt.scatter(variances, errors, alpha=0.6, color='blue')
    
#     plt.xlabel('Prediction Variance')
#     plt.ylabel('Absolute Error')
#     plt.title('Variance vs Error for Greedily Selected Points')
#     plt.grid(True, alpha=0.3)
    
#     correlation = np.corrcoef(variances, errors)[0,1]
#     plt.text(0.02, 0.98, f'Correlation: {correlation:.3f}', 
#              transform=plt.gca().transAxes, 
#              bbox=dict(facecolor='white', alpha=0.8))
    
#     plt.show()
    
# def plot_greedy_confidence(num_marked, confidence):
#     plt.figure(figsize=(10, 6))
#     sorted_indices = np.argsort(num_marked)
#     sorted_num_marked = np.array(num_marked)[sorted_indices]
#     sorted_confidence = np.array(confidence)[sorted_indices]

#     plt.plot(sorted_num_marked, sorted_confidence, color='green', linewidth=2)
#     plt.ylim(50, 100)
#     plt.xlabel('Number of Marked Submissions')
#     plt.ylabel('Confidence in the Auto-Marked Submissions')
#     plt.title('Confidence vs Number of Marked Submissions')
#     plt.grid(True, alpha=0.3)
#     plt.show()

def plot_greedy_points(all_variances, all_errors):
    plt.figure(figsize=(10, 6))
    # print(len(all_variances))
    # print(len(all_variances[0]))
    # print(len(all_variances[1]))
    # print(len(all_variances[2]))
    # print(len(all_variances[3]))
    # print(len(all_variances[4]))
    
    # Means and Standard Deviations
    mean_variances = np.mean(all_variances, axis=0)
    mean_errors = np.mean(all_errors, axis=0)
    stderr_variances = np.std(all_variances, axis=0) / np.sqrt(len(all_variances))
    stderr_errors = np.std(all_errors, axis=0) / np.sqrt(len(all_errors))
    
    plt.errorbar(mean_variances, mean_errors, 
                xerr=stderr_variances, 
                yerr=stderr_errors,
                fmt='o', 
                alpha=0.6, 
                color='blue',
                capsize=3,
                label='Mean with Std Error')
    
    # correlation on means
    correlation = np.corrcoef(mean_variances, mean_errors)[0,1]
    plt.text(0.02, 0.98, f'Correlation: {correlation:.3f}', 
             transform=plt.gca().transAxes, 
             bbox=dict(facecolor='white', alpha=0.8))
    
    plt.xlabel('Prediction Variance')
    plt.ylabel('Absolute Error')
    plt.title('Variance vs Error for Greedily Selected Points')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('greedy_points.png', dpi=300, bbox_inches='tight')
    plt.show()
    
# plotting confidence graph
def plot_greedy_confidence(all_num_marked, all_confidence, file_name):
    plt.figure(figsize=(10, 6))
    
    # calculating means and standard errors for each unique number of marked submissions
    unique_marked = np.unique(np.concatenate(all_num_marked))
    mean_confidence = []
    stderr_confidence = []
    
    for n in unique_marked:
        confidences_at_n = []
        for run_marked, run_conf in zip(all_num_marked, all_confidence):
            indices = np.where(np.array(run_marked) == n)[0]
            if len(indices) > 0:
                confidences_at_n.extend(np.array(run_conf)[indices])
        
        mean_confidence.append(np.mean(confidences_at_n))
        stderr_confidence.append(np.std(confidences_at_n) / np.sqrt(len(confidences_at_n)))
    
    mean_confidence = np.array(mean_confidence)
    stderr_confidence = np.array(stderr_confidence)
    
    plt.plot(unique_marked, mean_confidence, color='green', linewidth=2, label='Mean Confidence')
    plt.fill_between(unique_marked, 
                     mean_confidence - stderr_confidence,
                     mean_confidence + stderr_confidence,
                     color='green', alpha=0.2, label='Std Error')
    
    plt.ylim(50, 100)
    plt.xlabel('Number of Marked Submissions')
    plt.ylabel('Confidence in the Auto-Marked Submissions')
    plt.title('Confidence vs Number of Marked Submissions')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(f'{file_name}greedy_confidence.png', dpi=300, bbox_inches='tight')
    # plt.show()

def plot_comparison_metrics(all_mc_num_marked, all_mc_pearson, all_mc_kappa,
                          all_rand_num_marked, all_rand_pearson, all_rand_kappa, file_name):
    # Color scheme
    mc_color = '#2E86AB'  # Deep blue
    rand_color = '#F24236'  # Coral red
    
    # Common styling function
    def style_plot(ax, title, ylabel):
        ax.grid(True, alpha=0.2, linestyle='--', color='gray')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.set_xlabel('Number of Marked Submissions', fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.set_title(title, fontsize=14, pad=20)
        ax.legend(frameon=False, fontsize=10)
        
    # Process data for Monte Carlo
    unique_mc_marked = np.unique(np.concatenate(all_mc_num_marked))
    mc_pearson_means, mc_pearson_stderr = [], []
    mc_kappa_means, mc_kappa_stderr = [], []
    
    for n in unique_mc_marked:
        pearson_at_n, kappa_at_n = [], []
        for run_marked, run_pearson, run_kappa in zip(all_mc_num_marked, all_mc_pearson, all_mc_kappa):
            indices = np.where(np.array(run_marked) == n)[0]
            if len(indices) > 0:
                pearson_at_n.extend(np.array(run_pearson)[indices])
                kappa_at_n.extend(np.array(run_kappa)[indices])
        
        mc_pearson_means.append(np.mean(pearson_at_n))
        mc_pearson_stderr.append(np.std(pearson_at_n) / np.sqrt(len(pearson_at_n)))
        mc_kappa_means.append(np.mean(kappa_at_n))
        mc_kappa_stderr.append(np.std(kappa_at_n) / np.sqrt(len(kappa_at_n)))
    
    # Process data for Random
    unique_rand_marked = np.unique(np.concatenate(all_rand_num_marked))
    rand_pearson_means, rand_pearson_stderr = [], []
    rand_kappa_means, rand_kappa_stderr = [], []
    
    for n in unique_rand_marked:
        pearson_at_n, kappa_at_n = [], []
        for run_marked, run_pearson, run_kappa in zip(all_rand_num_marked, all_rand_pearson, all_rand_kappa):
            indices = np.where(np.array(run_marked) == n)[0]
            if len(indices) > 0:
                pearson_at_n.extend(np.array(run_pearson)[indices])
                kappa_at_n.extend(np.array(run_kappa)[indices])
        
        rand_pearson_means.append(np.mean(pearson_at_n))
        rand_pearson_stderr.append(np.std(pearson_at_n) / np.sqrt(len(pearson_at_n)))
        rand_kappa_means.append(np.mean(kappa_at_n))
        rand_kappa_stderr.append(np.std(kappa_at_n) / np.sqrt(len(kappa_at_n)))
    
    # Plot Pearson's r
    fig_pearson, ax_pearson = plt.subplots(figsize=(10, 6))
    
    ax_pearson.plot(unique_mc_marked, mc_pearson_means, 
                   label='Monte Carlo', color=mc_color, linewidth=2.5)
    ax_pearson.fill_between(unique_mc_marked,
                          np.array(mc_pearson_means) - np.array(mc_pearson_stderr),
                          np.array(mc_pearson_means) + np.array(mc_pearson_stderr),
                          color=mc_color, alpha=0.15)
    
    ax_pearson.plot(unique_rand_marked, rand_pearson_means, 
                   label='Random', color=rand_color, linewidth=2.5)
    ax_pearson.fill_between(unique_rand_marked,
                          np.array(rand_pearson_means) - np.array(rand_pearson_stderr),
                          np.array(rand_pearson_means) + np.array(rand_pearson_stderr),
                          color=rand_color, alpha=0.15)
    
    style_plot(ax_pearson, "Comparison of Pearson's r: Monte Carlo vs Random", "Pearson's r")
    plt.tight_layout()
    plt.savefig(f'{file_name}_pearson_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot Kappa scores
    fig_kappa, ax_kappa = plt.subplots(figsize=(10, 6))
    
    ax_kappa.plot(unique_mc_marked, mc_kappa_means, 
                 label='Monte Carlo', color=mc_color, linewidth=2.5)
    ax_kappa.fill_between(unique_mc_marked,
                        np.array(mc_kappa_means) - np.array(mc_kappa_stderr),
                        np.array(mc_kappa_means) + np.array(mc_kappa_stderr),
                        color=mc_color, alpha=0.15)
    
    ax_kappa.plot(unique_rand_marked, rand_kappa_means, 
                 label='Random', color=rand_color, linewidth=2.5)
    ax_kappa.fill_between(unique_rand_marked,
                        np.array(rand_kappa_means) - np.array(rand_kappa_stderr),
                        np.array(rand_kappa_means) + np.array(rand_kappa_stderr),
                        color=rand_color, alpha=0.15)
    
    style_plot(ax_kappa, 'Comparison of Quadratic Kappa Score: Monte Carlo vs Random', 'Quadratic Kappa Score')
    plt.tight_layout()
    plt.savefig(f'{file_name}_kappa_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()  
  
    
# initial marked subset sampling function
def initial_cluster_sampling(embeddings, n_clusters=10, samples_per_cluster=2):
    target_total = 2 * n_clusters  # Always want this many samples total
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
    
    if len(selected_indices) < target_total:
        remaining_needed = target_total - len(selected_indices)
        # Get indices of all points not yet selected
        unselected_indices = list(set(range(len(embeddings))) - set(selected_indices))
        additional_indices = np.random.choice(
            unselected_indices, 
            size=remaining_needed, 
            replace=False
        )
        selected_indices.extend(additional_indices)
    
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

def predict_random_marks(df, marked_indices, embeddings, knn):
        predictions = np.zeros(len(df))
        true_marks = df['marks'].values
        
        #print(marked_indices)
        
        # For marked submissions, use their actual marks
        for idx in marked_indices:
            predictions[idx] = true_marks[idx]
        
        # For unmarked submissions, predict based on cluster means or kNN
        unmarked = set(range(len(df))) - marked_indices
        for cluster in df['cluster'].unique():
            cluster_mask = df['cluster'] == cluster
            cluster_marked = [i for i in marked_indices if df.iloc[i]['cluster'] == cluster]
            
            if cluster_marked:
                # if cluster has marked submissions, use cluster mean
                cluster_mean = df.iloc[cluster_marked]['marks'].mean()
                for idx in unmarked:
                    if cluster_mask[idx]:
                        predictions[idx] = cluster_mean
            else:
                #print('hi')
                # If no marked submissions in cluster, then knn
                for idx in unmarked:
                    if cluster_mask[idx]:
                        distances, indices = knn.kneighbors([embeddings[idx]])
                        marked_neighbors = [i for i in indices[0] if i in marked_indices]
                        if marked_neighbors:
                            # weighted average based on inverse distance
                            weights = 1 / (distances[0][:len(marked_neighbors)] + 1e-6)
                            neighbor_marks = df.iloc[marked_neighbors]['marks'].values
                            predictions[idx] = np.average(neighbor_marks, weights=weights)
                        else:
                            predictions[idx] = df.iloc[list(marked_indices)]['marks'].mean()
        
        return predictions

# Greedy monte carlo process
def monte_simulation(valid_df, initial_samples, n_clusters, embeddings, knn, monte_carlo_ratio, n_iterations):
    # initial samples from clusters and cluster labels
    samples_per_cluster = max(1, initial_samples // n_clusters)
    initial_indices, cluster_labels = initial_cluster_sampling(
        embeddings, n_clusters, samples_per_cluster)
    
    print(len(initial_indices))
    
    # Monte Carlo sampling
    n_monte_carlo_samples = int(len(initial_indices) * monte_carlo_ratio)
    all_predictions = np.zeros((len(valid_df), n_iterations))

    for i in range(n_iterations):
        mc_indices = np.random.choice(
            initial_indices, n_monte_carlo_samples, replace=False)
        all_predictions[:, i] = predict_marks(
            valid_df, mc_indices, initial_indices, embeddings, knn, cluster_labels)

    # Mean and Variance for each submission
    prediction_means = np.mean(all_predictions, axis=1)
    prediction_vars = np.std(all_predictions, axis=1)

    pearson_scores = []
    quadratic_kappa_scores = []
    num_marked = []

    # initial marked set and its metrics
    marked_indices = set(initial_indices)

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

    # Greedy sampling process
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
        
        # Add to marked set
        marked_indices.add(next_idx)
        remaining_indices.remove(next_idx)

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

        # Recalculating mean and variance for each submission
        prediction_means = np.mean(all_predictions, axis=1)
        prediction_vars = np.std(all_predictions, axis=1)
    return num_marked, pearson_scores, quadratic_kappa_scores, greedy_variances, greedy_errors, greedy_confidence, conf_unmarked

# random sampling process
def random_simulation(valid_df, n_clusters, embeddings, knn):
    ahc = KMeans(n_clusters=n_clusters, random_state=None)
    predicted_labels = ahc.fit_predict(embeddings)

    valid_df['cluster'] = predicted_labels
    marked_indices = set()
    num_submissions = len(valid_df)

    run_pearson_scores = []
    run_quadratic_kappa_scores = []
    run_num_marked = []

    # incrementally marking submissions and calculating metrics
    for i in range(num_submissions):
        unmarked = list(set(range(num_submissions)) - marked_indices)
        if not unmarked:
            break
        
        selected_idx = np.random.choice(unmarked)
        marked_indices.add(selected_idx)

        if len(marked_indices) <= num_submissions - 2:
            # predicting the marks for unmarked submsisions
            predicted_marks = predict_random_marks(valid_df, marked_indices, embeddings, knn)

            # Calculating metrics
            unmarked_mask = [idx for idx in range(num_submissions) if idx not in marked_indices]
            true_marks_unmarked = valid_df['marks'].iloc[unmarked_mask]
            predicted_marks_unmarked = predicted_marks[unmarked_mask]

            pearson = pearsonr(true_marks_unmarked, predicted_marks_unmarked)[0]
            quadratic_kappa = cohen_kappa_score(true_marks_unmarked.round(), predicted_marks_unmarked.round(), weights='quadratic')

            run_pearson_scores.append(pearson)
            run_quadratic_kappa_scores.append(quadratic_kappa)
            run_num_marked.append(len(marked_indices))

    return run_num_marked, run_pearson_scores, run_quadratic_kappa_scores


n_runs = 10 # 10 runs for averaging
file_path = './IDSA_data_final.xlsx'
file_name = 'Exam2022-Q3'
# file_name = 'Exam2022-Q2'
# file_name = 'Exam2022-Q1'
# file_name = 'TheoryTest1-2022-Q2'
# file_name ='Exam2020-Q1'
# file_name = 'Quiz4-2020-Q2'
df = pd.read_excel(file_path, sheet_name=file_name)
df['marks'] = df['marks'].round(0).astype(int)

# Model configuration
model_name = './bert_epoch_48'
n_clusters = 25
initial_samples = 50
monte_carlo_ratio = 0.75
n_iterations = 300

texts = list(df['response'])
embeddings_retriever = EmbeddingsRetriever(model_name)
embeddings_dict = embeddings_retriever.get_multiple_embeddings(texts)

# Process embeddings
embeddings = []
valid_texts = []
for text in texts:
    emb = embeddings_dict.get(text, np.array([]))
    if emb.size > 0:
        embeddings.append(emb)
        valid_texts.append(text)

embeddings = np.array(embeddings)
valid_df = df[df['response'].isin(valid_texts)].reset_index(drop=True)

# Normalize embeddings
normalizer = Normalizer(norm='l2')
embeddings = normalizer.fit_transform(embeddings)

# kNN model
knn = NearestNeighbors(n_neighbors=3, metric='cosine')
knn.fit(embeddings)

all_mc_num_marked = []
all_mc_pearson_scores = [] 
all_mc_kappa_scores = []
all_greedy_variances = []
all_greedy_errors = []
all_greedy_confidence = []
all_conf_unmarked = []

all_rand_num_marked = []
all_rand_pearson_scores = []
all_rand_kappa_scores = []

# storing metrics over runs for both sampling techniques
for r in range(n_runs):
    print("Run: ", r)
    num_marked, pearson_scores, quadratic_kappa_scores, greedy_variances, greedy_errors, greedy_confidence, conf_unmarked = monte_simulation(valid_df, initial_samples,n_clusters,embeddings,knn, monte_carlo_ratio, n_iterations)

    random_num_marked, random_pearson_scores, random_quadratic_kappa_scores = random_simulation(valid_df, n_clusters, embeddings, knn)
    
    all_mc_num_marked.append(num_marked)
    all_mc_pearson_scores.append(pearson_scores)
    all_mc_kappa_scores.append(quadratic_kappa_scores)
    all_greedy_variances.append(greedy_variances)
    all_greedy_confidence.append(greedy_confidence)
    all_greedy_errors.append(greedy_errors)
    all_conf_unmarked.append(conf_unmarked)
    
    all_rand_num_marked.append(random_num_marked)
    all_rand_pearson_scores.append(random_pearson_scores)
    all_rand_kappa_scores.append(random_quadratic_kappa_scores)

file_path = f'{file_name}_greedy_confidence_data.csv'
with open(file_path, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["Conf Unmarked", "Greedy Variances"])
    for conf, var in zip(conf_unmarked, greedy_variances):
        writer.writerow([conf, var])

print(f'Confidence Data successfully saved to {file_path}')

plot_greedy_confidence(all_conf_unmarked, all_greedy_confidence, file_name)
plot_comparison_metrics(all_mc_num_marked, all_mc_pearson_scores, all_mc_kappa_scores,
                       all_rand_num_marked, all_rand_pearson_scores, all_rand_kappa_scores, file_name)
