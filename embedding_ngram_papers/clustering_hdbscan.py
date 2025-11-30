import numpy as np
import hdbscan
import umap
import json
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd

# HDBSCAN clustering of embeddings
def hdbscan_clustering(model):
    npy_file = f"{model}_paper_claims_embeddings.npy"
    embeddings = np.load(npy_file)

    # Perform HDBSCAN clustering
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=5, 
        metric='euclidean'
    ).fit(embeddings)
    cluster_labels = clusterer.labels_
    print("Cluster labels:", np.unique(cluster_labels))

    # Use UMAP for dimensionality reduction for visualization
    umap_model = umap.UMAP(n_neighbors=15, min_dist=0.1).fit(embeddings)
    embeddings_2d = umap_model.transform(embeddings)
    return embeddings_2d, cluster_labels

# Load original conclusion texts to analyze clusters
def load_conclusions(cluster_labels):
    input_file = "paper_claims.jsonl"
    with open(input_file, "r", encoding="utf-8") as f:
        data = [json.loads(line)["Claim"] for line in f]
    
    def get_top_terms_for_cluster(docs_in_cluster, n_terms=5):
        vectorizer = TfidfVectorizer(stop_words="english")
        X = vectorizer.fit_transform(docs_in_cluster)
        scores = zip(vectorizer.get_feature_names_out(), X.sum(axis=0).A1)
        sorted_terms = sorted(scores, key=lambda x: x[1], reverse=True)
        return [term for term, _ in sorted_terms[:n_terms]]
    
    cluster_topics = {}

    for cluster_id in np.unique(cluster_labels):
        if cluster_id == -1:
            continue
        indices = [i for i, lab in enumerate(cluster_labels) if lab == cluster_id]
        docs_in_cluster = [data[i] for i in indices]
        cluster_topics[cluster_id] = get_top_terms_for_cluster(docs_in_cluster)

    # Display topics
    for cid, topics in cluster_topics.items():
        print(f"Cluster {cid}: {topics}")
    return cluster_topics

# Plot clustering results
def plot_clusters(embeddings_2d, cluster_labels, cluster_topics, model):
    plt.figure(figsize=(10, 7))
    for cid in np.unique(cluster_labels):
        if cid == -1:
            continue
        indices = cluster_labels == cid
        plt.scatter(
            embeddings_2d[indices, 0], 
            embeddings_2d[indices, 1], 
            label=f"Cluster {cid}: {', '.join(cluster_topics[cid])}",
            s=30,
            alpha=0.7
        )
    plt.legend(bbox_to_anchor=(1.05, 1), loc='best')
    plt.title(f"HDBSCAN Clustering of {model} Embeddings")
    plt.savefig(f"plots/{model}_hdbscan_clusters.png")
    plt.show()

def main():
    models = [
        "openai",
        "voyage",
        "mxbai",
    ]


    for model in models:
        embeddings_2d, cluster_labels = hdbscan_clustering(model)
        cluster_topics = load_conclusions(cluster_labels)
        plot_clusters(embeddings_2d, cluster_labels, cluster_topics, model)

if __name__ == "__main__":
    main()