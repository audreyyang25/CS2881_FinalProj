from sklearn.cluster import MiniBatchKMeans
import numpy as np
import joblib
import json

# Perform clustering on the .npy embeddings
def cluster_embeddings(model):
    npy_file = f"{model}_paper_claims_embeddings.npy"
    embeddings = np.load(npy_file)

    n_clusters = 10  # Set desired number of clusters
    kmeans = MiniBatchKMeans(n_clusters=n_clusters, batch_size=32)
    kmeans.fit(embeddings)

    # Save the clustering model
    model_file = f"{model}_kmeans_model.joblib"
    joblib.dump(kmeans, model_file)
    print(f"Saved KMeans model to {model_file}")

    # Assign cluster labels to each embedding
    cluster_labels = kmeans.labels_

    # Save cluster labels to a .jsonl file
    output_file = f"{model}_paper_claims_clustered.jsonl"
    with open(output_file, "w", encoding="utf-8") as f:
        for i, label in enumerate(cluster_labels):
            obj = {
                "embedding_index": i,
                "cluster_label": int(label)
            }
            f.write(json.dumps(obj) + "\n")
    print(f"Saved clustered data to {output_file}")

# Use partial fit to update clustering model with new data
def insert_new_data(new_embeddings, model):
    # Load existing clustering model
    model_file = f"{model}_kmeans_model.joblib"
    kmeans = joblib.load(model_file)

    # Add new data
    new_data = new_embeddings

    # Assign to nearest cluster
    cluster_id = kmeans.predict(new_data)
    print(f"New data assigned to cluster ID: {cluster_id}")

    # Update the clustering model with new data
    kmeans.partial_fit(new_data)

    # Save the updated clustering model
    updated_model_file = f"{model}_kmeans_model_updated.joblib"
    joblib.dump(kmeans, updated_model_file)
    print(f"Saved updated KMeans model to {updated_model_file}")

def main():
    models = [
        "openai",
        "voyage",
        "mxbai",
    ]

    for model in models:
        cluster_embeddings(model)

if __name__ == "__main__":
    main()

