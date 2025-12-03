import json
import numpy as np
import umap
import matplotlib.pyplot as plt
import torch
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from sentence_transformers import SentenceTransformer
import voyageai
from openai import OpenAI
import os
from dotenv import load_dotenv

# Load data

def load_dataset(path):
    texts = []
    labels = []

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            texts.append(item["text"])
            labels.append(1 if item["label"] == "safe" else 0)

    return texts, np.array(labels)

# Embedding Functions

def embed_mxbai(texts):
    print("Loading mxbai-embed-large-v1...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SentenceTransformer("mixedbread-ai/mxbai-embed-large-v1", device=device)
    return model.encode(texts, convert_to_numpy=True)

def embed_voyage(texts):
    load_dotenv()
    client = voyageai.Client(api_key=os.getenv("VOYAGE_API_KEY"))
    result = client.embed(texts=texts, model="voyage-lite-02-instruct")
    return np.array(result.embeddings)

def embed_openai(texts):
    load_dotenv()
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    emb = client.embeddings.create(model="text-embedding-3-small", input=texts)
    return np.array([e.embedding for e in emb.data])


MODELS = {
    "mxbai": embed_mxbai,
    "Voyage Lite": embed_voyage,
    "OpenAI Small": embed_openai
}

# Centroids
def compute_centroids(embeddings, labels):
    safe_centroid = embeddings[labels == 1].mean(axis=0)
    unsafe_centroid = embeddings[labels == 0].mean(axis=0)
    return safe_centroid, unsafe_centroid


def safety_score(embedding, safe_c, unsafe_c):
    d_safe = np.linalg.norm(embedding - safe_c)
    d_unsafe = np.linalg.norm(embedding - unsafe_c)
    return d_unsafe - d_safe

# UMAP + Logistic Boundary Background
def plot_umap_for_model(model_name, embeddings, labels, test_embedding, score):
    scaler = StandardScaler()
    emb_scaled = scaler.fit_transform(embeddings)

    reducer = umap.UMAP(
        n_neighbors=15,
        min_dist=0.1,
        metric="cosine",
        random_state=42
    )
    emb2 = reducer.fit_transform(emb_scaled)

    # transform test embedding to same space
    test_scaled = scaler.transform(test_embedding.reshape(1, -1))
    test2 = reducer.transform(test_scaled)[0]

    # centroids in 2D
    safe_centroid = emb2[labels == 1].mean(axis=0)
    unsafe_centroid = emb2[labels == 0].mean(axis=0)

    # train boundary classifier
    clf = LogisticRegression().fit(emb2, labels)

    # grid for decision boundary
    x_min, x_max = emb2[:,0].min()-1, emb2[:,0].max()+1
    y_min, y_max = emb2[:,1].min()-1, emb2[:,1].max()+1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 300),
                         np.linspace(y_min, y_max, 300))
    grid = np.c_[xx.ravel(), yy.ravel()]
    probs = clf.predict_proba(grid)[:,1].reshape(xx.shape)

    # plotting
    plt.figure(figsize=(7,6))
    plt.contourf(xx, yy, probs, levels=20, cmap="RdBu", alpha=0.15)

    plt.scatter(emb2[:,0], emb2[:,1], c=labels, cmap="coolwarm", s=40, edgecolor="k")

    # centroids
    plt.scatter(*safe_centroid, c="blue", s=240, marker="X", edgecolor="white", label="Safe Centroid")
    plt.scatter(*unsafe_centroid, c="red", s=240, marker="X", edgecolor="white", label="Unsafe Centroid")

    # test point
    plt.scatter(test2[0], test2[1], color="yellow", s=200, edgecolors="black", label="Test Claim")
    plt.text(test2[0]+0.1, test2[1]+0.1, f"score={score:.2f}", fontsize=10)

    plt.title(f"{model_name} – Safe vs Unsafe Space (UMAP 2D)")
    plt.xlabel("UMAP-1")
    plt.ylabel("UMAP-2")
    plt.legend()
    plt.tight_layout()
    plt.show()

def evaluate_and_plot(claim_text, dataset_path="semantically_similar_claims.jsonl"):
    texts, labels = load_dataset(dataset_path)

    print(f"\nEvaluating claim:\n  \"{claim_text}\"\n")

    for model_name, embed_fn in MODELS.items():
        print(f"\n=== MODEL: {model_name} ===")

        # embed full dataset
        embeddings = embed_fn(texts)

        # centroids
        safe_c, unsafe_c = compute_centroids(embeddings, labels)

        # embed claim
        test_emb = embed_fn([claim_text])[0]

        # calculate safety score
        score = safety_score(test_emb, safe_c, unsafe_c)
        print(f"Safety Score: {score:.4f} (positive=safe)")

        # plot
        plot_umap_for_model(
            model_name=model_name,
            embeddings=embeddings,
            labels=labels,
            test_embedding=test_emb,
            score=score
        )

if __name__ == "__main__":
    claim = input("Please enter a claim to be evaluated: ")
    evaluate_and_plot(claim)
