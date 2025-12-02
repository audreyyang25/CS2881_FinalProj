# Requirements:
# pip install umap-learn matplotlib scikit-learn numpy pandas plotly sentence-transformers voyageai openai python-dotenv

import os
import json
import numpy as np
import umap
import matplotlib.pyplot as plt
import voyageai
from openai import OpenAI
from dotenv import load_dotenv
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sentence_transformers import SentenceTransformer
import torch

# Load dataset

def load_claims(path="semantically_similar_claims.jsonl"):
    claims = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            entry = json.loads(line)
            claims.append(entry)
    return claims

# Embedding functions for each model

def embed_openai(texts):
    load_dotenv()
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=texts
    )
    return np.array([item.embedding for item in response.data])


def embed_voyage(texts):
    load_dotenv()
    client = voyageai.Client(api_key=os.getenv("VOYAGE_API_KEY"))
    result = client.embed(
        texts=texts,
        model="voyage-lite-02-instruct",
        input_type="document"
    )
    return np.array(result.embeddings)


def embed_mxbai(texts):
    print("Loading mxbai-embed-large-v1...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SentenceTransformer("mixedbread-ai/mxbai-embed-large-v1", device=device)
    return model.encode(texts, convert_to_numpy=True)


# Map "model_name" to embedding function
MODEL_DISPATCH = {
    "openai": embed_openai,
    "voyage": embed_voyage,
    "mxbai": embed_mxbai
}


# Full pipeline for a single model

def run_pipeline(model_name, claims):
    print(f"\n==============================")
    print(f"Running model: {model_name}")
    print(f"==============================")

    output_dir = f"outputs/{model_name}"
    os.makedirs(output_dir, exist_ok=True)

    texts = [c["text"] for c in claims]
    labels = np.array([1 if c["label"] == "safe" else 0 for c in claims])  # 1=safe

    # ---- 3.1 Embeddings ----
    embed_fn = MODEL_DISPATCH[model_name]
    embeddings = embed_fn(texts)
    np.save(f"{output_dir}/similar_embeddings.npy", embeddings)

    # ---- 3.2 Reduce to 2D with UMAP ----
    scaler = StandardScaler()
    emb_scaled = scaler.fit_transform(embeddings)
    reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=42)
    emb2 = reducer.fit_transform(emb_scaled)

    # ---- 3.3 Train classifier on 2D ----
    X_train, X_test, y_train, y_test = train_test_split(
        emb2, labels, test_size=0.25, random_state=42, stratify=labels
    )

    clf = LogisticRegression().fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    y_proba = clf.predict_proba(X_test)[:, 1]

    # ---- 3.4 Metrics ----
    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba)
    print(f"Accuracy = {acc:.3f} | AUC = {auc:.3f}")

    # ---- 3.5 Centroids ----
    centroid_safe = emb2[labels == 1].mean(axis=0)
    centroid_unsafe = emb2[labels == 0].mean(axis=0)

    # ---- 3.6 Plot ----
    plt.figure(figsize=(9, 7))
    plt.scatter(emb2[:, 0], emb2[:, 1], c=labels, cmap='coolwarm', s=80, edgecolors='k')

    # Centroids
    plt.scatter(*centroid_safe, c="blue", marker="X", s=200, edgecolor="white", label="Safe centroid")
    plt.scatter(*centroid_unsafe, c="red", marker="X", s=200, edgecolor="white", label="Unsafe centroid")

    # Decision boundary
    xx, yy = np.meshgrid(
        np.linspace(emb2[:, 0].min() - 1, emb2[:, 0].max() + 1, 300),
        np.linspace(emb2[:, 1].min() - 1, emb2[:, 1].max() + 1, 300),
    )
    grid = np.c_[xx.ravel(), yy.ravel()]
    z = clf.predict_proba(grid)[:, 1].reshape(xx.shape)
    plt.contourf(xx, yy, z, levels=20, cmap="RdBu", alpha=0.15)

    plt.title(f"{model_name.upper()} — Ethical Claim Embedding Space")
    plt.xlabel("UMAP-1")
    plt.ylabel("UMAP-2")
    plt.legend()

    plot_path = f"{output_dir}/similar_claims_umap_plot.png"
    plt.savefig(plot_path, dpi=200)
    plt.close()

    print(f"Saved plot to {plot_path}")

    # ---- 3.7 Centroid-based evaluation ----
    d_safe = np.linalg.norm(emb2 - centroid_safe, axis=1)
    d_unsafe = np.linalg.norm(emb2 - centroid_unsafe, axis=1)
    centroid_pred = (d_safe < d_unsafe).astype(int)

    print("\nCentroid classifier report:")
    print(classification_report(labels, centroid_pred, target_names=["unsafe", "safe"]))

    return {
        "accuracy": acc,
        "auc": auc
    }

# Run all 3 models

if __name__ == "__main__":
    claims = load_claims()
    summary = {}

    for model in ["openai", "voyage", "mxbai"]:
        summary[model] = run_pipeline(model, claims)

    print("\n=========== SUMMARY ===========")
    for m, res in summary.items():
        print(f"{m.upper()}:  Acc={res['accuracy']:.3f},  AUC={res['auc']:.3f}")
    print("================================\n")
