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
    output_dir = f"outputs/{model_name}"
    os.makedirs(output_dir, exist_ok=True)

    texts = [c["text"] for c in claims]
    labels = np.array([1 if c["label"] == "safe" else 0 for c in claims])  # 1=safe

    # ---- Generate Embeddings ----
    print(f"\nGenerating {model_name} embeddings...")
    embed_fn = MODEL_DISPATCH[model_name]
    embeddings = embed_fn(texts)

    # ---- Save as JSONL ----
    output_file = f"{model_name}_ethical_claims_embedded.jsonl"
    print(f"Saving to {output_file}...")

    with open(output_file, "w", encoding="utf-8") as f:
        for claim, embedding in zip(claims, embeddings):
            # Create record with original claim data + embedding
            record = {
                "id": claim["id"],
                "text": claim["text"],
                "label": claim["label"],
                "embedding": embedding.tolist()
            }
            f.write(json.dumps(record) + "\n")

    # Also save embeddings as .npy for backwards compatibility
    npy_file = os.path.join(output_dir, "embeddings.npy")
    np.save(npy_file, embeddings)

    print(f"  Saved JSONL: {output_file}")
    print(f"  Saved NPY: {npy_file}")
    print(f"  Total records: {len(claims)}")

# Run all 3 models
if __name__ == "__main__":
    print("="*70)
    print("GENERATING EMBEDDINGS FOR ALL MODELS")
    print("="*70)

    claims = load_claims()
    print(f"\nLoaded {len(claims)} claims from semantically_similar_claims.jsonl")

    models = ["openai", "voyage", "mxbai"]

    for i, model in enumerate(models, 1):
        print(f"\n{'='*70}")
        print(f"MODEL {i}/{len(models)}: {model.upper()}")
        print(f"{'='*70}")
        run_pipeline(model, claims)

    print(f"\n{'='*70}")
    print("ALL EMBEDDINGS GENERATED SUCCESSFULLY!")
    print(f"{'='*70}")
