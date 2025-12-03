import os
import json
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, accuracy_score, classification_report
from sklearn.model_selection import train_test_split


# ============================================================
# Load embedded JSONL file
# ============================================================

def load_embedded_jsonl(path):
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            data.append(json.loads(line))
    return data


# ============================================================
# 1D PROJECTION (ONLY X AXIS)
# The y-axis is discarded as requested
# ============================================================

def project_embeddings_1d(embeddings):
    """
    embeddings: (N, D)
    index 0: refsafe
    index 1: refunsafe

    Returns an array of shape (N,) containing ONLY the X projection.
    """

    ref_safe = embeddings[0]
    ref_unsafe = embeddings[1]

    d12 = np.linalg.norm(ref_safe - ref_unsafe)

    xs = []

    for emb in embeddings:
        d1 = np.linalg.norm(emb - ref_safe)
        d2 = np.linalg.norm(emb - ref_unsafe)

        # 1D projection onto the reference axis
        x = (d1**2 - d2**2 + d12**2) / (2 * d12)
        xs.append(x)

    return np.array(xs)


# ============================================================
# PIPELINE
# ============================================================

def run_pipeline(model_name, jsonl_path):
    print(f"\n======================================")
    print(f"Running model: {model_name}")
    print(f"Reading embeddings from: {jsonl_path}")
    print(f"======================================")

    # -------------------------
    # Load JSONL
    # -------------------------
    records = load_embedded_jsonl(jsonl_path)

    texts = [r["text"] for r in records]
    labels = np.array([1 if r["label"] == "safe" else 0 for r in records])
    embeddings = np.array([r["embedding"] for r in records])

    # -------------------------
    # Compute 1D projection
    # -------------------------
    xs = project_embeddings_1d(embeddings)

    # -------------------------
    # REMOVE references from evaluation
    # -------------------------
    xs_eval = xs[2:]       # drop refsafe (0) + refunsafe (1)
    labels_eval = labels[2:]

    # -------------------------
    # Train/test split
    # -------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        xs_eval.reshape(-1, 1),
        labels_eval,
        test_size=0.25,
        random_state=42,
        stratify=labels_eval
    )

    # -------------------------
    # Logistic Regression on 1D axis
    # -------------------------
    clf = LogisticRegression().fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    y_proba = clf.predict_proba(X_test)[:, 1]

    # -------------------------
    # Metrics
    # -------------------------
    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba)

    print(f"\n---- Logistic Regression (1D projection) ----")
    print(f"Accuracy: {acc:.3f}")
    print(f"AUC:      {auc:.3f}\n")
    print(classification_report(y_test, y_pred, target_names=["unsafe", "safe"]))

    # -------------------------
    # Centroid classifier (1D)
    # -------------------------
    safe_centroid = xs_eval[labels_eval == 1].mean()
    unsafe_centroid = xs_eval[labels_eval == 0].mean()

    d_safe = np.abs(xs_eval - safe_centroid)
    d_unsafe = np.abs(xs_eval - unsafe_centroid)
    centroid_pred = (d_safe < d_unsafe).astype(int)

    print("\n---- Centroid Classifier (1D) ----")
    print(classification_report(labels_eval, centroid_pred, target_names=["unsafe", "safe"]))

    # -------------------------
    # Visualization (1D axis plot)
    # -------------------------
    outdir = f"projection_eval/{model_name}"
    os.makedirs(outdir, exist_ok=True)

    plt.figure(figsize=(12, 3))
    plt.scatter(xs_eval, np.zeros_like(xs_eval), c=labels_eval, cmap="coolwarm",
                s=80, edgecolors="k")

    # Plot SAFE / UNSAFE centroids
    plt.scatter([safe_centroid], [0], c="blue", marker="X", s=200, label="Safe centroid")
    plt.scatter([unsafe_centroid], [0], c="red", marker="X", s=200, label="Unsafe centroid")

    # Plot references
    plt.scatter([xs[0]], [0], c="green", s=150, marker="D", label="REFSAFE")
    plt.scatter([xs[1]], [0], c="black", s=150, marker="D", label="REFUNSAFE")

    plt.title(f"{model_name.upper()} — 1D Projection Axis")
    plt.yticks([])
    plt.xlabel("Projection X (distance along the reference axis)")
    plt.legend()

    plot_path = f"{outdir}/projection_1d_plot.png"
    plt.savefig(plot_path, dpi=200, bbox_inches="tight")
    plt.close()

    print(f"Saved plot to {plot_path}")

    return {
        "accuracy": acc,
        "auc": auc
    }


# ============================================================
# RUN ALL MODELS
# ============================================================

if __name__ == "__main__":
    summary = {}

    model_files = {
        "openai": "openai_ethical_claims_embedded.jsonl",
        # "voyage": "voyage_ethical_claims_embedded.jsonl",
        # "mxbai":  "mxbai_ethical_claims_embedded.jsonl"
    }

    for model_name, path in model_files.items():
        summary[model_name] = run_pipeline(model_name, path)

    print("\n=========== SUMMARY ===========")
    for m, res in summary.items():
        print(f"{m.upper()}: Acc={res['accuracy']:.3f}, AUC={res['auc']:.3f}")
    print("================================\n")
