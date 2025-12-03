import os
import json
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier


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
    xs_eval = xs[2:]
    labels_eval = labels[2:]

    # ============================================================
    # SHARED TRAIN–TEST SPLIT FOR ALL MODELS
    # ============================================================

    indices = np.arange(len(xs_eval))

    train_idx, test_idx = train_test_split(
        indices,
        test_size=0.25,
        random_state=42,
        stratify=labels_eval
    )

    # 1D projection data
    X_train = xs_eval[train_idx].reshape(-1, 1)
    X_test = xs_eval[test_idx].reshape(-1, 1)
    y_train = labels_eval[train_idx]
    y_test = labels_eval[test_idx]

    # Full embedding data
    X_full = embeddings[2:]
    y_full = labels[2:]

    X_train_full = X_full[train_idx]
    X_test_full = X_full[test_idx]
    y_train_full = y_full[train_idx]
    y_test_full = y_full[test_idx]

    # ============================================================
    # LOGISTIC REGRESSION (1D PROJECTION)
    # ============================================================

    clf = LogisticRegression().fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    y_proba = clf.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba)

    print(f"\n---- Logistic Regression (1D projection) ----")
    print(f"Accuracy: {acc:.3f}")
    print(f"AUC:      {auc:.3f}\n")
    print(classification_report(y_test, y_pred, target_names=["unsafe", "safe"]))

    # ============================================================
    # GRADIENT BOOSTING (FULL EMBEDDING)
    # ============================================================

    gb = GradientBoostingClassifier()
    gb.fit(X_train_full, y_train_full)

    y_pred_full = gb.predict(X_test_full)
    y_proba_full = gb.predict_proba(X_test_full)[:, 1]

    acc_full = accuracy_score(y_test_full, y_pred_full)
    auc_full = roc_auc_score(y_test_full, y_proba_full)

    print(f"\n---- Gradient Boosting (FULL embedding) ----")
    print(f"Accuracy: {acc_full:.3f}")
    print(f"AUC:      {auc_full:.3f}\n")
    print(classification_report(y_test_full, y_pred_full,
                                target_names=["unsafe", "safe"]))

    # ============================================================
    # NEURAL NETWORK (MLP) — FULL EMBEDDING
    # ============================================================

    nn = MLPClassifier(
        hidden_layer_sizes=(256, 128),
        activation="relu",
        solver="adam",
        max_iter=100,
        random_state=42
    )

    nn.fit(X_train_full, y_train_full)

    y_pred_nn = nn.predict(X_test_full)
    y_proba_nn = nn.predict_proba(X_test_full)[:, 1]

    acc_nn = accuracy_score(y_test_full, y_pred_nn)
    auc_nn = roc_auc_score(y_test_full, y_proba_nn)

    print(f"\n---- Neural Net (MLP) — FULL embedding ----")
    print(f"Accuracy: {acc_nn:.3f}")
    print(f"AUC:      {auc_nn:.3f}\n")
    print(classification_report(y_test_full, y_pred_nn,
                                target_names=["unsafe", "safe"]))

    # ============================================================
    # CENTROID CLASSIFIER (1D)
    # ============================================================

    safe_centroid = xs_eval[labels_eval == 1].mean()
    unsafe_centroid = xs_eval[labels_eval == 0].mean()

    d_safe = np.abs(xs_eval - safe_centroid)
    d_unsafe = np.abs(xs_eval - unsafe_centroid)
    centroid_pred = (d_safe < d_unsafe).astype(int)

    print("\n---- Centroid Classifier (1D) ----")
    print(classification_report(labels_eval, centroid_pred,
                                target_names=["unsafe", "safe"]))

    # ============================================================
    # VISUALIZATION
    # ============================================================

    outdir = f"projection_eval/{model_name}"
    os.makedirs(outdir, exist_ok=True)

    plt.figure(figsize=(12, 3))
    plt.scatter(xs_eval, np.zeros_like(xs_eval), c=labels_eval,
                cmap="coolwarm", s=80, edgecolors="k")

    plt.scatter([safe_centroid], [0], c="blue", marker="X", s=200,
                label="Safe centroid")
    plt.scatter([unsafe_centroid], [0], c="red", marker="X", s=200,
                label="Unsafe centroid")

    plt.scatter([xs[0]], [0], c="green", s=150, marker="D", label="REFSAFE")
    plt.scatter([xs[1]], [0], c="black", s=150, marker="D", label="REFUNSAFE")

    plt.title(f"{model_name.upper()} — 1D Projection Axis")
    plt.yticks([])
    plt.xlabel("Projection X (distance along reference axis)")
    plt.legend()

    plot_path = f"{outdir}/projection_1d_plot.png"
    plt.savefig(plot_path, dpi=200, bbox_inches="tight")
    plt.close()

    print(f"Saved plot to {plot_path}")

    return {
        "accuracy": acc,
        "auc": auc,
        "gb_accuracy": acc_full,
        "gb_auc": auc_full,
        "nn_accuracy": acc_nn,
        "nn_auc": auc_nn
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
        print(
            f"{m.upper()}: "
            f"Acc={res['accuracy']:.3f}, AUC={res['auc']:.3f}, "
            f"GB_Acc={res['gb_accuracy']:.3f}, GB_AUC={res['gb_auc']:.3f}, "
            f"NN_Acc={res['nn_accuracy']:.3f}, NN_AUC={res['nn_auc']:.3f}"
        )
    print("================================\n")