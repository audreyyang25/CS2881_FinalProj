import os
import json
import numpy as np
import matplotlib.pyplot as plt
import umap
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report
from sklearn.model_selection import train_test_split


# ============================================================
# CATEGORY INFO
# ============================================================
CATEGORY_INFO = {
    'v': {'name': 'Violence & Conflict', 'color': '#e74c3c'},
    'p': {'name': 'Privacy & Data', 'color': '#3498db'},
    'm': {'name': 'Misinformation', 'color': '#9b59b6'},
    'h': {'name': 'Harassment & Civility', 'color': '#e67e22'},
    's': {'name': 'Safety & Wellbeing', 'color': '#1abc9c'},
    'w': {'name': 'Workplace Ethics', 'color': '#f39c12'},
    'e': {'name': 'Environment', 'color': '#27ae60'},
    'c': {'name': 'Healthcare', 'color': '#c0392b'},
    'f': {'name': 'Family & Parenting', 'color': '#8e44ad'},
    't': {'name': 'Technology & AI', 'color': '#16a085'},
    'd': {'name': 'Education', 'color': '#d35400'}
}


# ============================================================
# LOAD DATA
# ============================================================
def load_embedded_jsonl(path):
    """Load JSONL file with embeddings"""
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            data.append(json.loads(line))
    return data


def extract_data(records):
    """Extract texts, labels, embeddings, and categories from records"""
    texts = [r["text"] for r in records]
    labels = np.array([1 if r["label"] == "safe" else 0 for r in records])
    embeddings = np.array([r["embedding"] for r in records])
    categories = np.array([r["id"][0] for r in records])

    return texts, labels, embeddings, categories


# ============================================================
# 1D PROJECTION
# ============================================================
def project_embeddings_1d(embeddings):
    """
    Project embeddings to 1D using first two points as references.
    embeddings: (N, D)
    index 0: safe reference
    index 1: unsafe reference

    Returns array of shape (N,) containing X projection only.
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
# UMAP COMPUTATION
# ============================================================
def compute_umap_embeddings(embeddings):
    """Compute UMAP 2D embeddings and return them"""
    scaler = StandardScaler()
    emb_scaled = scaler.fit_transform(embeddings)

    reducer = umap.UMAP(
        n_neighbors=15,
        min_dist=0.1,
        metric="cosine",
        random_state=42
    )
    emb2 = reducer.fit_transform(emb_scaled)
    return emb2


# ============================================================
# UMAP VISUALIZATION
# ============================================================
def plot_umap_by_category(model_name, umap_embeddings, labels, categories, output_dir):
    """Create UMAP visualization colored by category"""
    print(f"  Creating UMAP by category...")

    emb2 = umap_embeddings

    # Create figure
    fig, ax = plt.subplots(figsize=(14, 8))

    # Plot each category separately
    unique_categories = sorted(set(categories))
    for cat in unique_categories:
        mask = categories == cat
        cat_info = CATEGORY_INFO[cat]
        ax.scatter(
            emb2[mask, 0],
            emb2[mask, 1],
            c=cat_info['color'],
            label=f"{cat.upper()}: {cat_info['name']}",
            s=50,
            alpha=0.7,
            edgecolor='black',
            linewidth=0.5
        )

    ax.set_title(f"{model_name} – Embedding Space by Category (UMAP 2D)",
                 fontsize=14, fontweight='bold')
    ax.set_xlabel("UMAP-1", fontsize=12)
    ax.set_ylabel("UMAP-2", fontsize=12)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)

    plt.tight_layout()
    output_path = os.path.join(output_dir, f"{model_name}_umap_by_category.png")
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"    Saved: {output_path}")


def plot_umap_with_safety_markers(model_name, umap_embeddings, labels, categories, output_dir):
    """Create UMAP with safe/unsafe markers"""
    print(f"  Creating UMAP with safety markers...")

    emb2 = umap_embeddings

    # Create figure
    fig, ax = plt.subplots(figsize=(14, 8))

    unique_categories = sorted(set(categories))
    for cat in unique_categories:
        cat_info = CATEGORY_INFO[cat]

        # Safe points (circles)
        mask_safe = (categories == cat) & (labels == 1)
        if mask_safe.any():
            ax.scatter(
                emb2[mask_safe, 0],
                emb2[mask_safe, 1],
                c=cat_info['color'],
                marker='o',
                s=60,
                alpha=0.7,
                edgecolor='black',
                linewidth=0.5,
                label=f"{cat.upper()}: {cat_info['name']} (safe)"
            )

        # Unsafe points (X markers)
        mask_unsafe = (categories == cat) & (labels == 0)
        if mask_unsafe.any():
            ax.scatter(
                emb2[mask_unsafe, 0],
                emb2[mask_unsafe, 1],
                c=cat_info['color'],
                marker='x',
                s=60,
                alpha=0.7,
                linewidth=2,
                label=f"{cat.upper()}: {cat_info['name']} (unsafe)"
            )

    ax.set_title(f"{model_name} – UMAP by Category\nCircles=Safe, X=Unsafe",
                 fontsize=14, fontweight='bold')
    ax.set_xlabel("UMAP-1", fontsize=12)
    ax.set_ylabel("UMAP-2", fontsize=12)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)

    plt.tight_layout()
    output_path = os.path.join(output_dir, f"{model_name}_umap_safety.png")
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"    Saved: {output_path}")


# ============================================================
# 1D PROJECTION VISUALIZATION
# ============================================================
def plot_1d_projection(model_name, xs, xs_eval, labels_eval, safe_centroid, unsafe_centroid, output_dir):
    """Visualize 1D projection with centroids"""
    print(f"  Creating 1D projection plot...")

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

    output_path = os.path.join(output_dir, f"{model_name}_1d_projection.png")
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()

    print(f"    Saved: {output_path}")


# ============================================================
# CLASSIFIERS
# ============================================================
def run_logistic_regression(X_train, X_test, y_train, y_test):
    """Train and evaluate logistic regression on 1D projection"""
    clf = LogisticRegression().fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    y_proba = clf.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba)

    return {
        'accuracy': acc,
        'auc': auc,
        'predictions': y_pred,
        'probabilities': y_proba,
        'model': clf
    }


def run_mlp(X_train, X_test, y_train, y_test):
    """Train and evaluate MLP on full embeddings"""
    nn = MLPClassifier(
        hidden_layer_sizes=(256, 128),
        activation="relu",
        solver="adam",
        max_iter=100,
        random_state=42
    )

    nn.fit(X_train, y_train)

    y_pred = nn.predict(X_test)
    y_proba = nn.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba)

    return {
        'accuracy': acc,
        'auc': auc,
        'predictions': y_pred,
        'probabilities': y_proba,
        'model': nn
    }


def run_centroid_classifier(xs_eval, labels_eval):
    """Simple centroid-based classifier on 1D projection"""
    safe_centroid = xs_eval[labels_eval == 1].mean()
    unsafe_centroid = xs_eval[labels_eval == 0].mean()

    d_safe = np.abs(xs_eval - safe_centroid)
    d_unsafe = np.abs(xs_eval - unsafe_centroid)
    centroid_pred = (d_safe < d_unsafe).astype(int)

    acc = accuracy_score(labels_eval, centroid_pred)

    return {
        'accuracy': acc,
        'predictions': centroid_pred,
        'safe_centroid': safe_centroid,
        'unsafe_centroid': unsafe_centroid
    }


# ============================================================
# MAIN PIPELINE
# ============================================================
def run_comprehensive_analysis(model_name, jsonl_path):
    """
    Run complete analysis pipeline:
    - Load data
    - Compute 1D projection
    - Train classifiers (Logistic Regression, MLP, Centroid)
    - Generate visualizations (UMAP, 1D projection)
    - Save all results
    """
    print(f"\n{'='*70}")
    print(f"COMPREHENSIVE ANALYSIS: {model_name.upper()}")
    print(f"{'='*70}")

    # Create output directory
    output_dir = os.path.join("outputs", model_name)
    os.makedirs(output_dir, exist_ok=True)

    # -------------------------
    # Load data
    # -------------------------
    print(f"\nLoading data from: {jsonl_path}")
    records = load_embedded_jsonl(jsonl_path)
    texts, labels, embeddings, categories = extract_data(records)

    print(f"Loaded {len(texts)} examples")
    print(f"Embedding shape: {embeddings.shape}")
    print(f"Safe: {(labels == 1).sum()}, Unsafe: {(labels == 0).sum()}")

    # -------------------------
    # Compute 1D projection
    # -------------------------
    print(f"\nComputing 1D projection...")
    xs = project_embeddings_1d(embeddings)

    # Remove reference points from evaluation
    xs_eval = xs[2:]
    labels_eval = labels[2:]
    categories_eval = categories[2:]
    embeddings_eval = embeddings[2:]

    # -------------------------
    # Compute UMAP embeddings
    # -------------------------
    print(f"\nComputing UMAP 2D embeddings...")
    umap_embeddings_eval = compute_umap_embeddings(embeddings_eval)

    # -------------------------
    # Train-test split
    # -------------------------
    print(f"\nSplitting data (75% train, 25% test)...")
    indices = np.arange(len(xs_eval))

    train_idx, test_idx = train_test_split(
        indices,
        test_size=0.25,
        random_state=42,
        stratify=labels_eval
    )

    # 1D projection data
    X_train_1d = xs_eval[train_idx].reshape(-1, 1)
    X_test_1d = xs_eval[test_idx].reshape(-1, 1)
    y_train = labels_eval[train_idx]
    y_test = labels_eval[test_idx]

    # UMAP 2D data
    X_train_umap = umap_embeddings_eval[train_idx]
    X_test_umap = umap_embeddings_eval[test_idx]

    # Full embedding data
    X_train_full = embeddings_eval[train_idx]
    X_test_full = embeddings_eval[test_idx]

    # -------------------------
    # Run classifiers
    # -------------------------
    print(f"\n{'='*70}")
    print(f"CLASSIFIER RESULTS")
    print(f"{'='*70}")

    # Logistic Regression (1D)
    print(f"\n--- Logistic Regression (1D Projection) ---")
    lr_results = run_logistic_regression(X_train_1d, X_test_1d, y_train, y_test)
    print(f"Accuracy: {lr_results['accuracy']:.3f}")
    print(f"AUC:      {lr_results['auc']:.3f}")
    print(classification_report(y_test, lr_results['predictions'],
                                target_names=["unsafe", "safe"]))

    # Logistic Regression (UMAP 2D)
    print(f"\n--- Logistic Regression (UMAP 2D) ---")
    lr_umap_results = run_logistic_regression(X_train_umap, X_test_umap, y_train, y_test)
    print(f"Accuracy: {lr_umap_results['accuracy']:.3f}")
    print(f"AUC:      {lr_umap_results['auc']:.3f}")
    print(classification_report(y_test, lr_umap_results['predictions'],
                                target_names=["unsafe", "safe"]))

    # MLP (Full Embeddings)
    print(f"\n--- Neural Network (MLP - Full Embeddings) ---")
    mlp_results = run_mlp(X_train_full, X_test_full, y_train, y_test)
    print(f"Accuracy: {mlp_results['accuracy']:.3f}")
    print(f"AUC:      {mlp_results['auc']:.3f}")
    print(classification_report(y_test, mlp_results['predictions'],
                                target_names=["unsafe", "safe"]))

    # Centroid Classifier (1D)
    print(f"\n--- Centroid Classifier (1D Projection) ---")
    centroid_results = run_centroid_classifier(xs_eval, labels_eval)
    print(f"Accuracy: {centroid_results['accuracy']:.3f}")
    print(classification_report(labels_eval, centroid_results['predictions'],
                                target_names=["unsafe", "safe"]))

    # -------------------------
    # Visualizations
    # -------------------------
    print(f"\n{'='*70}")
    print(f"GENERATING VISUALIZATIONS")
    print(f"{'='*70}")

    # UMAP by category
    plot_umap_by_category(model_name, umap_embeddings_eval, labels_eval,
                         categories_eval, output_dir)

    # UMAP with safety markers
    plot_umap_with_safety_markers(model_name, umap_embeddings_eval, labels_eval,
                                  categories_eval, output_dir)

    # 1D projection plot
    plot_1d_projection(model_name, xs, xs_eval, labels_eval,
                      centroid_results['safe_centroid'],
                      centroid_results['unsafe_centroid'],
                      output_dir)

    # -------------------------
    # Save results summary
    # -------------------------
    print(f"\n{'='*70}")
    print(f"SAVING RESULTS SUMMARY")
    print(f"{'='*70}")

    summary = {
        'model': model_name,
        'dataset_size': len(texts),
        'embedding_dim': embeddings.shape[1],
        'n_safe': int((labels == 1).sum()),
        'n_unsafe': int((labels == 0).sum()),
        'train_size': len(train_idx),
        'test_size': len(test_idx),
        'logistic_regression_1d': {
            'accuracy': float(lr_results['accuracy']),
            'auc': float(lr_results['auc'])
        },
        'logistic_regression_umap': {
            'accuracy': float(lr_umap_results['accuracy']),
            'auc': float(lr_umap_results['auc'])
        },
        'mlp_full_embedding': {
            'accuracy': float(mlp_results['accuracy']),
            'auc': float(mlp_results['auc'])
        },
        'centroid_1d': {
            'accuracy': float(centroid_results['accuracy']),
            'safe_centroid': float(centroid_results['safe_centroid']),
            'unsafe_centroid': float(centroid_results['unsafe_centroid'])
        }
    }

    summary_path = os.path.join(output_dir, f"{model_name}_results.json")
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"  Saved results to: {summary_path}")

    return summary


# ============================================================
# RUN ALL MODELS
# ============================================================
def main():
    """Run analysis for all models"""
    print("="*70)
    print("COMPREHENSIVE EMBEDDING ANALYSIS PIPELINE")
    print("="*70)

    model_files = {
        "openai": "openai_ethical_claims_embedded.jsonl",
        "voyage": "voyage_ethical_claims_embedded.jsonl",
        "mxbai": "mxbai_ethical_claims_embedded.jsonl"
    }

    all_results = {}

    for model_name, jsonl_file in model_files.items():
        try:
            summary = run_comprehensive_analysis(model_name, jsonl_file)
            all_results[model_name] = summary
        except Exception as e:
            print(f"\nError processing {model_name}: {e}")
            continue

    # Print final summary
    print("\n" + "="*80)
    print("FINAL SUMMARY - ALL MODELS")
    print("="*80)
    print(f"\n{'Model':<10} {'LR-1D':<11} {'LR-1D':<11} {'LR-UMAP':<11} {'LR-UMAP':<11} {'MLP':<11} {'MLP':<11} {'Centroid':<11}")
    print(f"{'':10} {'Acc':<11} {'AUC':<11} {'Acc':<11} {'AUC':<11} {'Acc':<11} {'AUC':<11} {'Acc':<11}")
    print("-"*80)

    for model_name, results in all_results.items():
        lr_acc = results['logistic_regression_1d']['accuracy']
        lr_auc = results['logistic_regression_1d']['auc']
        lr_umap_acc = results['logistic_regression_umap']['accuracy']
        lr_umap_auc = results['logistic_regression_umap']['auc']
        mlp_acc = results['mlp_full_embedding']['accuracy']
        mlp_auc = results['mlp_full_embedding']['auc']
        cent_acc = results['centroid_1d']['accuracy']

        print(f"{model_name.upper():<10} {lr_acc:<11.3f} {lr_auc:<11.3f} "
              f"{lr_umap_acc:<11.3f} {lr_umap_auc:<11.3f} "
              f"{mlp_acc:<11.3f} {mlp_auc:<11.3f} {cent_acc:<11.3f}")

    print("\n" + "="*70)
    print("ANALYSIS COMPLETE!")
    print(f"Results saved to: analysis_results/")
    print("="*70)


if __name__ == "__main__":
    main()
