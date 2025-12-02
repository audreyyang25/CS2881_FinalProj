import os
import json
import numpy as np
import pandas as pd
import glob
import umap
import matplotlib.pyplot as plt
import plotly.express as px

def load_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]

# CLASSIFY SMOKING CLAIMS

def classify_smoking_stance(text):
    t = text.lower()

    bad_keywords = [
        "harm", "harmful", "detrimental", "dangerous", "bad",
        "cancer", "disease", "risk", "unsafe", "cause", "kills",
        "secondhand", "major cause", "preventable", "issues"
    ]
    good_keywords = [
        "beneficial", "healthy", "cleanse", "remedy", "soothing",
        "harmless", "calming", "divine gift", "fashionable",
        "benign", "relaxing"
    ]

    if any(k in t for k in bad_keywords):
        return "BAD (smoking harmful)"

    if any(k in t for k in good_keywords):
        return "GOOD (smoking beneficial)"

    return "Neutral"

# PROCESS EMBEDDING FILE

def process_embedding_file(input_file, model_name):
    print(f"\nProcessing {model_name} embeddings with UMAP...")

    data = load_jsonl(input_file)

    embeddings = np.array([np.array(d["embedding"]) for d in data])
    claims = [d["Claim"] for d in data]
    labels = [classify_smoking_stance(c) for c in claims]

    # UMAP REDUCTION
    reducer = umap.UMAP(
        n_neighbors=15,
        min_dist=0.1,
        metric="cosine",
        random_state=42,
    )
    emb2 = reducer.fit_transform(embeddings)

    df = pd.DataFrame({
        "x": emb2[:, 0],
        "y": emb2[:, 1],
        "claim": claims,
        "label": labels
    })

    # PLOTLY INTERACTIVE
    fig_html = px.scatter(
        df,
        x="x",
        y="y",
        color="label",
        title=f"{model_name} UMAP — Smoking Good vs Bad",
        hover_data={"claim": True, "label": True}
    )

    # MATPLOTLIB PNG
    fig, ax = plt.subplots(figsize=(10, 7))

    color_map = {
        "BAD (smoking harmful)": "#1f77b4",   # Blue
        "GOOD (smoking beneficial)": "#d62728",  # Red
        "Neutral": "#7f7f7f"                 # Grey
    }

    ax.scatter(df["x"], df["y"], c=[color_map[l] for l in df["label"]], s=40)
    ax.set_title(f"{model_name} — UMAP (Smoking Good vs Bad)")
    ax.set_xlabel("UMAP-1")
    ax.set_ylabel("UMAP-2")
    ax.grid(True, linestyle="--", alpha=0.3)

    # Legend
    handles = [
        plt.Line2D([0], [0], marker='o', color='w', label=lab,
                   markerfacecolor=color_map[lab], markersize=10)
        for lab in color_map
    ]
    ax.legend(handles=handles)

    # SAVE OUTPUT
    outdir = "umap_smoking"
    os.makedirs(outdir, exist_ok=True)

    html_f = os.path.join(outdir, f"{model_name}_umap_goodbad.html")
    png_f = os.path.join(outdir, f"{model_name}_umap_goodbad.png")

    fig_html.write_html(html_f)
    fig.savefig(png_f, dpi=300)
    plt.close(fig)

    print(f"✓ Saved UMAP for {model_name}")
    print(f"   {html_f}")
    print(f"   {png_f}")

def main():
    files = glob.glob("*_smoking_claims_embedded.jsonl")
    if not files:
        print("No embedding files found.")
        return

    for f in files:
        model_name = f.replace("_smoking_claims_embedded.jsonl", "")
        process_embedding_file(f, model_name)

if __name__ == "__main__":
    main()
