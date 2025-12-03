import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import glob


def load_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def euclidean(a, b):
    return np.linalg.norm(a - b)


def process_embedding_file(input_file, model_name):
    """Process a single embedding file and generate safe/unsafe visualizations"""
    print(f"\nProcessing {model_name} embeddings from {input_file}...")

    data = load_jsonl(input_file)

    # -----------------------------
    # Extract reference embeddings
    # -----------------------------
    ref_safe = np.array(data[0]["embedding"])
    ref_unsafe = np.array(data[1]["embedding"])
    d12 = euclidean(ref_safe, ref_unsafe)

    xs, ys, texts, labels, types = [], [], [], [], []

    # Reference SAFE
    xs.append(0.0)
    ys.append(0.0)
    texts.append(data[0]["text"])
    labels.append("refsafe")
    types.append("Reference")

    # Reference UNSAFE
    xs.append(d12)
    ys.append(0.0)
    texts.append(data[1]["text"])
    labels.append("refunsafe")
    types.append("Reference")

    # -----------------------------
    # Project all other claims
    # -----------------------------
    for entry in data[2:]:
        emb = np.array(entry["embedding"])

        d1 = euclidean(emb, ref_safe)
        d2 = euclidean(emb, ref_unsafe)

        # Linear projection formulas
        x = (d1**2 - d2**2 + d12**2) / (2 * d12)
        y_sq = d1**2 - x**2
        y = np.sqrt(max(y_sq, 0.0))

        xs.append(x)
        ys.append(y)
        texts.append(entry["text"])
        labels.append(entry["label"])
        types.append("Claim")

    # -----------------------------
    # Create DataFrame
    # -----------------------------
    df = pd.DataFrame({
        "x": xs,
        "y": ys,
        "text": texts,
        "label": labels,
        "type": types
    })

    df_claims = df[df["type"] == "Claim"].copy()

    # -----------------------------
    # Compute centroids
    # -----------------------------
    safe_df = df_claims[df_claims["label"] == "safe"]
    unsafe_df = df_claims[df_claims["label"] == "unsafe"]

    safe_centroid_x = safe_df["x"].mean()
    safe_centroid_y = safe_df["y"].mean()

    unsafe_centroid_x = unsafe_df["x"].mean()
    unsafe_centroid_y = unsafe_df["y"].mean()

    # -----------------------------
    # PLOTLY (color = safe / unsafe)
    # -----------------------------
    fig_html = px.scatter(
        df_claims,
        x="x",
        y="y",
        color="label",
        hover_data={"text": True, "label": True, "x": False, "y": False},
        title=f"Safe vs Unsafe Projection — {model_name}",
        color_discrete_map={"safe": "green", "unsafe": "red"},
    )

    # Reference annotations
    fig_html.add_annotation(x=0, y=0.05, text="Reference SAFE", showarrow=False)
    fig_html.add_annotation(x=d12, y=0.05, text="Reference UNSAFE", showarrow=False)

    # SAFE centroid
    fig_html.add_scatter(
        x=[safe_centroid_x],
        y=[safe_centroid_y],
        mode="markers",
        marker=dict(color="green", size=16, symbol="diamond"),
        name="Safe Centroid",
        hovertext="SAFE centroid",
    )

    # UNSAFE centroid
    fig_html.add_scatter(
        x=[unsafe_centroid_x],
        y=[unsafe_centroid_y],
        mode="markers",
        marker=dict(color="red", size=16, symbol="diamond"),
        name="Unsafe Centroid",
        hovertext="UNSAFE centroid",
    )

    # -----------------------------
    # MATPLOTLIB PNG
    # -----------------------------
    fig, ax = plt.subplots(figsize=(20, 6))

    color_map = df_claims["label"].map({"safe": "green", "unsafe": "red"})
    ax.scatter(df_claims["x"], df_claims["y"], c=color_map, s=40)

    # Centroid markers
    ax.scatter(
        safe_centroid_x, safe_centroid_y,
        c="green", s=160, marker="D", edgecolor="black", label="Safe Centroid"
    )
    ax.scatter(
        unsafe_centroid_x, unsafe_centroid_y,
        c="red", s=160, marker="D", edgecolor="black", label="Unsafe Centroid"
    )

    # Reference points labels
    ax.text(0, 0, "Reference SAFE", fontsize=10, va="bottom")
    ax.text(d12, 0, "Reference UNSAFE", fontsize=10, va="bottom")

    ax.set_title(f"Safe vs Unsafe — {model_name}")
    ax.set_xlabel("X-axis (aligned to reference embeddings)")
    ax.set_ylabel("Y-axis (perpendicular)")
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.legend()

    # -----------------------------
    # SAVE OUTPUT
    # -----------------------------
    output_dir = "plots"
    os.makedirs(output_dir, exist_ok=True)

    html_path = os.path.join(output_dir, f"{model_name}_ethical_plot.html")
    png_path = os.path.join(output_dir, f"{model_name}_ethical_plot.png")

    fig_html.write_html(html_path)
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"✓ Saved {model_name} plots:")
    print(f"   - {html_path}")
    print(f"   - {png_path}")

    return [html_path, png_path]


def main():
    embedding_files = glob.glob("*_ethical_claims_embedded.jsonl")

    if not embedding_files:
        print("No ethical embedding files found!")
        return

    print("="*60)
    print("Safe/Unsafe Projection Plot Generator (With Centroids)")
    print("="*60)

    all_outputs = []

    for file in embedding_files:
        model_name = file.replace("_ethical_claims_embedded.jsonl", "")
        try:
            outputs = process_embedding_file(file, model_name)
            all_outputs.extend(outputs)
        except Exception as e:
            print(f"Error processing {file}: {e}")

    print("\nDone.")
    print("="*60)
    print(f"Generated {len(all_outputs)} files.")
    print("="*60)


if __name__ == "__main__":
    main()
