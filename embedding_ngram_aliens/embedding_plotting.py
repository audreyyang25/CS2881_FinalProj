import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px

def load_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]

def euclidean(a, b):
    return np.linalg.norm(a - b)

def main():
    data = load_jsonl("alien_claims_embedded.jsonl")

    # -----------------------------
    # Extract embeddings
    # -----------------------------
    ref1 = np.array(data[0]["embedding"])
    ref2 = np.array(data[1]["embedding"])
    d12 = euclidean(ref1, ref2)

    xs, ys, claims, dates, types = [], [], [], [], []

    # Reference point 1 (exists)
    xs.append(0.0)
    ys.append(0.0)
    claims.append(data[0]["Claim"])
    dates.append(0)
    types.append("Reference (exists)")

    # Reference point 2 (does not exist)
    xs.append(d12)
    ys.append(0.0)
    claims.append(data[1]["Claim"])
    dates.append(0)
    types.append("Reference (does not exist)")

    # Other claims
    for entry in data[2:]:
        emb = np.array(entry["embedding"])

        d1 = euclidean(emb, ref1)
        d2 = euclidean(emb, ref2)

        # Compute 2D projection
        x = (d1**2 - d2**2 + d12**2) / (2 * d12)
        y_sq = d1**2 - x**2
        y = np.sqrt(max(y_sq, 0.0))

        xs.append(x)
        ys.append(y)
        claims.append(entry["Claim"])
        dates.append(int(entry["Date"]))
        types.append("Claim")

    # Build DataFrame
    df = pd.DataFrame({
        "x": xs,
        "y": ys,
        "claim": claims,
        "date": dates,
        "type": types
    })

    # Exclude reference year 0 from color scaling
    real_years = df[df["type"] == "Claim"]["date"]
    min_year = real_years.min()
    max_year = real_years.max()

    # ============================================================
    # CREATE TWO SUBSETS
    # ============================================================

    df_claims = df[df["type"] == "Claim"]

    # A) Version WITHOUT -1 dates
    df_no_minus1 = df_claims[df_claims["date"] != -1].copy()
    min_year_A = df_no_minus1["date"].min()
    max_year_A = df_no_minus1["date"].max()

    # B) Version WITH -1 dates (colored red)
    df_with_minus1 = df_claims.copy()
    min_year_B = df_with_minus1[df_with_minus1["date"] != -1]["date"].min()
    max_year_B = df_with_minus1[df_with_minus1["date"] != -1]["date"].max()

    # ============================================================
    # 1) PLOTLY VERSION — NO -1 DATES
    # ============================================================

    fig_html_A = px.scatter(
        df_no_minus1,
        x="x",
        y="y",
        color="date",
        hover_data={"claim": True, "date": True, "x": False, "y": False},
        color_continuous_scale="Viridis",
        range_color=[min_year_A, max_year_A],
        title="Embedding (No -1 dates)"
    )
    fig_html_A.add_annotation(x=0, y=0.02, text="Ref: do exist", showarrow=False)
    fig_html_A.add_annotation(x=d12, y=0.02, text="Ref: does not exist", showarrow=False)

    # ============================================================
    # 1b) PLOTLY VERSION — INCLUDE -1 DATES IN RED (FIXED)
    # ============================================================

    # Split data
    df_norm = df_with_minus1[df_with_minus1["date"] != -1]
    df_red  = df_with_minus1[df_with_minus1["date"] == -1]

    fig_html_B = px.scatter(
        df_norm,
        x="x",
        y="y",
        color="date",
        hover_data={"claim": True, "date": True, "x": False, "y": False},
        color_continuous_scale="Viridis",
        range_color=[min_year_B, max_year_B],
        title="Embedding (Including -1 dates in red)"
    )

    # Add red points as a separate trace
    fig_html_B.add_scatter(
        x=df_red["x"],
        y=df_red["y"],
        mode="markers",
        marker=dict(color="red", size=8),
        name="Date = -1",
        hovertext=df_red["claim"],
        hoverinfo="text"
    )

    # Add reference annotations
    fig_html_A.add_annotation(x=0, y=0.02, text="Ref: do exist", showarrow=False)
    fig_html_A.add_annotation(x=d12, y=0.02, text="Ref: does not exist", showarrow=False)

    # Clean layout
    fig_html_B.update_layout(
        xaxis_title="X-axis (aligned to reference embeddings)",
        yaxis_title="Y-axis (perpendicular)",
        hoverlabel=dict(bgcolor="white", font_size=14),
        coloraxis_colorbar=dict(title="Year (non -1)")
    )

    # ============================================================
    # 2) MATPLOTLIB PNG — NO -1 DATES
    # ============================================================

    figA, axA = plt.subplots(figsize=(20, 6))

    norm_A = (df_no_minus1["date"] - min_year_A) / (max_year_A - min_year_A)
    colors_A = plt.cm.viridis(norm_A)

    axA.scatter(df_no_minus1["x"], df_no_minus1["y"], c=colors_A, s=40)
    axA.text(0, 0, "Ref: aliens do not exist", fontsize=10, va="bottom")
    axA.text(d12, 0, "Ref: aliens exist", fontsize=10, va="bottom")
    axA.set_title("Embedding (No -1 dates)")
    axA.set_xlabel("X-axis")
    axA.set_ylabel("Y-axis")
    axA.grid(True, linestyle="--", alpha=0.3)

    smA = plt.cm.ScalarMappable(
        cmap="viridis",
        norm=plt.Normalize(vmin=min_year_A, vmax=max_year_A)
    )
    smA._A = []
    figA.colorbar(smA, ax=axA).set_label("Year")

    # ============================================================
    # 2b) MATPLOTLIB PNG — INCLUDE -1 DATES IN RED
    # ============================================================

    figB, axB = plt.subplots(figsize=(20, 6))

    # Viridis for normal years
    normal_mask = df_with_minus1["date"] != -1
    norm_B = (df_with_minus1.loc[normal_mask, "date"] - min_year_B) / (max_year_B - min_year_B)
    normal_colors = plt.cm.viridis(norm_B)

    # Red for -1 points
    red_mask = df_with_minus1["date"] == -1
    red_points = df_with_minus1[red_mask]

    axB.scatter(
        df_with_minus1.loc[normal_mask, "x"],
        df_with_minus1.loc[normal_mask, "y"],
        c=normal_colors,
        s=40,
        label="Dated Claims"
    )
    axB.scatter(
        red_points["x"],
        red_points["y"],
        c="red",
        s=40,
        label="-1 Claims"
    )

    axB.text(0, 0, "Ref: aliens exist", fontsize=10, va="bottom")
    axB.text(d12, 0, "Ref: aliens do not exist", fontsize=10, va="bottom")
    axB.set_title("Embedding (Including -1 dates in red)")
    axB.set_xlabel("X-axis")
    axB.set_ylabel("Y-axis")
    axB.grid(True, linestyle="--", alpha=0.3)

    smB = plt.cm.ScalarMappable(
        cmap="viridis",
        norm=plt.Normalize(vmin=min_year_B, vmax=max_year_B)
    )
    smB._A = []
    figB.colorbar(smB, ax=axB).set_label("Year (non -1 only)")
    axB.legend()

    # ============================================================
    # SAVE
    # ============================================================

    output_dir = "plots"
    os.makedirs(output_dir, exist_ok=True)

    fig_html_A.write_html(os.path.join(output_dir, "plot_no_minus1.html"))
    fig_html_B.write_html(os.path.join(output_dir, "plot_with_minus1.html"))

    figA.savefig(os.path.join(output_dir, "plot_no_minus1.png"),
                 dpi=300, bbox_inches="tight")
    figB.savefig(os.path.join(output_dir, "plot_with_minus1.png"),
                 dpi=300, bbox_inches="tight")

    plt.close(figA)
    plt.close(figB)

    print("Saved:")
    print(" - plots/plot_no_minus1.html")
    print(" - plots/plot_with_minus1.html")
    print(" - plots/plot_no_minus1.png")
    print(" - plots/plot_with_minus1.png")

if __name__ == "__main__":
    main()
