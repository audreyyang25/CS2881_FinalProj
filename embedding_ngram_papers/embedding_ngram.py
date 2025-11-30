import json
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import glob

def load_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]

def project_scalar(P, A, B):
    """Return scalar projection t of P onto line A→B."""
    AB = B - A
    AP = P - A
    return np.dot(AP, AB) / np.dot(AB, AB)

def market_update(scores, alpha=0.15):
    """
    market smoothing:
    S_t = alpha * v_t + (1-alpha) * S_(t-1)
    """
    trend = []
    S_prev = scores[0]  # initialize with first vote
    trend.append(S_prev)

    for v in scores[1:]:
        S_new = alpha * v + (1 - alpha) * S_prev
        trend.append(S_new)
        S_prev = S_new

    return np.array(trend)

def process_embedding_file(input_file, model_name):
    """Process a single embedding file and generate ngram plot"""
    print(f"\nProcessing {model_name} embeddings from {input_file}...")

    data = load_jsonl(input_file)

    # Reference embeddings
    ref_harmful = np.array(data[0]["embedding"])
    ref_beneficial = np.array(data[1]["embedding"])

    years = []
    scores = []

    for entry in data[2:]:
        year = int(entry["Date"])

        # Ignore all -1 dates
        if year == -1:
            continue

        P = np.array(entry["embedding"])

        t = project_scalar(P, ref_harmful, ref_beneficial)
        score = (1 - t) * 100  # beneficial=0, harmful=100

        years.append(year)
        scores.append(score)

    # DataFrame
    df = pd.DataFrame({"year": years, "score": scores}).sort_values("year")

    # Market-style trend
    market_trend = market_update(df["score"].values, alpha=0.15)

    # -----------------------------
    # Plot
    # -----------------------------
    plt.figure(figsize=(12, 6))

    # Scatter points
    plt.scatter(df["year"], df["score"], color="blue", alpha=0.4, label="Claim votes")

    # Kalshi-style trendline
    plt.plot(df["year"], market_trend, color="black", linewidth=2.5,
             label="Market-style belief trend")

    plt.title(f"Trend of AI Safety Belief Over Time ({model_name})\n"
              "(0 = AI is dangerous, 100 = AI is safe)")
    plt.xlabel("Year")
    plt.ylabel("Belief Score")
    plt.grid(True, linestyle="--", alpha=0.3)
    plt.legend()

    # Save output
    os.makedirs("plots", exist_ok=True)
    output_file = f"plots/{model_name}_paper_claims_ngram_market.png"
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Saved: {output_file}")
    return output_file

def main():
    """Process all model-specific embedding files"""
    # Find all embedding files matching the pattern
    embedding_files = glob.glob("*_paper_claims_embedded.jsonl")

    if not embedding_files:
        print("No embedding files found!")
        return

    print("="*60)
    print("N-gram Market Trend Generator")
    print("="*60)
    print(f"Found {len(embedding_files)} embedding file(s):")
    for f in embedding_files:
        print(f"  - {f}")
    print("="*60)

    generated_plots = []

    for embedding_file in embedding_files:
        # Extract model name from filename
        # e.g., "openai_paper_claims_embedded.jsonl" -> "openai"
        if embedding_file == "paper_claims_embedded.jsonl":
            model_name = "default"
        else:
            model_name = embedding_file.replace("_paper_claims_embedded.jsonl", "")

        try:
            output_file = process_embedding_file(embedding_file, model_name)
            generated_plots.append(output_file)
        except Exception as e:
            print(f"Error processing {embedding_file}: {e}")

    print("\n" + "="*60)
    print(f"Generated {len(generated_plots)} plot(s):")
    for plot in generated_plots:
        print(f"  - {plot}")
    print("="*60)

if __name__ == "__main__":
    main()
