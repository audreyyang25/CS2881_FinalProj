import json
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os

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

def main():
    data = load_jsonl("smoking_claims_embedded.jsonl")

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

    plt.title("Trend of Smoking Belief Over Time\n"
              "(0 = Smoking is Good, 100 = Smoking is Harmful)")
    plt.xlabel("Year")
    plt.ylabel("Belief Score")
    plt.grid(True, linestyle="--", alpha=0.3)
    plt.legend()

    # Save output
    os.makedirs("plots", exist_ok=True)
    plt.savefig("plots/smoking_claims_ngram_market.png", dpi=300, bbox_inches="tight")
    plt.close()

    print("Saved: plots/smoking_claims_ngram_market.png")

if __name__ == "__main__":
    main()
