import numpy as np
import json

def convert_jsonl_to_npy(model):
    input_file = f"{model}_paper_claims_embedded.jsonl"
    embeddings = []
    with open(input_file, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            embeddings.append(obj["embedding"])
    embeddings = np.array(embeddings)
    npy_file = f"{model}_paper_claims_embeddings.npy"
    np.save(npy_file, embeddings)
    print(f"Saved embeddings to {npy_file}")

def main():
    models = [
        "openai",
        "voyage",
        "mxbai",
    ]

    for model in models:
        convert_jsonl_to_npy(model)

if __name__ == "__main__":
    main()