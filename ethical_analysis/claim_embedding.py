import json
from openai import OpenAI
# import voyageai
# from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
import os
# import torch

def get_openai_embedding(client, text):
    """Get embedding from OpenAI API"""
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=text
    )
    return response.data[0].embedding

def get_voyage_embedding(client, text):
    """Get embedding from Voyage AI API"""
    result = client.embed(
        texts=[text],
        model="voyage-lite-02-instruct",
        input_type="document"
    )
    return result.embeddings[0]

def get_huggingface_embedding(model, text):
    """Get embedding from HuggingFace sentence-transformers model"""
    embedding = model.encode(text, convert_to_numpy=True)
    return embedding.tolist()

def embed_with_model(model_name, input_file, output_file):
    """
    Embed all entries in input_file using the specified model.

    Args:
        model_name: One of 'openai', 'voyage', 'mxbai'
        input_file: Path to input JSONL file
        output_file: Path to output JSONL file
    """
    load_dotenv()

    # Initialize the appropriate model/client
    if model_name == "openai":
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY not found.")
        client = OpenAI(api_key=api_key)
        embed_fn = lambda text: get_openai_embedding(client, text)

    elif model_name == "voyage":
        api_key = os.getenv("VOYAGE_API_KEY")
        if not api_key:
            raise ValueError("VOYAGE_API_KEY not found.")
        client = voyageai.Client(api_key=api_key)
        embed_fn = lambda text: get_voyage_embedding(client, text)

    elif model_name == "mxbai":
        print("Loading mxbai-embed-large-v1 model...")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = SentenceTransformer("mixedbread-ai/mxbai-embed-large-v1", device=device)
        embed_fn = lambda text: get_huggingface_embedding(model, text)

    else:
        raise ValueError(f"Unknown model: {model_name}")

    print(f"Processing {input_file} with model: {model_name}")
    count = 0

    with open(input_file, "r", encoding="utf-8") as infile, \
         open(output_file, "w", encoding="utf-8") as outfile:

        for line in infile:
            line = line.strip()
            if not line:
                continue

            obj = json.loads(line)
            claim_text = obj["text"]  # UPDATED KEY

            # Create embedding
            embedding = embed_fn(claim_text)
            obj["embedding"] = embedding  # Add new field

            outfile.write(json.dumps(obj) + "\n")
            count += 1

            if count % 10 == 0:
                print(f"  Processed {count} entries...")

    print(f"Finished writing {count} entries to: {output_file}")

def main():
    input_file = "semantically_similar_claims.jsonl"

    models_to_run = [
        "openai",
        # "voyage",
        # "mxbai",
    ]

    print("="*60)
    print("Ethical Claims Embedding Generator")
    print("="*60)

    for model_name in models_to_run:
        output_file = f"{model_name}_ethical_claims_embedded.jsonl"
        try:
            embed_with_model(model_name, input_file, output_file)
            print()
        except Exception as e:
            print(f"Error with model {model_name}: {e}")
            print()

    print("="*60)
    print("All embeddings generated!")
    print("="*60)

if __name__ == "__main__":
    main()
