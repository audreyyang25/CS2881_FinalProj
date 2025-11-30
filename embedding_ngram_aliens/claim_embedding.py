import json
from openai import OpenAI
import voyageai
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
import os
import torch

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
    Embed all claims in input_file using the specified model.

    Args:
        model_name: One of 'openai', 'voyage', 'mxbai'
        input_file: Path to input JSONL file
        output_file: Path to output JSONL file
    """
    # Load environment variables
    load_dotenv()

    # Initialize the appropriate model/client
    if model_name == "openai":
        openai_api_key = os.getenv("OPENAI_API_KEY")
        if not openai_api_key:
            raise ValueError("OPENAI_API_KEY not found in .env file.")
        client = OpenAI(api_key=openai_api_key)
        embed_fn = lambda text: get_openai_embedding(client, text)

    elif model_name == "voyage":
        voyage_api_key = os.getenv("VOYAGE_API_KEY")
        if not voyage_api_key:
            raise ValueError("VOYAGE_API_KEY not found in .env file.")
        client = voyageai.Client(api_key=voyage_api_key)
        embed_fn = lambda text: get_voyage_embedding(client, text)

    elif model_name == "mxbai":
        print(f"Loading mxbai-embed-large-v1 model (~1.3GB)...")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = SentenceTransformer("mixedbread-ai/mxbai-embed-large-v1", device=device)
        embed_fn = lambda text: get_huggingface_embedding(model, text)

    else:
        raise ValueError(f"Unknown model: {model_name}")

    # Process the file
    print(f"Processing {input_file} with {model_name}...")
    count = 0

    with open(input_file, "r", encoding="utf-8") as infile, \
         open(output_file, "w", encoding="utf-8") as outfile:

        for line in infile:
            line = line.strip()
            if not line:
                continue

            obj = json.loads(line)
            claim_text = obj["Claim"]

            # Create embedding
            embedding = embed_fn(claim_text)
            obj["embedding"] = embedding

            # Write updated JSONL entry
            outfile.write(json.dumps(obj) + "\n")
            count += 1

            if count % 10 == 0:
                print(f"  Processed {count} claims...")

    print(f"Finished writing {count} claims to: {output_file}")

def main():
    """Generate embeddings for all specified models"""
    input_file = "alien_claims.jsonl"

    # Define which models to use
    models_to_run = [
        # "openai",      # OpenAI text-embedding-3-small (API)
        "voyage",      # Voyage lite 02 instruct (API)
        "mxbai",       # mxbai-embed-large-v1 (~1.3GB - safe)
    ]

    print("="*60)
    print("Multi-Model Embedding Generator")
    print("="*60)
    print(f"Input file: {input_file}")
    print(f"Models to run: {', '.join(models_to_run)}")
    print("="*60)

    for model_name in models_to_run:
        output_file = f"{model_name}_alien_claims_embedded.jsonl"
        try:
            embed_with_model(model_name, input_file, output_file)
            print()
        except Exception as e:
            print(f"Error with {model_name}: {e}")
            print()

    print("="*60)
    print("All models completed!")
    print("="*60)

if __name__ == "__main__":
    main()
