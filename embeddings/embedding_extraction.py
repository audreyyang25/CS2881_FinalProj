# Extract conclusions from pdfs and embed using LLMs
# Models: SFR-Embedding-Mistral, voyage-lite-02-instruct, will use Google Gecko once I get the key working
# Stores embeddings in separate ChromaDB directories

import os
import sys

# Add parent directory to path to import config
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
import voyageai
import PyPDF2
from transformers import AutoTokenizer, AutoModel
import torch
import numpy as np
import json
from config import PAPERS_DIR, CHROMA_DIR, VOYAGEAI_API_KEY

MODELS = {
    "Salesforce/SFR-Embedding-Mistral": "local",
    "voyage-lite-02-instruct": "voyageai"
    # "intfloat/multilingual-e5-large": "local"
}

# Load models
loaded_models = {}
vo = voyageai.Client(api_key=VOYAGEAI_API_KEY)

for name, src in MODELS.items():
    if src == "local":
        loaded_models[name] = SentenceTransformer(name)
    else:
        loaded_models[name] = "voyage_api"

# Load manifest metadata

def load_manifest():
    manifest_path = os.path.join(PAPERS_DIR, "manifest.json")
    if os.path.exists(manifest_path):
        with open(manifest_path, "r") as f:
            papers = json.load(f)
    
    manifest_dict = {}
    for paper in papers:
        pdf_name = f"{paper['id']}.pdf"
        manifest_dict[pdf_name] = paper

    return manifest_dict

# Embed text

def embed_text(model_name, texts):
    if model_name == "voyage-lite-02-instruct":
        return vo.embed(texts, model=model_name).embeddings
    else:
        return loaded_models[model_name].encode(texts)

# Extract texts from PDFs

def extract_text_from_pdf(path):
    text = ""
    try:
        with open(path, "rb") as f:
            reader = PyPDF2.PdfReader(f)
            for page in reader.pages:
                text += page.extract_text() + "\n"
    except (PyPDF2.errors.PdfReadError, Exception) as e:
        print(f"  Error reading PDF: {e}")
        return None
    return text

# Extract conclusions from texts

def extract_conclusions(text):
    lines = text.split(". ")
    return [l for l in lines if "conclud" in l.lower() or "finding" in l.lower()]

# ChromaDB

def get_chroma_client(db_name):
    return chromadb.Client(Settings(
        anonymized_telemetry=False,
        allow_reset=True,
        persist_directory=f"{CHROMA_DIR}/{db_name}"
    ))

# Checkpoint management

def load_checkpoint(checkpoint_path):
    """Load checkpoint to resume processing from where we left off"""
    if os.path.exists(checkpoint_path):
        with open(checkpoint_path, "r") as f:
            return json.load(f)
    return {}

def save_checkpoint(checkpoint_path, checkpoint_data):
    """Save checkpoint after processing each paper"""
    with open(checkpoint_path, "w") as f:
        json.dump(checkpoint_data, f, indent=2)

# Main pipeline

def process_papers():
    manifest = load_manifest()
    checkpoint_path = os.path.join(CHROMA_DIR, "checkpoint.json")
    checkpoint = load_checkpoint(checkpoint_path)

    print("Starting embedding extraction pipeline...")
    print(f"Found {len([f for f in os.listdir(PAPERS_DIR) if f.endswith('.pdf')])} PDF files")

    for model_idx, model in enumerate(MODELS.keys(), 1):
        print(f"\n{'='*60}")
        print(f"Processing with model {model_idx}/{len(MODELS)}: {model}")
        print(f"{'='*60}")

        # Initialize checkpoint for this model if not exists
        if model not in checkpoint:
            checkpoint[model] = {"processed_papers": [], "total_conclusions": 0}

        client = get_chroma_client(model)
        collection = client.get_or_create_collection(
            name="conclusions",
            metadata={"model": model}
        )

        pdf_files = [f for f in os.listdir(PAPERS_DIR) if f.endswith(".pdf")]
        processed_count = len(checkpoint[model]["processed_papers"])

        print(f"Resuming: {processed_count}/{len(pdf_files)} papers already processed")

        for pdf_idx, pdf in enumerate(pdf_files, 1):
            # Skip if already processed
            if pdf in checkpoint[model]["processed_papers"]:
                continue

            print(f"\n[{pdf_idx}/{len(pdf_files)}] Processing: {pdf}")

            path = os.path.join(PAPERS_DIR, pdf)
            text = extract_text_from_pdf(path)

            if text is None:
                print(f"  Failed to read PDF (corrupted or malformed), skipping")
                checkpoint[model]["processed_papers"].append(pdf)
                save_checkpoint(checkpoint_path, checkpoint)
                continue

            conclusions = extract_conclusions(text)

            if not conclusions:
                print(f"  No conclusions found, skipping")
                checkpoint[model]["processed_papers"].append(pdf)
                save_checkpoint(checkpoint_path, checkpoint)
                continue

            print(f"  Found {len(conclusions)} conclusion(s)")
            embeddings = embed_text(model, conclusions)
            metadata = manifest.get(pdf, {})

            collection.add(
                ids=[f"{pdf}_{i}" for i in range(len(conclusions))],
                documents=conclusions,
                embeddings=embeddings,
                metadatas=[
                    {
                        "paper": pdf,
                        "title": metadata.get("title", ""),
                        "year": str(metadata.get("year", "")),
                        "authors": ", ".join(metadata.get("authors", [])) if isinstance(metadata.get("authors", []), list) else str(metadata.get("authors", "")),
                    }
                    for _ in conclusions
                ]
            )

            # Update checkpoint
            checkpoint[model]["processed_papers"].append(pdf)
            checkpoint[model]["total_conclusions"] += len(conclusions)
            save_checkpoint(checkpoint_path, checkpoint)
            print(f"  Embedded and saved to ChromaDB")

        client.persist()
        print(f"\nModel {model} complete: {len(checkpoint[model]['processed_papers'])} papers, {checkpoint[model]['total_conclusions']} total conclusions")

if __name__ == "__main__":
    os.makedirs(CHROMA_DIR, exist_ok=True)
    process_papers()
    print("\n" + "="*60)
    print("Embedding extraction complete!")
    print("="*60)





