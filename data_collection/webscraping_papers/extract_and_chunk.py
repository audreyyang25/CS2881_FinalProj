import os, json
import pdfplumber
from tqdm import tqdm
from config import PAPERS_DIR, TEXT_DIR, CHUNKS_DIR, CHUNK_SIZE, CHUNK_OVERLAP

os.makedirs(TEXT_DIR, exist_ok=True)
os.makedirs(CHUNKS_DIR, exist_ok=True)

def extract_text_from_pdf(pdf_path):
    text_pages = []
    with pdfplumber.open(pdf_path) as pdf:
        for p in pdf.pages:
            text = p.extract_text()
            if text:
                text_pages.append(text)
    return "\n\n".join(text_pages)

def chunk_text(text, chunk_size=2500, overlap=200):
    chunks = []
    start = 0
    L = len(text)
    while start < L:
        end = min(start + chunk_size, L)
        chunk = text[start:end]
        chunks.append(chunk)
        start = end - overlap
        if start < 0:
            start = 0
        if start >= L:
            break
    return chunks

def run():
    downloads_file = os.path.join(PAPERS_DIR, "downloads.json")
    if not os.path.exists(downloads_file):
        raise FileNotFoundError("Run downloader first.")
    with open(downloads_file, "r", encoding="utf-8") as f:
        downloads = json.load(f)
    index = []
    for d in tqdm(downloads):
        pdf_path = d["path"]
        title = d["title"]
        try:
            text = extract_text_from_pdf(pdf_path)
            txt_path = os.path.join(TEXT_DIR, os.path.basename(pdf_path) + ".txt")
            with open(txt_path, "w", encoding="utf-8") as f:
                f.write(text)
            chunks = chunk_text(text, size=CHUNK_SIZE, overlap=CHUNK_OVERLAP)
            chunk_paths = []
            for i, c in enumerate(chunks):
                cp = os.path.join(CHUNKS_DIR, os.path.basename(pdf_path) + f".chunk{i}.txt")
                with open(cp, "w", encoding="utf-8") as cf:
                    cf.write(c)
                chunk_paths.append(cp)
            index.append({"title": title, "pdf_path": pdf_path, "txt_path": txt_path, "chunks": chunk_paths})
        except Exception as e:
            print("Error extracting", pdf_path, e)

    with open(os.path.join(PAPERS_DIR, "text_index.json"), "w", encoding="utf-8") as f:
        json.dump(index, f, indent=2, ensure_ascii=False)
    print("Extraction and chunking done.")

if __name__ == "__main__":
    run()
