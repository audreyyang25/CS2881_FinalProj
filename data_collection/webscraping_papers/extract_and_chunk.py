import os, json
import pdfplumber
from tqdm import tqdm
from config import PAPERS_DIR, TEXT_DIR, CHUNKS_DIR, CHUNK_SIZE, CHUNK_OVERLAP

os.makedirs(TEXT_DIR, exist_ok=True)
os.makedirs(CHUNKS_DIR, exist_ok=True)

def extract_text_from_pdf_streaming(pdf_path):
    """
    Generator that yields text page by page from a PDF
    """
    with pdfplumber.open(pdf_path) as pdf:
        for page_num, page in enumerate(pdf.pages, 1):
            text = page.extract_text()
            if text:
                yield page_num, text

def extract_text_from_pdf(pdf_path):
    """
    Extract all text from PDF at once (legacy method)
    """
    text_pages = []
    with pdfplumber.open(pdf_path) as pdf:
        for p in pdf.pages:
            text = p.extract_text()
            if text:
                text_pages.append(text)
    return "\n\n".join(text_pages)

def chunk_text(text, size=2500, overlap=200):
    """
    Chunk text into overlapping segments (legacy method)
    """
    chunks = []
    start = 0
    L = len(text)
    while start < L:
        end = min(start + size, L)
        chunk = text[start:end]
        chunks.append(chunk)
        start = end - overlap
        if start < 0:
            start = 0
        if start >= L:
            break
    return chunks

def chunk_text_streaming(page_generator, size=2500, overlap=200):
    """
    Generator that yields chunks from pages as they're processed.
    Maintains overlap across page boundaries.

    Args:
        page_generator: Generator yielding (page_num, text) tuples
        size: Size of each chunk
        overlap: Overlap between consecutive chunks

    Yields:
        (chunk_index, chunk_text, page_numbers) tuples
    """
    buffer = ""
    page_numbers = []
    chunk_index = 0

    for page_num, page_text in page_generator:
        # Add page separator and new page text to buffer
        if buffer:
            buffer += "\n\n" + page_text
            page_numbers.append(page_num)
        else:
            buffer = page_text
            page_numbers = [page_num]

        # Extract as many complete chunks as possible from buffer
        while len(buffer) >= size:
            chunk = buffer[:size]
            yield chunk_index, chunk, page_numbers.copy()
            chunk_index += 1

            # Move forward in buffer, keeping overlap
            buffer = buffer[size - overlap:]
            # Keep track of which pages remain in buffer (approximate)

    # Yield remaining buffer as final chunk(s)
    start = 0
    while start < len(buffer):
        end = min(start + size, len(buffer))
        chunk = buffer[start:end]
        if chunk.strip():  # Only yield non-empty chunks
            yield chunk_index, chunk, page_numbers.copy()
            chunk_index += 1
        start = end - overlap
        if start >= len(buffer):
            break

def run_streaming():
    """
    Process PDFs page-by-page using streaming to minimize memory usage
    """
    downloads_file = os.path.join(PAPERS_DIR, "downloads.json")
    if not os.path.exists(downloads_file):
        raise FileNotFoundError("Run downloader first.")
    with open(downloads_file, "r", encoding="utf-8") as f:
        downloads = json.load(f)

    index = []
    for d in tqdm(downloads, desc="Processing PDFs"):
        pdf_path = d["path"]
        title = d["title"]
        try:
            # Open text file for streaming writes
            txt_path = os.path.join(TEXT_DIR, os.path.basename(pdf_path) + ".txt")
            chunk_paths = []

            # Stream pages and create chunks on the fly
            page_gen = extract_text_from_pdf_streaming(pdf_path)

            with open(txt_path, "w", encoding="utf-8") as txt_file:
                # Track all pages for txt file
                all_pages = []

                # Create a generator that also writes to txt file
                def page_gen_with_write():
                    for page_num, page_text in extract_text_from_pdf_streaming(pdf_path):
                        all_pages.append(page_text)
                        yield page_num, page_text

                # Stream chunks
                for chunk_idx, chunk_text, page_nums in chunk_text_streaming(
                    page_gen_with_write(),
                    size=CHUNK_SIZE,
                    overlap=CHUNK_OVERLAP
                ):
                    # Write chunk to file
                    cp = os.path.join(CHUNKS_DIR, os.path.basename(pdf_path) + f".chunk{chunk_idx}.txt")
                    with open(cp, "w", encoding="utf-8") as cf:
                        cf.write(chunk_text)
                    chunk_paths.append(cp)

                # Write complete text file
                txt_file.write("\n\n".join(all_pages))

            index.append({
                "title": title,
                "pdf_path": pdf_path,
                "txt_path": txt_path,
                "chunks": chunk_paths
            })

        except Exception as e:
            print(f"Error extracting {pdf_path}: {e}")

    with open(os.path.join(PAPERS_DIR, "text_index.json"), "w", encoding="utf-8") as f:
        json.dump(index, f, indent=2, ensure_ascii=False)
    print(f"Streaming extraction and chunking done. Processed {len(index)} PDFs.")

def run():
    """
    Process PDFs using legacy method (loads full document into memory)
    """
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
    # Use streaming by default for better memory efficiency
    run_streaming()
    # To use legacy method: run()
