import os, json, requests
from urllib.parse import urlparse
from tqdm import tqdm
from config import PAPERS_DIR

MANIFEST = os.path.join(PAPERS_DIR, "manifest.json")

def sanitize_filename(s):
    keep = (" ", ".", "_", "-")
    return "".join(c for c in s if c.isalnum() or c in keep).rstrip()

def download_file(url, dest_path):
    try:
        r = requests.get(url, stream=True, timeout=30)
        r.raise_for_status()
        with open(dest_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
        return True
    except Exception as e:
        print("Failed to download", url, e)
        return False
    
def run():
    with open(MANIFEST, "r", encoding="utf-8") as f:
        manifest = json.load(f)
    
    os.makedirs(PAPERS_DIR, exist_ok=True)
    downloads = []
    for item in tqdm(manifest, desc="Preparing downloads"):
        pdf_url = None
        if item.get("source" == "arXiv" and item.get("pdf_url")):
            pdf_url = item["pdf_url"]
        elif item.get("openAccessPdf") and item["openAccessPdf"].get("url"):
            pdf_url = item["openAccessPdf"]["url"]
        
        if not pdf_url:
            continue

        fname = sanitize_filename(item.get("title", item.get("id", "paper")))[:180] + ".pdf"
        dest = os.path.join(PAPERS_DIR, fname)
        if os.path.exists(dest):
            continue
        ok = download_file(pdf_url, dest)
        if ok:
            downloads.append({"title": item.get("title"), "path": dest, "source_url": pdf_url})

        print(f"Downloaded {len(downloads)} papers.")
        # Save an index of downloads
        with open(os.path.join(PAPERS_DIR, "downloads.json"), "w", encoding="utf-8") as f:
            json.dump(downloads, f, indent=2, ensure_ascii=False)

if __name__ == "__main__":
    run()