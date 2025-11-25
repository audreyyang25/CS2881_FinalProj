import requests, json, os
from tqdm import tqdm
import arxiv
from config import SEMANTIC_SCHOLAR_KEY, PAPERS_DIR, MAX_RESULTS

os.makedirs(PAPERS_DIR, exist_ok=True)

def search_semantic_scholar(query, limit=100):
    """
    Search Semantic Scholar for papers matching the query
    """
    endpoint = "https://api.semanticscholar.org/graph/v1/paper/search"
    params = {
        "query": query,
        "limit": limit,
        "fields": "title,authors,year,venue,abstract,externalIds,openAccessPdf,url"
    }
    headers = {}
    if SEMANTIC_SCHOLAR_KEY:
        headers["x-api-key"] = SEMANTIC_SCHOLAR_KEY
    resp = requests.get(endpoint, params=params, headers=headers)   
    resp.raise_for_status()
    return resp.json()

def search_arxiv(query, max_results=100):
    """
    Search arXiv for papers matching the query
    """
    client = arxiv.Client()
    search = arxiv.Search(
        query=query,
        max_results=max_results,
        sort_by=arxiv.SortCriterion.Relevance
    )
    results = []
    for result in search.results():
        results.append({
            "title": result.title,
            "authors": [author.name for author in result.authors],
            "year": result.published.year,
            "pdf_url": result.pdf_url,
            "id": result.get_short_id(),
            "abstract": result.summary,
            "source": "arXiv"
        })
    return results

def build_manifest(queries):
    manifest = []
    for q in queries:
        print("Semantic Scholar search for:", q)
        data = search_semantic_scholar(q, limit=MAX_RESULTS)
        for item in data.get("data", []):
            paper = {
                "title": item.get("title"),
                "authors": [author.get("name") for author in item.get("authors", [])],
                "year": item.get("year"),
                "abstract": item.get("abstract"),
                "sematicscholar_url": f"https://www.semanticscholar.org/paper/{item.get('paperId')}", # may have to check if 'paperId' exists
                "openAccessPdf": item.get("openAccessPdf"),
                "source": "Semantic Scholar",
                "id": item.get("paperId")
            }
            manifest.append(paper)
        
    for q in queries:
        print("arXiv search for:", q)
        for r in search_arxiv(q, max_results=50):
            manifest.append(r)

    # Write manifest to file
    with open(os.path.join(PAPERS_DIR, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    print(f"Manifest saved with {len(manifest)} papers to {PAPERS_DIR}/manifest.json.")
    return manifest

if __name__ == "__main__":
    queries = [
        "AI safety", "AI alignment", "alignment problem", "robustness machine learning",
        "interpretability AI", "reward hacking", "corrigibility AI", "adversarial robustness",
        "safe exploration reinforcement learning", "value alignment", "AI ethics",
        "human values", "AI governance", "machine ethics", "trustworthy AI",
    ]
    build_manifest(queries)