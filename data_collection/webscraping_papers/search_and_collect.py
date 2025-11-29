import requests, json, os, time
from tqdm import tqdm
import arxiv
import sys

# Add parent directory to path to import config
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import SEMANTIC_SCHOLAR_KEY, PAPERS_DIR, MAX_RESULTS, REQUEST_DELAY, MAX_RETRIES

os.makedirs(PAPERS_DIR, exist_ok=True)

def search_semantic_scholar(query, limit=100, max_retries=MAX_RETRIES):
    """
    Search Semantic Scholar for papers matching the query with retry logic
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

    for attempt in range(max_retries):
        try:
            resp = requests.get(endpoint, params=params, headers=headers, timeout=30)

            # Handle rate limiting
            if resp.status_code == 429:
                retry_after = int(resp.headers.get('Retry-After', 60))
                print(f"Rate limited. Waiting {retry_after} seconds before retry...")
                time.sleep(retry_after)
                continue

            resp.raise_for_status()
            return resp.json()

        except requests.exceptions.RequestException as e:
            wait_time = 2 ** attempt  # Exponential backoff: 1, 2, 4, 8, 16 seconds
            print(f"Error on attempt {attempt + 1}/{max_retries}: {e}")
            if attempt < max_retries - 1:
                print(f"Retrying in {wait_time} seconds...")
                time.sleep(wait_time)
            else:
                print(f"Failed to fetch results for query '{query}' after {max_retries} attempts")
                return {"data": []}  # Return empty result instead of crashing

    return {"data": []}

def search_arxiv(query, max_results=100, max_retries=3):
    """
    Search arXiv for papers matching the query with retry logic
    """
    for attempt in range(max_retries):
        try:
            client = arxiv.Client(
                page_size=100,
                delay_seconds=3,  # Built-in delay between API calls
                num_retries=3
            )
            search = arxiv.Search(
                query=query,
                max_results=max_results,
                sort_by=arxiv.SortCriterion.Relevance
            )
            results = []
            for result in client.results(search):
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
        except Exception as e:
            wait_time = 2 ** attempt
            print(f"arXiv error on attempt {attempt + 1}/{max_retries}: {e}")
            if attempt < max_retries - 1:
                print(f"Retrying in {wait_time} seconds...")
                time.sleep(wait_time)
            else:
                print(f"Failed to fetch arXiv results for query '{query}' after {max_retries} attempts")
                return []
    return []

def build_manifest(queries, delay_between_queries=REQUEST_DELAY):
    manifest = []

    # Search Semantic Scholar
    print(f"\n=== Searching Semantic Scholar for {len(queries)} queries ===")
    for i, q in enumerate(queries, 1):
        print(f"[{i}/{len(queries)}] Semantic Scholar search for: {q}")
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

        # Add delay between queries to avoid rate limiting
        if i < len(queries):
            print(f"Waiting {delay_between_queries} seconds before next query...")
            time.sleep(delay_between_queries)

    # Search arXiv
    print(f"\n=== Searching arXiv for {len(queries)} queries ===")
    for i, q in enumerate(queries, 1):
        print(f"[{i}/{len(queries)}] arXiv search for: {q}")
        for r in search_arxiv(q, max_results=50):
            manifest.append(r)

        # Add delay between queries
        if i < len(queries):
            print(f"Waiting {delay_between_queries} seconds before next query...")
            time.sleep(delay_between_queries)

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