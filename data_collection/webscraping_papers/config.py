import os
from dotenv import load_dotenv
load_dotenv()

# API Keys
SEMANTIC_SCHOLAR_KEY = os.getenv("SEMANTIC_SCHOLAR_KEY", "")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")

# Paths
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
PAPERS_DIR = os.path.join(ROOT_DIR, "data", "papers")
TEXT_DIR = os.path.join(ROOT_DIR, "data", "texts")
CHUNKS_DIR = os.path.join(ROOT_DIR, "data", "chunks")
DB_PATH = os.path.join(ROOT_DIR, "data", "papers.db")

# Pipeline params
MAX_RESULTS = 100
MIN_RELEVANCE_YEAR = 2000
CHUNK_SIZE = 2500
CHUNK_OVERLAP = 200

# Rate limiting
REQUEST_DELAY = 2  # Seconds to wait between API requests
MAX_RETRIES = 5  # Maximum number of retries for failed requests