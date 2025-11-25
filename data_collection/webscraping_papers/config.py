import os
from dotenv import load_dotenv
load_dotenv()

# API Keys
SEMANTIC_SCHOLAR_KEY = os.getenv("SEMANTIC_SCHOLAR_KEY", "")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

# Paths
PAPERS_DIR = "../data/papers"
TEXT_DIR = "../data/texts"
CHUNKS_DIR = "../data/chunks"
DB_PATH = "../data/papers.db"

# Pipeline params
MAX_RESULTS = 100
MIN_RELEVANCE_YEAR = 2000
CHUNK_SIZE = 2500
CHUNK_OVERLAP = 200