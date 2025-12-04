# CS2881 Final Project: Embedding Analysis for Ethical Claims

This project performs comprehensive embedding analysis on ethical claims and other text data using multiple embedding models. It includes data collection, embedding generation, visualization, classification, and evaluation pipelines.

## Project Overview

This project analyzes text embeddings across multiple domains (ethical claims, research papers, social media posts) using three different embedding models:
- **OpenAI** (`text-embedding-3-small`)
- **Voyage AI** (`voyage-lite-02-instruct`)
- **mxbai** (`mixedbread-ai/mxbai-embed-large-v1`)

The analysis includes 2D projections, UMAP visualizations, classification tasks, and clustering to understand how different embedding models represent semantic information.

## Project Structure

```
CS2881_FinalProj/
├── data/                          # Data storage -- folder NOT INCLUDED in Github due to size constraints. Run files in data_collection to collect data
│   ├── papers/                    # Research paper PDFs and metadata
│   └── socialmediaposts/          # Social media post archives
├── data_collection/               # Data collection scripts
│   └── webscraping_papers/        # Paper downloading and extraction
├── embedding_ngram_aliens/        # Alien claims analysis
├── embedding_ngram_papers/        # Research paper claims analysis
├── embedding_ngram_smoking/       # Smoking-related claims analysis
├── ethical_embeddings/             # Main ethical claims analysis
│   ├── comprehensive_analysis.py   # Full analysis pipeline
│   ├── generate_embeddings.py     # Embedding generation
│   ├── eval_new_claim.py          # Evaluate new claims
│   └── outputs/                   # Analysis results and visualizations
├── config.py                      # Configuration and API keys
└── requirements.txt               # Python dependencies
```

## Setup

### Prerequisites

- Python 3.8+
- API keys for:
  - OpenAI (for OpenAI embeddings)
  - Voyage AI (for Voyage embeddings)
  - Semantic Scholar (optional, for paper collection)

### Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd CS2881_FinalProj
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up environment variables:
Create a `.env` file in the root directory with your API keys:
```
OPENAI_API_KEY=your_openai_key
VOYAGE_API_KEY=your_voyage_key
VOYAGEAI_API_KEY=your_voyage_key
SEMANTIC_SCHOLAR_KEY=your_semantic_scholar_key
ANTHROPIC_API_KEY=your_anthropic_key
```

## Usage

### 1. Generate Embeddings

Generate embeddings for ethical claims using all three models:

```bash
cd ethical_embeddings
python generate_embeddings.py
```

This will create three JSONL files:
- `openai_ethical_claims_embedded.jsonl`
- `voyage_ethical_claims_embedded.jsonl`
- `mxbai_ethical_claims_embedded.jsonl`

### 2. Run Comprehensive Analysis

Run the full analysis pipeline including visualizations, classifications, and evaluations:

```bash
cd ethical_embeddings
python comprehensive_analysis.py
```

This will:
- Load embeddings for all three models
- Compute 2D Euclidean projections
- Generate UMAP 2D embeddings
- Train and evaluate classifiers:
  - Logistic Regression on 2D projection
  - Logistic Regression on UMAP 2D
  - MLP Neural Network on full embeddings
  - Centroid-based classifier on 2D projection
- Create visualizations:
  - UMAP plots by category
  - UMAP plots with safety markers
  - 2D projection plots with centroids
- Save results to `outputs/{model_name}/` directories

### 3. Evaluate New Claims

Evaluate a new ethical claim using trained classifiers:

```bash
cd ethical_embeddings
python eval_new_claim.py
```

Modify the script to input your claim text and select which embedding model to use.

## Key Features

### 2D Euclidean Projection

The project uses a custom 2D projection method that:
- Uses the first two data points as reference (safe and unsafe)
- Projects all embeddings into a 2D space based on distances to these references
- Enables visualization and simple classification

### UMAP Visualization

- Reduces high-dimensional embeddings to 2D using UMAP
- Visualizes embeddings colored by category and safety labels
- Helps understand the structure of the embedding space

### Classification Methods

1. **Logistic Regression (2D Projection)**: Classifies based on 2D Euclidean projection
2. **Logistic Regression (UMAP 2D)**: Classifies based on UMAP-reduced embeddings
3. **MLP Neural Network**: Uses full embedding dimensions
4. **Centroid Classifier**: Simple distance-based classifier using 2D centroids

### Categories

The ethical claims dataset includes 11 categories:
- **v**: Violence & Conflict
- **p**: Privacy & Data
- **m**: Misinformation
- **h**: Harassment & Civility
- **s**: Safety & Wellbeing
- **w**: Workplace Ethics
- **e**: Environment
- **c**: Healthcare
- **f**: Family & Parenting
- **t**: Technology & AI
- **d**: Education

## Output Files

### Analysis Results

For each model, results are saved in `ethical_embeddings/outputs/{model_name}/`:
- `{model_name}_results.json`: Summary of all metrics
- `{model_name}_umap_by_category.png`: UMAP visualization by category
- `{model_name}_umap_safety.png`: UMAP with safe/unsafe markers
- `{model_name}_2d_projection.png`: 2D Euclidean projection plot

### Embedding Files

- `*_ethical_claims_embedded.jsonl`: JSONL files with text, embeddings, labels, and metadata
- `*_ethical_claims_embedded.npy`: NumPy arrays of embeddings (optional)

## Data Format

Embedded data files use JSONL format with each line containing:
```json
{
  "id": "v001",
  "text": "Claim text here...",
  "label": "safe",
  "embedding": [0.123, 0.456, ...],
  "category": "v"
}
```

## Dependencies

Key dependencies include:
- `numpy`: Numerical computations
- `scikit-learn`: Machine learning models and metrics
- `umap-learn`: Dimensionality reduction
- `matplotlib`: Plotting and visualization
- `sentence-transformers`: mxbai embeddings
- `openai`: OpenAI API client
- `voyageai`: Voyage AI API client
- `torch`: PyTorch for neural networks

See `requirements.txt` for the complete list.

## Configuration

Edit `config.py` to adjust:
- API keys and paths
- Data collection parameters (MAX_RESULTS, MIN_RELEVANCE_YEAR)
- Chunking parameters (CHUNK_SIZE, CHUNK_OVERLAP)
- Rate limiting settings

## Notes

- The first two data points in the dataset are used as reference points for 2D projection
- UMAP parameters can be adjusted in `compute_umap_embeddings()` function
- Classification uses a 75/25 train-test split with stratification
- All visualizations are saved as high-resolution PNG files (300 DPI)

## Authors

Eric Gong
Audrey Yang

## Acknowledgments

- OpenAI for embedding API
- Voyage AI for embedding API
- mixedbread.ai for mxbai embeddings
- UMAP developers for dimensionality reduction tools



