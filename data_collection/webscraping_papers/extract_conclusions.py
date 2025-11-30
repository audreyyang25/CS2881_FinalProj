import os
import json
import re
import pdfplumber
from pathlib import Path
from tqdm import tqdm

def extract_text_from_pdf(pdf_path):
    """Extract all text from a PDF file"""
    try:
        with pdfplumber.open(pdf_path) as pdf:
            text = ""
            for page in pdf.pages:
                # Use layout mode to preserve spacing
                page_text = page.extract_text(layout=False, x_tolerance=2, y_tolerance=3)
                if page_text:
                    text += page_text + "\n"
            return text
    except Exception as e:
        # Try alternate method with PyPDF2 if pdfplumber fails
        try:
            import PyPDF2
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                text = ""
                for page in pdf_reader.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n"
                return text
        except Exception as e2:
            print(f"Error reading {pdf_path}: {e}")
            return None

def find_conclusion_section(text):
    """
    Extract the conclusion section from academic paper text.
    Looks for common section headers like "Conclusion", "Conclusions",
    "Discussion and Conclusion", etc.
    """
    if not text:
        return None

    # Common conclusion section headers (case insensitive)
    conclusion_patterns = [
        r'\n\s*(\d+\.?\s*)?(Conclusion|Conclusions|Concluding Remarks|Summary and Conclusion|Discussion and Conclusion|Final Remarks)\s*\n',
        r'\n\s*(VI|V|VII|VIII|IX|X)\.?\s*(Conclusion|Conclusions|Concluding Remarks)\s*\n',
    ]

    # Try to find conclusion section
    for pattern in conclusion_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            # Extract text from this point onwards
            start_idx = match.end()

            # Try to find where the conclusion ends (References, Acknowledgments, etc.)
            end_patterns = [
                r'\n\s*(\d+\.?\s*)?(References|Bibliography|Acknowledgments|Acknowledgements|Appendix)\s*\n',
                r'\n\s*(References|REFERENCES|Bibliography|BIBLIOGRAPHY)\s*\n',
            ]

            end_idx = len(text)
            for end_pattern in end_patterns:
                end_match = re.search(end_pattern, text[start_idx:], re.IGNORECASE)
                if end_match:
                    end_idx = start_idx + end_match.start()
                    break

            conclusion_text = text[start_idx:end_idx].strip()

            # Clean up the text
            conclusion_text = clean_conclusion_text(conclusion_text)

            # Only return if we got substantial text (more than 50 chars)
            if len(conclusion_text) > 50:
                return conclusion_text

    return None

def clean_conclusion_text(text):
    """Clean up extracted conclusion text"""
    # Stop at common reference patterns
    ref_patterns = [
        r'\[\d+\]\s+[A-Z]',  # [1] AuthorName pattern
        r'^\s*\[\d+\]',       # Numbered references at start of line
    ]

    for pattern in ref_patterns:
        match = re.search(pattern, text, re.MULTILINE)
        if match:
            text = text[:match.start()]
            break

    # Remove excessive whitespace but preserve paragraph breaks
    text = re.sub(r' +', ' ', text)  # Multiple spaces to single space
    text = re.sub(r'\n\s*\n\s*\n+', '\n\n', text)  # Multiple newlines to double newline

    # Remove page numbers (standalone numbers)
    text = re.sub(r'\n\s*\d{1,3}\s*\n', '\n', text)

    # Remove common artifacts
    text = re.sub(r'Fig\.\s*\d+', '', text)
    text = re.sub(r'Figure\s*\d+', '', text)
    text = re.sub(r'Table\s*\d+', '', text)

    # Remove citation brackets like [1,2,3] or [Smith et al.]
    text = re.sub(r'\[\d+(?:,\s*\d+)*\]', '', text)

    # Trim
    text = text.strip()

    # Limit length (keep first 1500 chars if too long)
    if len(text) > 1500:
        # Try to cut at a sentence boundary
        truncated = text[:1500]
        last_period = truncated.rfind('.')
        if last_period > 1000:  # Only use if we get a decent amount
            text = truncated[:last_period + 1]
        else:
            text = truncated + "..."

    return text

def get_paper_title(pdf_path):
    """Extract paper title from filename"""
    filename = Path(pdf_path).stem
    # Clean up the filename to make a readable title
    title = filename.replace('_', ' ').replace('-', ' ')
    return title

def sanitize_filename(s):
    """Sanitize string to match PDF filename convention (from downloader.py)"""
    keep = (" ", ".", "_", "-")
    return "".join(c for c in s if c.isalnum() or c in keep).rstrip()

def load_manifest(manifest_path):
    """Load the manifest.json file and create a mapping from sanitized filename to metadata"""
    try:
        with open(manifest_path, 'r', encoding='utf-8') as f:
            manifest = json.load(f)

        # Create a mapping from sanitized filename to metadata
        # This matches how downloader.py creates PDF filenames
        filename_map = {}
        for entry in manifest:
            title = entry.get('title', entry.get('id', 'paper'))
            # Sanitize the title the same way downloader.py does
            sanitized = sanitize_filename(title)[:180]  # Match the 180 char limit

            filename_map[sanitized] = {
                'year': entry.get('year', -1),
                'authors': entry.get('authors', []),
                'title': title
            }

        print(f"Loaded manifest with {len(filename_map)} entries")
        return filename_map
    except Exception as e:
        print(f"Warning: Could not load manifest.json: {e}")
        return {}

def match_pdf_to_manifest(pdf_filename, filename_map):
    """
    Try to match a PDF filename to a manifest entry.
    Returns (year, title) tuple, or (-1, filename) if no match.
    """
    # Get the base filename without extension
    filename_base = Path(pdf_filename).stem

    # Try exact match (should work most of the time now)
    if filename_base in filename_map:
        entry = filename_map[filename_base]
        return entry['year'], entry['title']

    # Fallback: try to find close matches
    # Sometimes PDFs might have slight variations
    filename_lower = filename_base.lower()

    for sanitized_name, entry in filename_map.items():
        sanitized_lower = sanitized_name.lower()

        # Check if they're very similar (one contains the other)
        if filename_lower == sanitized_lower:
            return entry['year'], entry['title']

        # Check for substring match (with minimum length requirement)
        if len(filename_lower) > 20 and len(sanitized_lower) > 20:
            if filename_lower in sanitized_lower or sanitized_lower in filename_lower:
                return entry['year'], entry['title']

    # No match found - return default
    return -1, filename_base

def process_pdfs(papers_dir, output_file, manifest_path=None, limit=None):
    """
    Process all PDFs in the papers directory and extract conclusions.

    Args:
        papers_dir: Directory containing PDF files
        output_file: Output JSONL file path
        manifest_path: Path to manifest.json file (optional)
        limit: Optional limit on number of PDFs to process
    """
    # Load manifest if provided
    title_map = {}
    if manifest_path and os.path.exists(manifest_path):
        title_map = load_manifest(manifest_path)
    else:
        print("Warning: No manifest provided. Using Date=-1 for all papers.")

    pdf_files = list(Path(papers_dir).glob("*.pdf"))

    if limit:
        pdf_files = pdf_files[:limit]

    print(f"Found {len(pdf_files)} PDF files")

    results = []
    successful = 0
    failed = 0
    matched_years = 0

    for pdf_path in tqdm(pdf_files, desc="Extracting conclusions"):
        # Extract text from PDF
        text = extract_text_from_pdf(pdf_path)

        if not text:
            failed += 1
            continue

        # Find conclusion section
        conclusion = find_conclusion_section(text)

        if conclusion:
            # Try to match PDF to manifest and get year
            year, title = match_pdf_to_manifest(pdf_path.name, title_map)

            if year != -1:
                matched_years += 1

            # Format as needed for claim_embedding.py
            entry = {
                "Date": year,
                "Claim": conclusion,
                "Source": title,  # Extra field for reference
                "File": pdf_path.name
            }

            results.append(entry)
            successful += 1
        else:
            failed += 1

    # Write to JSONL file
    os.makedirs(Path(output_file).parent, exist_ok=True)

    with open(output_file, "w", encoding="utf-8") as f:
        for entry in results:
            f.write(json.dumps(entry) + "\n")

    print(f"\n{'='*60}")
    print(f"Extraction Complete")
    print(f"{'='*60}")
    print(f"Successful extractions: {successful}")
    print(f"Failed extractions: {failed}")
    print(f"Total PDFs processed: {len(pdf_files)}")
    if title_map:
        print(f"Papers matched to manifest: {matched_years}/{successful} ({100*matched_years/successful if successful > 0 else 0:.1f}%)")
    print(f"Output file: {output_file}")
    print(f"{'='*60}")

    return results

def main():
    # Configuration
    papers_dir = "../../data/papers"
    manifest_path = "../../data/papers/manifest.json"
    output_file = "../../data/paper_conclusions.jsonl"

    # Optional: set a limit for testing (None = process all)
    limit = None  # Change to a number like 10 for testing

    print("="*60)
    print("PDF Conclusion Extractor")
    print("="*60)
    print(f"Papers directory: {papers_dir}")
    print(f"Manifest file: {manifest_path}")
    print(f"Output file: {output_file}")
    if limit:
        print(f"Processing limit: {limit} files")
    print("="*60)
    print()

    results = process_pdfs(papers_dir, output_file, manifest_path=manifest_path, limit=limit)

    # Show sample
    if results:
        print(f"\nSample conclusion (first result):")
        print(f"Title: {results[0]['Source']}")
        print(f"Year: {results[0]['Date']}")
        print(f"Conclusion: {results[0]['Claim'][:200]}...")

if __name__ == "__main__":
    main()
