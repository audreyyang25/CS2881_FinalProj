import os, json, time
import pdfplumber
from tqdm import tqdm
from config import (
    PAPERS_DIR, TEXT_DIR, OPENAI_API_KEY, ANTHROPIC_API_KEY,
    REQUEST_DELAY, MAX_RETRIES
)

os.makedirs(TEXT_DIR, exist_ok=True)

def extract_full_text_streaming(pdf_path):
    """
    Extract full text from PDF using page-by-page streaming.
    Returns the complete text without loading entire PDF into memory at once.
    """
    pages = []
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text = page.extract_text()
            if text:
                pages.append(text)
    return "\n\n".join(pages)

def summarize_with_openai(text, title="", max_retries=MAX_RETRIES):
    """
    Summarize text using OpenAI API (supports large context windows)
    """
    try:
        from openai import OpenAI
    except ImportError:
        print("OpenAI library not installed. Run: pip install openai")
        return None

    if not OPENAI_API_KEY:
        print("OPENAI_API_KEY not set in .env file")
        return None

    client = OpenAI(api_key=OPENAI_API_KEY)

    prompt = f"""Please provide a comprehensive summary of this research paper{f' titled "{title}"' if title else ''}.

Include:
1. Main research question/objective
2. Key methodology
3. Major findings
4. Conclusions and implications
5. Relevance to AI safety/alignment (if applicable)

Paper text:
{text}"""

    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model="gpt-4o",  # or "gpt-4-turbo" for larger contexts
                messages=[
                    {"role": "system", "content": "You are a research assistant specialized in summarizing academic papers, particularly in AI safety and alignment."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=2000
            )
            return response.choices[0].message.content

        except Exception as e:
            wait_time = 2 ** attempt
            print(f"OpenAI error on attempt {attempt + 1}/{max_retries}: {e}")
            if attempt < max_retries - 1:
                print(f"Retrying in {wait_time} seconds...")
                time.sleep(wait_time)
            else:
                print(f"Failed after {max_retries} attempts")
                return None
    return None

def summarize_with_anthropic(text, title="", max_retries=MAX_RETRIES):
    """
    Summarize text using Anthropic Claude API (200k+ token context window)
    """
    try:
        from anthropic import Anthropic
    except ImportError:
        print("Anthropic library not installed. Run: pip install anthropic")
        return None

    if not ANTHROPIC_API_KEY:
        print("ANTHROPIC_API_KEY not set in .env file")
        return None

    client = Anthropic(api_key=ANTHROPIC_API_KEY)

    prompt = f"""Please provide a comprehensive summary of this research paper{f' titled "{title}"' if title else ''}.

Include:
1. Main research question/objective
2. Key methodology
3. Major findings
4. Conclusions and implications
5. Relevance to AI safety/alignment (if applicable)

Paper text:
{text}"""

    for attempt in range(max_retries):
        try:
            response = client.messages.create(
                model="claude-3-5-sonnet-20241022",  # Large context window
                max_tokens=2000,
                temperature=0.3,
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )
            return response.content[0].text

        except Exception as e:
            wait_time = 2 ** attempt
            print(f"Anthropic error on attempt {attempt + 1}/{max_retries}: {e}")
            if attempt < max_retries - 1:
                print(f"Retrying in {wait_time} seconds...")
                time.sleep(wait_time)
            else:
                print(f"Failed after {max_retries} attempts")
                return None
    return None

def run_summarization(provider="anthropic", max_papers=None, delay_between_requests=REQUEST_DELAY):
    """
    Summarize PDFs using LLM without chunking.

    Args:
        provider: "anthropic" or "openai"
        max_papers: Limit number of papers to process (None for all)
        delay_between_requests: Seconds to wait between API calls
    """
    downloads_file = os.path.join(PAPERS_DIR, "downloads.json")
    if not os.path.exists(downloads_file):
        raise FileNotFoundError("Run downloader first.")

    with open(downloads_file, "r", encoding="utf-8") as f:
        downloads = json.load(f)

    if max_papers:
        downloads = downloads[:max_papers]

    summarize_fn = summarize_with_anthropic if provider == "anthropic" else summarize_with_openai

    summaries = []
    for i, d in enumerate(tqdm(downloads, desc=f"Summarizing with {provider}")):
        pdf_path = d["path"]
        title = d["title"]

        try:
            # Extract full text using streaming (memory efficient)
            print(f"\n[{i+1}/{len(downloads)}] Extracting text from: {title[:60]}...")
            text = extract_full_text_streaming(pdf_path)

            # Check if text is too long (rough estimate: 1 token ≈ 4 chars)
            estimated_tokens = len(text) / 4
            if provider == "openai" and estimated_tokens > 120000:
                print(f"  WARNING: Paper may exceed OpenAI context limit (~{int(estimated_tokens)} tokens)")
            elif provider == "anthropic" and estimated_tokens > 180000:
                print(f"  WARNING: Paper may exceed Anthropic context limit (~{int(estimated_tokens)} tokens)")

            # Summarize with LLM
            print(f"  Summarizing with {provider}...")
            summary = summarize_fn(text, title)

            if summary:
                summaries.append({
                    "title": title,
                    "pdf_path": pdf_path,
                    "summary": summary,
                    "provider": provider,
                    "estimated_tokens": int(estimated_tokens)
                })

                # Save summary to individual file
                summary_path = os.path.join(TEXT_DIR, os.path.basename(pdf_path) + ".summary.txt")
                with open(summary_path, "w", encoding="utf-8") as f:
                    f.write(f"Title: {title}\n\n")
                    f.write(f"Summary (via {provider}):\n\n")
                    f.write(summary)

                print(f"  ✓ Summary saved to {summary_path}")
            else:
                print(f"  ✗ Failed to generate summary")

            # Rate limiting delay
            if i < len(downloads) - 1:
                print(f"  Waiting {delay_between_requests} seconds before next request...")
                time.sleep(delay_between_requests)

        except Exception as e:
            print(f"  Error processing {pdf_path}: {e}")

    # Save all summaries to index
    summaries_file = os.path.join(PAPERS_DIR, "summaries.json")
    with open(summaries_file, "w", encoding="utf-8") as f:
        json.dump(summaries, f, indent=2, ensure_ascii=False)

    print(f"\n{'='*60}")
    print(f"Summarization complete!")
    print(f"  Total papers processed: {len(summaries)}/{len(downloads)}")
    print(f"  Summaries saved to: {summaries_file}")
    print(f"  Individual summaries in: {TEXT_DIR}")
    print(f"{'='*60}")

if __name__ == "__main__":
    # Choose your provider
    # "anthropic" - Claude with 200k token context (recommended for long papers)
    # "openai" - GPT-4 with 128k token context

    run_summarization(
        provider="anthropic",  # or "openai"
        max_papers=None,  # Set to a number to test on fewer papers first
        delay_between_requests=2
    )
