import json
from openai import OpenAI
from dotenv import load_dotenv
import os

def main():
    # Load API key
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY not found in .env file.")

    client = OpenAI(api_key=api_key)

    input_file = "alien_claims.jsonl"
    output_file = "alien_claims_embedded.jsonl"

    with open(input_file, "r", encoding="utf-8") as infile, \
         open(output_file, "w", encoding="utf-8") as outfile:

        for line in infile:
            line = line.strip()
            if not line:
                continue

            obj = json.loads(line)
            claim_text = obj["Claim"]

            # Create embedding
            response = client.embeddings.create(
                model="text-embedding-3-small",
                input=claim_text
            )

            embedding = response.data[0].embedding
            obj["embedding"] = embedding

            # Write updated JSONL entry
            outfile.write(json.dumps(obj) + "\n")

    print("Finished writing:", output_file)

if __name__ == "__main__":
    main()
