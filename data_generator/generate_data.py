import os
import argparse
from openai import AzureOpenAI
import json
from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------
# SETUP
# ---------------------------------------------------------
# Defaults
DEFAULT_MODEL = "gpt-4o"
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_FILE = os.path.join(SCRIPT_DIR, "train.jsonl")

def load_file(filepath):
    """Reads a text file and returns content."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Missing file: {filepath}")
    with open(filepath, "r", encoding="utf-8") as f:
        return f.read()

def generate_data(num_examples):
    client = AzureOpenAI(
        api_key=os.getenv("OPENAI_API_KEY"),  
        api_version="2024-12-01-preview",
        azure_endpoint="https://aimsoa.iglb.intel.com/",
    )

    # 1. Load Context
    print("📂 Loading Fact Sheet and Prompts...")
    facts = load_file(os.path.join(SCRIPT_DIR, "fact_sheet.txt"))
    base_prompt = load_file(os.path.join(SCRIPT_DIR, "data_generation_prompt.txt"))

    # 2. Construct the Final Prompt
    # We inject the facts into the prompt template
    final_system_prompt = base_prompt.replace("{fact_sheet_content}", facts).replace("{num_examples}", str(num_examples))

    print(f"🤖 Connecting to OpenAI ({DEFAULT_MODEL})...")
    print(f"📝 Generating {num_examples} examples. This may take a minute...")

    # 3. Call API
    # We ask for the data in one shot (or you could loop for batches if >50 examples)
    response = client.chat.completions.create(
        model=DEFAULT_MODEL,
        messages=[
            {"role": "system", "content": final_system_prompt},
            {"role": "user", "content": "Generate the dataset now. Output ONLY raw JSONL."}
        ],
        temperature=0.7 # Slight creativity allowed for phrasing
    )

    # 4. Process Output
    raw_content = response.choices[0].message.content
    
    # Strip markdown code blocks if present
    clean_content = raw_content.replace("```json", "").replace("```", "").strip()

    # 5. Save
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        f.write(clean_content)

    print(f"✅ Success! Data saved to: {OUTPUT_FILE}")
    print(f"   (Preview of first line): {clean_content.splitlines()[0][:100]}...")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate synthetic training data from a Fact Sheet.")
    parser.add_argument("--count", type=int, default=50, help="Number of examples to generate")
    
    args = parser.parse_args()
    
    generate_data(args.count)