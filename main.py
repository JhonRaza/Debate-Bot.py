import argparse
import sys
sys.path.append('src')
from src.fact_checking import fact_check_query_cli, query_huggingface_api_cli
import warnings
warnings.simplefilter("ignore")
from src.retrieval import populate_chromadb 
import os

# Default values for model settings
DEFAULT_MODEL = "google/gemma-2-27b-it"
DEFAULT_MAX_TOKENS = 200
DEFAULT_TEMPERATURE = 0.7
DEFAULT_COSINE_DISTANCE = 0.85

def main():
    parser = argparse.ArgumentParser(description="Fact-Checking System (CLI) - RAG-Based with LLM Support",
                                     epilog="Example:\n  python main.py 'China has a great economy'\n",
                                    formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("claim", type=str, help="Enter a claim to fact-check.")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL, help="Select LLM for text generation.")
    parser.add_argument("--max_tokens", type=int, default=DEFAULT_MAX_TOKENS, help="Max tokens for LLM response.")
    parser.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE, help="LLM temperature setting.")
    parser.add_argument("--cosine_threshold", type=float, default=DEFAULT_COSINE_DISTANCE, help="Cosine similarity threshold.")

    args = parser.parse_args()
    user_query = args.claim.strip()

    if not user_query:
        print("\nPlease enter a valid claim to fact-check.")
        return
    
    #checking for chromadb
    if not os.path.exists("chromadb_store"):
        print("\nPopulating ChromaDB with fact-checking data...")
        populate_chromadb()
    
    print("\n🔍 Checking claim:", user_query)
    
    try:
        response, similar_claim = fact_check_query_cli(user_query, args.model, args.max_tokens, args.temperature, args.cosine_threshold)
        if response != "NOPEE":
            print("\n🧐 Fact-check result:")
            print(response)

        if similar_claim:
            print("\n🔍 Similar claim found in dataset:")
            print(f"➡️ {similar_claim['statement']}")
            print(f"📊 Label: {similar_claim['label']}")
        else:
            print("\n⚠️ No matching fact-check found. Using LLM for analysis...")
            llm_response, _ = query_huggingface_api_cli(user_query, args.model, args.max_tokens, args.temperature, None)
            print("\n🤖 LLM-Based fact-check result:")
            print(llm_response)

    except Exception as e:
        print(f"\n❌ An error occurred: {str(e)}")

if __name__ == "__main__":
    main()
