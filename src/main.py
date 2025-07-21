from agents.schema_retriever import run_agent
import sys

def main():
    """Enhanced main function with better user interaction."""
    print("=" * 50)
    
    while True:
        try:
            print("\nEnter your natural language query (or 'quit' to exit):")
            nl_text = input("> ").strip()
            
            if nl_text.lower() in ['quit', 'exit', 'q']:
                print("Goodbye!")
                break
                
            if not nl_text:
                print("Please enter a valid query.")
                continue
            
            print(f"\n Processing query: {nl_text}")
            print("-" * 50)
            
            schema_result = run_agent(nl_text)
            
            print(f"\n Result:")
            print("=" * 50)
            print(schema_result)
            print("=" * 50)
            
        except KeyboardInterrupt:
            print("\n\nOperation cancelled. Goodbye! ")
            break
        except Exception as e:
            print(f"\nError: {e}")
            continue

if __name__ == "__main__":
    main()