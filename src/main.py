from agents.schema_retriever import run_agent

def main():
    """Enhanced main function with better user interaction."""
    print("🚀 Enhanced T2S RAG Schema Retriever & SQL Generator")
    print("=" * 60)
    
    while True:
        try:
            print("\nEnter your natural language query (or 'quit' to exit):")
            nl_text = input("> ").strip()
            
            if nl_text.lower() in ['quit', 'exit', 'q']:
                print("Goodbye! 👋")
                break
                
            if not nl_text:
                print("Please enter a valid query.")
                continue
            
            print(f"\n🔄 Processing query: {nl_text}")
            print("-" * 50)
            
            result = run_agent(nl_text)
            
            if "error" in result:
                print(f"\n❌ Error: {result['error']}")
                continue
            
            print(f"\n📊 Retrieved Schema:")
            print("=" * 50)
            print(result.get('schema', 'No schema retrieved'))
            
            print(f"\n🛠️ Generated SQL Query:")
            print("=" * 50)
            print(result.get('generated_sql', 'No SQL generated'))
            print("=" * 50)
            
        except KeyboardInterrupt:
            print("\n\nOperation cancelled. Goodbye! 👋")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}")
            continue
        print(f"\n🛠️ Final SQL Query:\n{result.get('final_sql', 'No SQL finalized')}")
        print(f"\n📊 Query Results:")
        if result.get('query_results'):
            for row in result['query_results']:
                print(row)
        else:
            print("No results or error occurred")

if __name__ == "__main__":
    main()
