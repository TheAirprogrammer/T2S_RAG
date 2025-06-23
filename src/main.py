from agents.schema_retriever import run_agent

def main():
    nl_text = input("Enter your natural language query: ")
    schema = run_agent(nl_text)
    print(f"Table Schema:\n{schema}")

if __name__ == "__main__":
    main()