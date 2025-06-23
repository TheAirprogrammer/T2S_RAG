def human_clarification(prompt):
    """Prompt user for clarification."""
    print(f"Clarification needed: {prompt}")
    return input("Please provide the correct table name or confirm: ")