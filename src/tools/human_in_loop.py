def human_clarification(prompt):
    """Prompt user for clarification."""
    print(f"\n Clarification needed: {prompt}")
    return input("Please provide the correct table name: ").strip()

def human_table_confirmation(nl_text, candidate_tables, extracted_entities):
    """Enhanced human confirmation with candidate table suggestions."""
    print(f"\n Query Analysis:")
    print(f"Original query: {nl_text}")
    print(f"Extracted entities: {', '.join(extracted_entities)}")
    print(f"\n Found {len(candidate_tables)} potential table(s):")
    
    if not candidate_tables:
        return human_clarification("No candidate tables found")
    
    # Display candidates with details
    for i, table in enumerate(candidate_tables, 1):
        confidence_bar = "█" * int(table['confidence_score'] * 10) + "░" * (10 - int(table['confidence_score'] * 10))
        print(f"\n{i}. Table: {table['table_name']}")
        print(f"   Confidence: {confidence_bar} ({table['confidence_score']:.2%})")
        print(f"   Reason: {table['reason']}")
        if 'schema_preview' in table:
            print(f"   Preview: {table['schema_preview']}")
    
    print(f"\nOptions:")
    for i, table in enumerate(candidate_tables, 1):
        print(f"{i}. {table['table_name']}")
    print(f"{len(candidate_tables) + 1}. None of the above (manual entry)")
    print("0. Cancel")
    
    while True:
        try:
            choice = input(f"\nSelect option (0-{len(candidate_tables) + 1}): ").strip()
            
            if choice == "0":
                return None
            elif choice == str(len(candidate_tables) + 1):
                return human_clarification("Please specify the correct table name")
            else:
                choice_idx = int(choice) - 1
                if 0 <= choice_idx < len(candidate_tables):
                    selected_table = candidate_tables[choice_idx]['table_name']
                    confirm = input(f"Confirm selection: {selected_table}? (y/n): ").strip().lower()
                    if confirm in ['y', 'yes']:
                        return selected_table
                    else:
                        continue
                else:
                    print("Invalid option. Please try again.")
        except ValueError:
            print("Please enter a valid number.")
        except KeyboardInterrupt:
            print("\nOperation cancelled.")
            return None

def display_analysis_results(table_name, sql_command_type, entities, confidence=None):
    """Display the analysis results to the user."""
    print(f"\n Analysis Complete:")
    print(f"Selected Table: {table_name}")
    print(f"SQL Command Type: {sql_command_type}")
    print(f"Relevant Entities: {', '.join(entities)}")
    if confidence:
        print(f"Confidence: {confidence:.2%}")
    print("-" * 50)