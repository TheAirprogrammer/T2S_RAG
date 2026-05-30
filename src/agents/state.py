from typing import TypedDict, List
from langchain_core.messages import HumanMessage

class AgentState(TypedDict):
    messages: List[HumanMessage]
    table_name: str
    candidate_tables: List[dict]  # List of {table_name, confidence_score, reason}
    schema: str
    needs_clarification: bool
    needs_table_confirmation: bool
    is_alter: bool
    alter_command: str
    sql_command_type: str  # SELECT, INSERT, UPDATE, DELETE, etc.
    extracted_entities: List[str]  # Entities found in NL text
    generated_sql: str  # Add this field for SQL generation
    query_results: List[dict]
    final_sql: str
    retry_count: int
    current_max_tokens: int