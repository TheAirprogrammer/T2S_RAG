from typing import TypedDict
from utils.llm_client import init_codestral_client, generate_sql_with_codestral
from langchain_core.messages import SystemMessage
import logging
from .state import AgentState  # Import from shared state module

logger = logging.getLogger(__name__)

def sql_query_generator_node(state: AgentState) -> AgentState:
    """Generate SQL query based on NL text, schema, and command type."""
    logger.info("Entering sql_query_generator_node")
    try:
        api_key = init_codestral_client("config/settings.yaml")
        nl_text = state["messages"][-1].content
        schema = state["schema"]
        sql_command_type = state["sql_command_type"]
        extracted_entities = state["extracted_entities"]
        
        if not schema:
            raise ValueError("No schema available for SQL generation")
        
        sql_query = generate_sql_with_codestral(
            api_key, nl_text, schema, sql_command_type, extracted_entities
        )
        
        state["generated_sql"] = sql_query
        state["messages"].append(SystemMessage(content=f"Generated SQL: {sql_query}"))
        logger.info(f"Generated SQL: {sql_query}")
    except Exception as e:
        logger.error(f"Error in sql_query_generator_node: {e}")
        state["generated_sql"] = ""
        state["messages"].append(SystemMessage(content=f"Error generating SQL: {e}"))
    return state
