import sqlite3
from typing import TypedDict
from utils.llm_client import init_deepseek_client, verify_sql_with_deepseek, load_config
from langchain_core.messages import SystemMessage
import logging
from .state import AgentState

logger = logging.getLogger(__name__)

def data_operator_node(state: AgentState) -> AgentState:
    """Verify, correct, or execute SQL query."""
    logger.info("Entering data_operator_node")
    try:
        api_key = init_deepseek_client("config/settings.yaml")
        nl_text = state["messages"][-1].content
        generated_sql = state["generated_sql"]
        schema = state["schema"]
        sql_command_type = state["sql_command_type"]
        
        if not generated_sql:
            raise ValueError("No generated SQL available")

        # Verify with DeepSeek
        verification = verify_sql_with_deepseek(api_key, nl_text, generated_sql, schema, sql_command_type)
        
        status = verification.get("status", "error")
        corrected_sql = verification.get("corrected_sql", generated_sql)
        
        # if status == "perfect" or status == "corrected":
        if status in {"perfect", "corrected"}:
            # Execute if valid
            config = load_config("config/settings.yaml")
            conn = sqlite3.connect(config['db_path'])
            cursor = conn.cursor()
            # cursor.execute(corrected_sql)
            # Strip any Markdown fences from the SQL
            from utils.llm_client import clean_sql
            sql_to_run = clean_sql(corrected_sql)
            logger.info(f"Executing cleaned SQL:\n{sql_to_run}")
            cursor.execute(sql_to_run)
            results = cursor.fetchall()
            column_names = [desc[0] for desc in cursor.description]
            state["query_results"] = [dict(zip(column_names, row)) for row in results]
            conn.close()
            state["messages"].append(SystemMessage(content="Query executed successfully"))
            logger.info("Query executed; results stored in state")
        elif status == "incomplete":
            # Increase max_tokens and retry sql_query_generator
            state["retry_count"] = state.get("retry_count", 0) + 1
            if state["retry_count"] > 2:
                raise ValueError("Max retries exceeded for SQL generation")
            state["current_max_tokens"] = state.get("current_max_tokens", 1000) + 500
            state["messages"].append(SystemMessage(content="Incomplete query detected; retrying generation"))
            logger.warning("Incomplete query; triggering retry")
            return state  # Will loop back via conditional
        else:
            raise ValueError("SQL verification failed")
        state["final_sql"] = corrected_sql
    except Exception as e:
        logger.error(f"Error in data_operator_node: {e}")
        state["query_results"] = []
        state["messages"].append(SystemMessage(content=f"Error in data operation: {e}"))
    return state