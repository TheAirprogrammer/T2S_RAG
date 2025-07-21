from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage, SystemMessage
from typing import TypedDict, List, Optional
from utils.llm_client import init_gemini_client, get_table_name_and_alter, find_relevant_tables_from_entities
from tools.vector_search import search_table_schema, search_relevant_tables_by_content
from tools.human_in_loop import human_table_confirmation, human_clarification
from tools.db_utils import update_db_and_vector_store
import yaml
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_config(config_path):
    try:
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    except Exception as e:
        logger.error(f"Failed to load config: {e}")
        raise

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

def thought_node(state: AgentState) -> AgentState:
    """Enhanced table name extraction with entity detection and SQL command identification."""
    logger.info("Entering enhanced thought_node")
    try:
        config = load_config("config/settings.yaml")
        client = init_gemini_client("config/settings.yaml")
        nl_text = state["messages"][-1].content
        
        # Get enhanced analysis including entities and SQL command type
        result = get_table_name_and_alter(client, nl_text)
        
        state["table_name"] = result.get("table_name", "UNCERTAIN")
        state["is_alter"] = result.get("is_alter", False)
        state["alter_command"] = result.get("alter_command", "")
        state["sql_command_type"] = result.get("sql_command_type", "SELECT")
        state["extracted_entities"] = result.get("extracted_entities", [])
        
        # If table name is uncertain, we need to search for candidates
        if state["table_name"] == "UNCERTAIN":
            state["needs_clarification"] = True
            state["needs_table_confirmation"] = False
        else:
            # Even if we have a table name, we should verify it exists
            state["needs_clarification"] = False
            state["needs_table_confirmation"] = True
            
        logger.info(f"Thought node: table_name={state['table_name']}, sql_command={state['sql_command_type']}, entities={state['extracted_entities']}")
    except Exception as e:
        logger.error(f"Error in thought_node: {e}")
        state["needs_clarification"] = True
        state["messages"].append(SystemMessage(content=f"Error processing query: {e}"))
    return state

def semantic_search_node(state: AgentState) -> AgentState:
    """Search for relevant tables based on entities and content when table name is uncertain."""
    logger.info("Entering semantic_search_node")
    
    if state["needs_clarification"] and state["extracted_entities"]:
        try:
            nl_text = state["messages"][-1].content
            # Search for tables that might contain relevant columns/content
            candidate_tables = search_relevant_tables_by_content(
                entities=state["extracted_entities"],
                nl_text=nl_text,
                config_path="config/settings.yaml"
            )
            
            state["candidate_tables"] = candidate_tables
            if candidate_tables:
                state["needs_clarification"] = False
                state["needs_table_confirmation"] = True
                logger.info(f"Found {len(candidate_tables)} candidate tables: {[t['table_name'] for t in candidate_tables]}")
            else:
                # Still need human clarification if no candidates found
                logger.warning("No candidate tables found through semantic search")
        except Exception as e:
            logger.error(f"Error in semantic_search_node: {e}")
            state["messages"].append(SystemMessage(content=f"Error searching for relevant tables: {e}"))
    
    return state

def human_table_confirmation_node(state: AgentState) -> AgentState:
    """Enhanced human confirmation with candidate table suggestions."""
    logger.info("Entering human_table_confirmation_node")
    
    if state["needs_table_confirmation"]:
        nl_text = state["messages"][-1].content
        
        if state["candidate_tables"]:
            # Present candidate tables for confirmation
            confirmed_table = human_table_confirmation(
                nl_text=nl_text,
                candidate_tables=state["candidate_tables"],
                extracted_entities=state["extracted_entities"]
            )
        elif state["table_name"] != "UNCERTAIN":
            # Confirm the detected table
            confirmed_table = human_table_confirmation(
                nl_text=nl_text,
                candidate_tables=[{"table_name": state["table_name"], "confidence_score": 1.0, "reason": "Directly mentioned"}],
                extracted_entities=state["extracted_entities"]
            )
        else:
            # Fallback to traditional clarification
            confirmed_table = human_clarification(f"Could not identify table in query: {nl_text}")
        
        if confirmed_table:
            state["table_name"] = confirmed_table
            state["needs_table_confirmation"] = False
            state["needs_clarification"] = False
            logger.info(f"Table confirmed by human: {confirmed_table}")
        else:
            state["needs_clarification"] = True
    
    return state

def human_in_loop_node(state: AgentState) -> AgentState:
    """Fallback clarification if needed."""
    logger.info("Entering fallback human_in_loop_node")
    if state["needs_clarification"]:
        nl_text = state["messages"][-1].content
        prompt = f"Could not identify table in query: {nl_text}\nExtracted entities: {state['extracted_entities']}"
        table_name = human_clarification(prompt)
        state["table_name"] = table_name
        state["needs_clarification"] = False
        logger.info(f"Human clarification: table_name={table_name}")
    return state

def alter_node(state: AgentState) -> AgentState:
    """Execute ALTER command and update vector store if needed."""
    logger.info(f"Entering alter_node for command type: {state['sql_command_type']}")
    if state["is_alter"] and state["alter_command"]:
        try:
            config = load_config("config/settings.yaml")
            result = update_db_and_vector_store(
                config_path="config/settings.yaml",
                db_path=config['db_path'],
                alter_command=state["alter_command"],
                table_name=state["table_name"]
            )
            state["messages"].append(SystemMessage(content=result))
            logger.info(f"Alter node: {result}")
        except Exception as e:
            logger.error(f"Error in alter_node: {e}")
            state["messages"].append(SystemMessage(content=f"Failed to alter table: {e}"))
    return state

def action_node(state: AgentState) -> AgentState:
    """Retrieve schema from vector store."""
    logger.info(f"Entering action_node for table: {state['table_name']}")
    try:
        config = load_config("config/settings.yaml")
        schema = search_table_schema(state["table_name"], "config/settings.yaml")
        if schema:
            state["schema"] = schema
            # Add command type information to the schema response
            command_info = f"\n\nDetected SQL Command Type: {state['sql_command_type']}"
            if state['extracted_entities']:
                command_info += f"\nRelevant Entities: {', '.join(state['extracted_entities'])}"
            state["schema"] += command_info
            logger.info(f"Action node: Retrieved schema for {state['table_name']}")
        else:
            state["needs_clarification"] = True
            state["messages"].append(SystemMessage(content=f"No schema found for table {state['table_name']}"))
            logger.warning(f"No schema found for {state['table_name']}")
    except Exception as e:
        logger.error(f"Error in action_node: {e}")
        state["needs_clarification"] = True
        state["messages"].append(SystemMessage(content=f"Error retrieving schema: {e}"))
    return state

def should_continue(state: AgentState):
    """Enhanced decision logic for the workflow."""
    logger.info(f"should_continue: needs_clarification={state['needs_clarification']}, needs_table_confirmation={state['needs_table_confirmation']}, is_alter={state['is_alter']}, schema={bool(state.get('schema', ''))}")
    
    if state["needs_clarification"]:
        if state["extracted_entities"]:  # Updated: Always try search if entities exist (removed 'and not state.get("candidate_tables")')
            return "semantic_search"
        return "human_in_loop"
    
    if state["needs_table_confirmation"]:
        return "human_table_confirmation"
    
    if state["is_alter"]:
        return "alter"
    
    if state.get("schema"):
        return END
    
    return "action"

def build_graph():
    """Build the enhanced LangGraph workflow."""
    graph = StateGraph(AgentState)
    
    graph.add_node("thought", thought_node)
    graph.add_node("semantic_search", semantic_search_node)
    graph.add_node("human_table_confirmation", human_table_confirmation_node)
    graph.add_node("human_in_loop", human_in_loop_node)
    graph.add_node("alter", alter_node)
    graph.add_node("action", action_node)
    
    graph.set_entry_point("thought")
    
    graph.add_conditional_edges("thought", should_continue)
    graph.add_conditional_edges("semantic_search", should_continue)
    graph.add_edge("human_table_confirmation", "action")
    graph.add_edge("human_in_loop", "action")
    graph.add_edge("alter", "action")
    graph.add_conditional_edges("action", should_continue)
    
    return graph.compile()

def run_agent(nl_text):
    """Run the enhanced agent with the given NL text."""
    try:
        logger.info(f"Running enhanced agent for query: {nl_text}")
        graph = build_graph()
        initial_state = {
            "messages": [HumanMessage(content=nl_text)],
            "table_name": "",
            "candidate_tables": [],
            "schema": "",
            "needs_clarification": False,
            "needs_table_confirmation": False,
            "is_alter": False,
            "alter_command": "",
            "sql_command_type": "SELECT",
            "extracted_entities": []
        }
        final_state = graph.invoke(initial_state)
        logger.info(f"Agent completed: schema={final_state.get('schema', '')}")
        return final_state.get("schema", "")
    except Exception as e:
        logger.error(f"Error running agent: {e}")
        return f"Error: {e}"

if __name__ == "__main__":
    nl_text = "What is the card Number of Customer ID 20002"
    schema = run_agent(nl_text)
    print(f"Retrieved Schema and Analysis:\n{schema}")