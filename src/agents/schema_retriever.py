from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage, SystemMessage
from typing import TypedDict, List
from utils.llm_client import init_gemini_client, get_table_name_and_alter
from tools.vector_search import search_table_schema
from tools.human_in_loop import human_clarification
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
    schema: str
    needs_clarification: bool
    is_alter: bool
    alter_command: str

def thought_node(state: AgentState) -> AgentState:
    """Extract table name and detect ALTER command from NL text."""
    logger.info("Entering thought_node")
    try:
        config = load_config("config/settings.yaml")
        client = init_gemini_client("config/settings.yaml")
        nl_text = state["messages"][-1].content
        result = get_table_name_and_alter(client, nl_text)
        
        state["table_name"] = result.get("table_name", "UNCERTAIN")
        state["is_alter"] = result.get("is_alter", False)
        state["alter_command"] = result.get("alter_command", "")
        state["needs_clarification"] = state["table_name"] == "UNCERTAIN"
        logger.info(f"Thought node: table_name={state['table_name']}, is_alter={state['is_alter']}")
    except Exception as e:
        logger.error(f"Error in thought_node: {e}")
        state["needs_clarification"] = True
        state["messages"].append(SystemMessage(content=f"Error processing query: {e}"))
    return state

def human_in_loop_node(state: AgentState) -> AgentState:
    """Prompt human for clarification if needed."""
    logger.info("Entering human_in_loop_node")
    if state["needs_clarification"]:
        nl_text = state["messages"][-1].content
        prompt = f"Could not identify table in query: {nl_text}"
        table_name = human_clarification(prompt)
        state["table_name"] = table_name
        state["needs_clarification"] = False
        logger.info(f"Human clarification: table_name={table_name}")
    return state

def alter_node(state: AgentState) -> AgentState:
    """Execute ALTER command and update vector store if needed."""
    logger.info("Entering alter_node")
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
    """Decide whether to continue or end."""
    logger.info(f"should_continue: needs_clarification={state['needs_clarification']}, is_alter={state['is_alter']}, schema={bool(state['schema'])}")
    if state["needs_clarification"]:
        return "human_in_loop"
    if state["is_alter"]:
        return "alter"
    if state["schema"]:
        return END
    return "action"

def build_graph():
    """Build the LangGraph workflow."""
    graph = StateGraph(AgentState)
    
    graph.add_node("thought", thought_node)
    graph.add_node("human_in_loop", human_in_loop_node)
    graph.add_node("alter", alter_node)
    graph.add_node("action", action_node)
    
    graph.set_entry_point("thought")
    graph.add_edge("thought", "human_in_loop")
    graph.add_edge("human_in_loop", "alter")
    graph.add_edge("alter", "action")
    graph.add_conditional_edges("action", should_continue)
    
    return graph.compile()

def run_agent(nl_text):
    """Run the agent with the given NL text."""
    try:
        logger.info(f"Running agent for query: {nl_text}")
        graph = build_graph()
        initial_state = {
            "messages": [HumanMessage(content=nl_text)],
            "table_name": "",
            "schema": "",
            "needs_clarification": False,
            "is_alter": False,
            "alter_command": ""
        }
        final_state = graph.invoke(initial_state)
        logger.info(f"Agent completed: schema={final_state['schema']}")
        return final_state["schema"]
    except Exception as e:
        logger.error(f"Error running agent: {e}")
        return f"Error: {e}"

if __name__ == "__main__":
    nl_text = "give me the schema of the table ShipMethod"
    schema = run_agent(nl_text)
    print(f"Retrieved Schema:\n{schema}")
