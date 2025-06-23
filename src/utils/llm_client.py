import google.generativeai as genai
import yaml
import json
import logging
import time
import pickle
import os
from google.api_core import retry
from google.api_core.exceptions import ResourceExhausted, DeadlineExceeded, ServiceUnavailable
from google.api_core.timeout import ConstantTimeout

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_config(config_path):
    try:
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    except Exception as e:
        logger.error(f"Failed to load config: {e}")
        raise

def init_gemini_client(config_path):
    config = load_config(config_path)
    try:
        genai.configure(api_key=config['gemini']['api_key'])
        model = genai.GenerativeModel('gemini-1.5-flash-latest')
        logger.info("Initialized Gemini client with model gemini-1.5-flash-latest")
        return model
    except Exception as e:
        logger.error(f"Failed to initialize Gemini client: {e}")
        raise

@retry.Retry(
    predicate=retry.if_exception_type(ResourceExhausted, DeadlineExceeded, ServiceUnavailable),
    initial=60,
    maximum=600,
    multiplier=2,
    deadline=300  # Total retry period: 5 minutes
)
def get_table_name_and_alter(client, nl_text):
    """Extract table name and detect ALTER command from NL text in JSON format."""
    cache_file = f"cache/{nl_text.replace(' ', '_')}.pkl"
    if os.path.exists(cache_file):
        with open(cache_file, 'rb') as f:
            logger.info(f"Loaded cached response for: {nl_text}")
            return pickle.load(f)

    logger.info(f"Calling Gemini API for query: {nl_text}")
    prompt = f"""
    Given the natural language query: "{nl_text}"
    1. Identify the database table name referenced in the query.Note that a reference query may have one or table name specified.
    2. Determine if the query involves an ALTER TABLE command (e.g., adding, modifying, or dropping a column).
    Return a JSON object with:
    - 'table_name': the table name as a string, or 'UNCERTAIN' if unclear.
    - 'is_alter': boolean indicating if an ALTER command is present.
    - 'alter_command': the SQL ALTER TABLE command if applicable, otherwise empty string.
    Example:
    -For "How many employees are present in department DataScience, economics and logistics":
        {{"table_name": "DataScience, economics, logistics","is_alter": false, "alter_command": "" }}
    - For "How many employees are present where salary is above 50 lkh":
      {{"table_name": "Employee", "is_alter": false, "alter_command": ""}}
    - For "give me the schema of the table ShipMethod":
      {{"table_name": "ShipMethod", "is_alter": false, "alter_command": ""}}
    - For "Alter the Employee table to add a Bonus column":
      {{"table_name": "Employee", "is_alter": true, "alter_command": "ALTER TABLE Employee ADD Bonus DECIMAL"}}
    """
    try:
        # with ConstantTimeout(timeout=30):  # 30-second timeout
        response = client.generate_content(prompt)
        logger.info(f"Received Gemini response: {response.text[:100]}...")
        # Clean response
        response_text = response.text.strip()
        if response_text.startswith('```json') and response_text.endswith('```'):
            response_text = response_text[7:-3].strip()
        result = json.loads(response_text)
        if not all(key in result for key in ['table_name', 'is_alter', 'alter_command']):
            logger.warning(f"Invalid JSON response: {response.text}")
            return {"table_name": "UNCERTAIN", "is_alter": False, "alter_command": ""}
        logger.info(f"Extracted table info: {result}")
        # Cache result
        os.makedirs("cache", exist_ok=True)
        with open(cache_file, 'wb') as f:
            pickle.dump(result, f)
        return result
    except json.JSONDecodeError as e:
        logger.warning(f"Failed to parse Gemini response: {response.text}, Error: {e}")
        return {"table_name": "UNCERTAIN", "is_alter": False, "alter_command": ""}
    except Exception as e:
        logger.error(f"Error calling Gemini API: {e}")
        raise
