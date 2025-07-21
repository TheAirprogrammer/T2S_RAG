import google.generativeai as genai
import yaml
import json
import logging
import time
import pickle
import os
import re  
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
        if response_text.startswith('``````'):
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

# Add this function to the existing file

def find_relevant_tables_from_entities(client, nl_text, entities):
    """Enhanced function to analyze entities and suggest relevant table names."""
    cache_file = f"cache/entities_{hash(nl_text + str(entities))}.pkl"
    if os.path.exists(cache_file):
        with open(cache_file, 'rb') as f:
            logger.info(f"Loaded cached entity analysis for: {nl_text}")
            return pickle.load(f)

    logger.info(f"Analyzing entities for table suggestions: {entities}")
    prompt = f"""
    Given the natural language query: "{nl_text}"
    And extracted entities: {entities}
    
    Analyze what database tables might be relevant based on these entities and the query context.
    Consider common database naming patterns and relationships.
    
    Return a JSON object with:
    - 'suggested_tables': list of likely table names that might contain these entities
    - 'confidence': confidence score (0-1) for each suggestion
    - 'reasoning': why each table might be relevant
    
    Example:
    For query "What is the card Number of Customer ID 20002" with entities ["card Number", "Customer ID"]:
    {{
        "suggested_tables": [
            {{"name": "CreditCard", "confidence": 0.9, "reasoning": "Contains card-related information"}},
            {{"name": "CustomerCard", "confidence": 0.85, "reasoning": "Links customers to their cards"}},
            {{"name": "Payment", "confidence": 0.7, "reasoning": "Might store payment card details"}}
        ]
    }}
    """
    
    try:
        response = client.generate_content(prompt)
        logger.info(f"Received Gemini response for entity analysis: {response.text[:150]}...")
        
        response_text = response.text.strip()
        
        # Robust stripping for common formats
        if response_text.startswith('``````'):
            response_text = response_text[7:].rstrip('`').strip()  # Strip ``````
        elif response_text.startswith('``````'):
            response_text = response_text[3:].rstrip('`').strip()  # Strip leading ``````
        elif response_text.startswith('{') and response_text.endswith('}'):
            pass  # Already good
        else:
            json_match = re.search(r'\{[\s\S]*\}', response_text)
            if json_match:
                response_text = json_match.group(0)
            else:
                raise ValueError("No valid JSON found in response")
        
        result = json.loads(response_text)
        
        # Cache result
        os.makedirs("cache", exist_ok=True)
        with open(cache_file, 'wb') as f:
            pickle.dump(result, f)
        
        return result
    except json.JSONDecodeError as e:
        logger.warning(f"Failed to parse Gemini response for entity analysis: {response.text}, Error: {e}")
        return {"suggested_tables": []}
    except Exception as e:
        logger.error(f"Error in entity analysis: {e}")
        return {"suggested_tables": []}

@retry.Retry(
    predicate=retry.if_exception_type(ResourceExhausted, DeadlineExceeded, ServiceUnavailable),
    initial=60,
    maximum=600,
    multiplier=2,
    deadline=300
)
def get_table_name_and_alter(client, nl_text):
    """Enhanced function to extract table name, entities, and SQL command type."""
    cache_file = f"cache/{nl_text.replace(' ', '_').replace('?', '').replace('.', '')}.pkl"
    if os.path.exists(cache_file):
        with open(cache_file, 'rb') as f:
            logger.info(f"Loaded cached response for: {nl_text}")
            return pickle.load(f)

    logger.info(f"Calling Gemini API for enhanced query analysis: {nl_text}")
    prompt = f"""
    Given the natural language query: "{nl_text}"
    
    Perform comprehensive analysis:
    1. Identify the database table name referenced (exact match preferred, or 'UNCERTAIN' if unclear)
    2. Extract key entities/attributes mentioned (column names, values, etc.)
    3. Determine the SQL command type (SELECT, INSERT, UPDATE, DELETE, ALTER, CREATE, DROP, etc.)
    4. Check if it's an ALTER/modification command
    5. Generate the SQL command if it's an ALTER operation
    
    Return a JSON object with:
    - 'table_name': exact table name or 'UNCERTAIN'
    - 'extracted_entities': list of column names, attributes, or key terms mentioned
    - 'sql_command_type': the primary SQL operation type
    - 'is_alter': boolean for structural modifications
    - 'alter_command': SQL ALTER statement if applicable
    - 'confidence': confidence score (0-1) for table identification
    
    Examples:
    For "What is the card Number of Customer ID 20002":
    {{
        "table_name": "UNCERTAIN",
        "extracted_entities": ["card Number", "Customer ID", "20002"],
        "sql_command_type": "SELECT",
        "is_alter": false,
        "alter_command": "",
        "confidence": 0.3
    }}
    
    For "Add a new column called bonus to Employee table":
    {{
        "table_name": "Employee",
        "extracted_entities": ["bonus", "column"],
        "sql_command_type": "ALTER",
        "is_alter": true,
        "alter_command": "ALTER TABLE Employee ADD COLUMN bonus DECIMAL",
        "confidence": 0.95
    }}
    
    For "give me the schema of the table ShipMethod":
    {{
        "table_name": "ShipMethod",
        "extracted_entities": ["schema"],
        "sql_command_type": "DESCRIBE",
        "is_alter": false,
        "alter_command": "",
        "confidence": 1.0
    }}
    """
    
    try:
        response = client.generate_content(prompt)
        logger.info(f"Received enhanced Gemini response: {response.text[:150]}...")
        
        response_text = response.text.strip()
        
        # Robust stripping for common formats
        if response_text.startswith('``````'):
            response_text = response_text[7:].rstrip('`').strip()  # Strip ``````
        elif response_text.startswith('``````'):
            response_text = response_text[3:].rstrip('`').strip()  # Strip leading ``````
        elif response_text.startswith('{') and response_text.endswith('}'):
            pass  # Already good
        else:
            json_match = re.search(r'\{[\s\S]*\}', response_text)
            if json_match:
                response_text = json_match.group(0)
            else:
                raise ValueError("No valid JSON found in response")
        
        # Now parse
        result = json.loads(response_text)
        
        # Validate and add defaults if missing
        required_fields = ['table_name', 'extracted_entities', 'sql_command_type', 'is_alter', 'alter_command', 'confidence']
        for field in required_fields:
            if field not in result:
                logger.warning(f"Missing field '{field}' in response, using default")
                if field == 'table_name':
                    result[field] = "UNCERTAIN"
                elif field == 'extracted_entities':
                    result[field] = []
                elif field == 'sql_command_type':
                    result[field] = "SELECT"
                elif field == 'is_alter':
                    result[field] = False
                elif field == 'alter_command':
                    result[field] = ""
                elif field == 'confidence':
                    result[field] = 0.0
        
        logger.info(f"Enhanced extraction result: {result}")
        
        # Cache result
        os.makedirs("cache", exist_ok=True)
        with open(cache_file, 'wb') as f:
            pickle.dump(result, f)
        
        return result
        
    except json.JSONDecodeError as e:
        logger.warning(f"Failed to parse enhanced Gemini response: {response.text}, Error: {e}")
        # Enhanced fallback: Try to extract entities manually if possible
        fallback_entities = re.findall(r'\"(.*?)\"', response.text)  # Simple extraction of quoted strings as potential entities
        return {
            "table_name": "UNCERTAIN", 
            "extracted_entities": fallback_entities or [],
            "sql_command_type": "SELECT",
            "is_alter": False, 
            "alter_command": "",
            "confidence": 0.0
        }
    except Exception as e:
        logger.error(f"Error calling enhanced Gemini API: {e}")
        raise