import google.generativeai as genai
import yaml
import json
import logging
import time
import pickle
import os
import re
import requests
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

def find_relevant_tables_from_entities(client, nl_text, entities):
    """Enhanced function to analyze entities and suggest relevant table names."""
    cache_file = f"cache/entities_{hash(nl_text + str(entities))}.pkl"
    if os.path.exists(cache_file):
        with open(cache_file, 'rb') as f:
            logger.info(f"Loaded cached entity analysis for: {nl_text}")
            return pickle.load(f)

    logger.info(f"Analyzing entities for table suggestions: {entities}")
    prompt = f"""
    Database Context: Custom SQLite database for student exam results with 9 tables grouped by batch (1_batch, 2_batch, 3_batch) and branch (AIML, CSD, AIDS). Table names: 1_batch_AIML_Results, 1_batch_CSD_Results, 1_batch_AIDS_Results, 2_batch_AIML_Results, 2_batch_CSD_Results, 2_batch_AIDS_Results, 3_batch_AIML_Results, 3_batch_CSD_Results, 3_batch_AIDS_Results.
    - Common columns: 'regno' (TEXT, primary key), 'name' (TEXT), 'semester' (TEXT), 'avg gpa' (REAL).
    - Exam columns: GPAs as REAL (e.g., 'DECEMBER - 2023 gpa'). Suggest tables based on batch/branch mentions or exam periods.
    - Supplementary exams: Infer tables for recovery GPAs (e.g., 'NOVEMBER 2023 gpa' suggests 1st/2nd batch tables).
    - Gold medal: Related to 'avg gpa' > 9.00.

    Given the natural language query: "{nl_text}"
    And extracted entities: {entities}
    
    Analyze what database tables might be relevant based on these entities and the query context.
    Consider batch, branch, exam periods, and common naming patterns.
    
    Return a JSON object with:
    - 'suggested_tables': list of likely table names (e.g., '2_batch_AIML_Results')
    - 'confidence': confidence score (0-1) for each suggestion
    - 'reasoning': why each table might be relevant
    
    Example:
    For query "Students with high GPA in 2nd batch AIML" with entities ["high GPA", "2nd batch", "AIML"]:
    {{
        "suggested_tables": [
            {{"name": "2_batch_AIML_Results", "confidence": 0.95, "reasoning": "Matches batch and branch, contains 'avg gpa'"}},
            {{"name": "2_batch_CSD_Results", "confidence": 0.4, "reasoning": "Same batch but different branch"}}
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
    Database Context: This is a custom SQLite database for student exam results with 9 tables grouped by batch (1_batch, 2_batch, 3_batch) and branch (AIML, CSD, AIDS). Table names: 1_batch_AIML_Results, 1_batch_CSD_Results, 1_batch_AIDS_Results, 2_batch_AIML_Results, 2_batch_CSD_Results, 2_batch_AIDS_Results, 3_batch_AIML_Results, 3_batch_CSD_Results, 3_batch_AIDS_Results.
    - Common columns across tables: 'regno' (TEXT, primary key), 'name' (TEXT), 'semester' (TEXT), 'avg gpa' (REAL).
    - Other columns are GPA for specific exam periods (e.g., 'DECEMBER - 2023 gpa' as REAL). Column names may have spaces and use double quotes in SQL.
    - Supplementary exams: 'NOVEMBER 2022 gpa'/'NOVEMBER 2023 gpa' for 1st batch, 'NOVEMBER 2023 gpa'/'AUGUST 2024 gpa' for 2nd batch, 'AUGUST 2024 gpa' for 3rd batch. Missing regular GPA implies failure; supplementary GPA shows recovery.
    - Gold medal: 'avg gpa' > 9.00.
    - Queries may imply multiple tables (e.g., compare batches) or filters on GPA, supplementary status, or gold medals.

    Given the natural language query: "{nl_text}"
    
    Perform comprehensive analysis using the database context:
    1. Identify the database table name(s) referenced (exact match preferred, or infer from batch/branch/exam mentions; list multiple as comma-separated if query implies joins; use 'UNCERTAIN' if unclear).
    2. Extract key entities/attributes mentioned (column names like 'avg gpa', values like '9.00', exam periods, batches, branches).
    3. Determine the SQL command type (SELECT, INSERT, UPDATE, DELETE, ALTER, CREATE, DROP, etc.), considering filters like GPA > 9.00 for gold medals or supplementary logic.
    4. Check if it's an ALTER/modification command.
    5. Generate the SQL command if it's an ALTER operation (SQLite-compatible, e.g., ALTER TABLE "table_name" ADD COLUMN ...).

    Return a JSON object with:
    - 'table_name': exact table name(s) as string (comma-separated if multiple) or 'UNCERTAIN'
    - 'extracted_entities': list of column names, attributes, or key terms mentioned (e.g., ['avg gpa', 'AUGUST 2024 gpa', 'gold medal'])
    - 'sql_command_type': the primary SQL operation type
    - 'is_alter': boolean for structural modifications
    - 'alter_command': SQL ALTER statement if applicable (SQLite-compatible)
    - 'confidence': confidence score (0-1) for table identification
    
    Examples:
    For "Get students with avg gpa > 9.00 in 2nd batch AIML for gold medal":
    {{
        "table_name": "2_batch_AIML_Results",
        "extracted_entities": ["avg gpa", "gold medal", "2nd batch", "AIML"],
        "sql_command_type": "SELECT",
        "is_alter": false,
        "alter_command": "",
        "confidence": 0.95
    }}
    
    For "Compare avg gpa between 1st batch CSD and 3rd batch AIDS":
    {{
        "table_name": "1_batch_CSD_Results,3_batch_AIDS_Results",
        "extracted_entities": ["avg gpa", "1st batch", "CSD", "3rd batch", "AIDS"],
        "sql_command_type": "SELECT",
        "is_alter": false,
        "alter_command": "",
        "confidence": 0.8
    }}
    
    For "Add a column for notes in 3_batch_AIML_Results":
    {{
        "table_name": "3_batch_AIML_Results",
        "extracted_entities": ["notes", "column"],
        "sql_command_type": "ALTER",
        "is_alter": true,
        "alter_command": "ALTER TABLE \"3_batch_AIML_Results\" ADD COLUMN notes TEXT",
        "confidence": 0.95
    }}
    
    For "Get names with missing regular GPA but passed supplementary in November 2023 for 2nd batch":
    {{
        "table_name": "2_batch_AIML_Results,2_batch_CSD_Results,2_batch_AIDS_Results",
        "extracted_entities": ["name", "NOVEMBER 2023 gpa", "supplementary", "2nd batch"],
        "sql_command_type": "SELECT",
        "is_alter": false,
        "alter_command": "",
        "confidence": 0.7
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

# New function for initializing Codestral client
def init_codestral_client(config_path):
    config = load_config(config_path)
    try:
        api_key = config['mistral']['api_key']
        if not api_key or api_key == "your_mistral_codestral_api_key_here":
            raise ValueError("Mistral API key not configured properly")
        logger.info("Initialized Codestral client")
        return api_key
    except KeyError:
        logger.error("Mistral API key not found in config")
        raise ValueError("Mistral API key not found in config. Please add 'mistral: api_key: your_key' to settings.yaml")
    except Exception as e:
        logger.error(f"Failed to initialize Codestral client: {e}")
        raise

@retry.Retry(
    predicate=retry.if_exception_type(DeadlineExceeded, ServiceUnavailable),
    initial=60,
    maximum=600,
    multiplier=2,
    deadline=300
)
def generate_sql_with_codestral(api_key, nl_text, schema, sql_command_type, extracted_entities):
    """Generate SQL query using Mistral Codestral API."""
    cache_file = f"cache/sql_{hash(nl_text + schema)}.pkl"
    if os.path.exists(cache_file):
        with open(cache_file, 'rb') as f:
            logger.info(f"Loaded cached SQL for: {nl_text}")
            return pickle.load(f)

    logger.info(f"Calling Codestral API for SQL generation: {nl_text}")
    prompt = f"""
    Database Context: Custom SQLite database for student exam results with tables by batch (1_batch, 2_batch, 3_batch) and branch (AIML, CSD, AIDS). Tables: 1_batch_AIML_Results, 1_batch_CSD_Results, 1_batch_AIDS_Results, 2_batch_AIML_Results, 2_batch_CSD_Results, 2_batch_AIDS_Results, 3_batch_AIML_Results, 3_batch_CSD_Results, 3_batch_AIDS_Results.
    - Common columns: "regno" (TEXT, primary key), "name" (TEXT), "semester" (TEXT), "avg gpa" (REAL).
    - Exam columns: GPAs as REAL (e.g., "DECEMBER - 2023 gpa"). Use double quotes for columns with spaces (e.g., "AUGUST - 2024 gpa").
    - Supplementary exams: "NOVEMBER 2022 gpa"/"NOVEMBER 2023 gpa" for 1st batch, "NOVEMBER 2023 gpa"/"AUGUST 2024 gpa" for 2nd batch, "AUGUST 2024 gpa" for 3rd batch. Missing regular GPA means failure (backlog); use IS NULL for missing and >0 for passed supplementary.
    - Gold medal: WHERE "avg gpa" > 9.00.
    - For queries about all batches/branches (e.g., "all candidates"), include ALL relevant tables with UNION or joins on "regno". Be concise: group by batch if possible.

    Given the natural language query: "{nl_text}"
    Detected command type: {sql_command_type}
    Relevant entities: {', '.join(extracted_entities)}
    Table schema:
    {schema}
    
    Generate a valid, executable SQLite3 SQL query that matches the query intent:
    - Use exact table/column names from schema (double-quote columns with spaces, e.g., "AUGUST - 2024 gpa").
    - For SELECT, include WHERE for filters (e.g., "avg gpa" > 9.00 for gold medals, IS NULL for missing regular GPA indicating backlogs).
    - Handle backlog/supplementary logic: Check IS NULL on regular exam columns to detect backlogs; if query is global, union across all batches.
    - Use UNION or joins if multiple tables (e.g., INNER JOIN on "regno" for comparisons, UNION for combining results from all batches).
    - Keep queries concise: Avoid listing every column manually if not needed; use * if appropriate, but prefer specific columns.
    - For backlogs: Identify candidates with any IS NULL in regular GPA columns (indicating failure/backlog).
    - Output ONLY the complete SQL query string (must end with ';'), no explanations, code blocks, or partial queries. Ensure it's valid SQLite3 syntax and not truncated.
    
    Examples (output only the SQL):
    For "Get gold medal eligible students in 2nd batch AIML":
    SELECT "name", "avg gpa" FROM "2_batch_AIML_Results" WHERE "avg gpa" > 9.00;
    
    For "Students who failed regular but passed supplementary in November 2023 for 2nd batch CSD":
    SELECT "name" FROM "2_batch_CSD_Results" WHERE "APRIL 2023 gpa" IS NULL AND "NOVEMBER 2023 gpa" > 0;
    
    For "Get all candidates with backlogs across all batches":
    SELECT "regno", "name" FROM "1_batch_AIML_Results" WHERE "JUNE 2022 gpa" IS NULL OR "SEPTEMBER 2022 gpa" IS NULL -- (shortened for example)
    UNION SELECT "regno", "name" FROM "1_batch_CSD_Results" WHERE ... -- continue for all tables;
    """
    try:
        response = requests.post(
            "https://api.mistral.ai/v1/chat/completions",  # Codestral endpoint
            headers={"Authorization": f"Bearer {api_key}"},
            json={
                "model": "codestral-latest",
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 1500,  # Increased to handle longer queries
                "temperature": 0.1  # Lower for more deterministic output
            },
            timeout=60  # Increased timeout for complex generations
        )
        response.raise_for_status()
        result = response.json()["choices"][0]["message"]["content"].strip()
        
        # Robust extraction: Capture everything after the first code block or assume raw SQL
        sql_match = re.search(r'``````', result, re.DOTALL | re.IGNORECASE)
        if sql_match:
            sql_query = sql_match.group(1).strip()
        else:
            # Fallback: Take the entire response, split by lines, and join until ';' is found
            lines = result.split('\n')
            sql_lines = []
            for line in lines:
                stripped = line.strip()
                if stripped:
                    sql_lines.append(stripped)
                if ';' in stripped:
                    break
            sql_query = ' '.join(sql_lines).strip()
        
        # Ensure it ends with ';'
        if not sql_query.endswith(';'):
            sql_query += ';'
        
        # Validation: Check for truncation (heuristic: count expected unions for all-batch queries)
        expected_unions = 8  # 9 tables - 1 = 8 UNIONs
        union_count = sql_query.upper().count('UNION')
        if 'UNION' in sql_query.upper() and union_count < expected_unions:
            logger.warning(f"Generated SQL may be truncated (found {union_count} UNIONs, expected ~{expected_unions})")
            sql_query += ' -- Warning: Query incomplete, add remaining tables manually'
        
        # Cache result
        os.makedirs("cache", exist_ok=True)
        with open(cache_file, 'wb') as f:
            pickle.dump(sql_query, f)
        
        return sql_query
    except Exception as e:
        logger.error(f"Error calling Codestral API: {e}")
        return "SELECT * FROM table -- Error generating SQL"