import sqlite3
import yaml
from chromadb import Client
from chromadb.utils import embedding_functions

def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def get_table_schemas(db_path, table_name=None):
    """Query SQLite database for table schemas, optionally for a specific table."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = cursor.fetchall()
    
    schemas = []
    for table in tables:
        tbl_name = table[0]
        if table_name and tbl_name != table_name:
            continue
        cursor.execute(f"PRAGMA table_info({tbl_name});")
        columns = cursor.fetchall()
        schema_content = f"Table: {tbl_name}\nColumns:\n"
        for col in columns:
            col_name = col[1]
            col_type = col[2]
            schema_content += f"- {col_name} ({col_type})\n"
        schemas.append({
            'table_name': tbl_name,
            'schema': schema_content
        })
    
    conn.close()
    return schemas

def update_vector_store(config_path, db_path, table_name=None):
    """Update Chroma vector store with schemas, optionally for a specific table."""
    config = load_config(config_path)
    chroma_client = Client()
    collection = chroma_client.get_or_create_collection(
        name=config['collection_name'],
        embedding_function=embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=config['embedding_model']
        )
    )
    
    # Clear existing data for the table if specified
    if table_name:
        existing_ids = collection.get(where={'table_name': table_name})['ids']
        if existing_ids:
            collection.delete(ids=existing_ids)
    
    schemas = get_table_schemas(db_path, table_name)
    batch_size = config['batch_size']
    for i in range(0, len(schemas), batch_size):
        batch = schemas[i:i + batch_size]
        collection.add(
            documents=[schema['schema'] for schema in batch],
            metadatas=[{'table_name': schema['table_name']} for schema in batch],
            ids=[f"schema_{schema['table_name']}_{j}" for j, schema in enumerate(batch)]
        )
    print(f"Updated vector store for {len(schemas)} schemas.")

def execute_alter_command(db_path, alter_command):
    """Execute an ALTER TABLE command on the database."""
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute(alter_command)
        conn.commit()
        conn.close()
        return True
    except sqlite3.Error as e:
        print(f"Error executing ALTER command: {e}")
        return False

def update_db_and_vector_store(config_path, db_path, alter_command, table_name):
    """Execute ALTER command and update vector store for the affected table."""
    success = execute_alter_command(db_path, alter_command)
    if success:
        update_vector_store(config_path, db_path, table_name)
        return f"Successfully executed {alter_command} and updated vector store."
    return f"Failed to execute {alter_command}."