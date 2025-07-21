import chromadb
from chromadb.utils import embedding_functions
from chromadb.config import Settings, DEFAULT_TENANT, DEFAULT_DATABASE
import yaml
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def search_table_schema(table_name, config_path):
    """Search for table schema in Chroma vector store for one or more table names."""
    logger.info(f"Searching schema for table(s): {table_name}")
    config = yaml.safe_load(open(config_path))
    
    # Initialize PersistentClient with persist_dir
    chroma_client = chromadb.PersistentClient(
        path=config['persist_dir'],
        settings=Settings(is_persistent=True),
        tenant=DEFAULT_TENANT,
        database=DEFAULT_DATABASE
    )
    
    try:
        # Debug: List all collections
        collections = chroma_client.list_collections()
        logger.info(f"Available collections: {[col.name for col in collections]}")
        
        collection = chroma_client.get_collection(
            name=config['collection_name'],
            embedding_function=embedding_functions.SentenceTransformerEmbeddingFunction(
                model_name=config['embedding_model']
            )
        )
        
        # Split table_name string into individual table names
        table_names = [name.strip() for name in table_name.split(',')]
        logger.info(f"Processing table names: {table_names}")
        
        # Query schemas for each table
        schemas = []
        for table in table_names:
            logger.info(f"Querying schema for table: {table}")
            results = collection.query(
                query_texts=[f"Table: {table}"],
                n_results=1,
                where={"table_name": table}
            )
            logger.info(f"Query results for {table}: {results['documents']}")
            if results['documents'] and results['documents'][0]:
                schemas.append(results['documents'][0][0])
            else:
                logger.warning(f"No schema found for table: {table}")
                schemas.append(f"No schema found for table: {table}")
        
        # Combine schemas into a single string
        combined_schema = "\n\n".join(schemas)
        return combined_schema if schemas else None
        
    except Exception as e:
        logger.error(f"Error querying collection: {e}")
        raise