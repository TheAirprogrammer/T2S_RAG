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
# Add these functions to the existing file

def search_relevant_tables_by_content(entities, nl_text, config_path, top_k=5):
    """Search for tables that might contain relevant columns based on entities."""
    logger.info(f"Searching for tables containing entities: {entities}")
    config = yaml.safe_load(open(config_path))
    
    chroma_client = chromadb.PersistentClient(
        path=config['persist_dir'],
        settings=Settings(is_persistent=True),
        tenant=DEFAULT_TENANT,
        database=DEFAULT_DATABASE
    )
    
    try:
        collection = chroma_client.get_collection(
            name=config['collection_name'],
            embedding_function=embedding_functions.SentenceTransformerEmbeddingFunction(
                model_name=config['embedding_model']
            )
        )
        
        candidate_tables = []
        
        # Search using different strategies
        search_queries = []
        
        # 1. Search by individual entities
        for entity in entities:
            search_queries.append(f"column {entity}")
            search_queries.append(f"field {entity}")
        
        # 2. Search by combined context
        if len(entities) > 1:
            search_queries.append(f"table with {' and '.join(entities)}")
        
        # 3. Search by natural language context
        search_queries.append(nl_text)
        
        for query in search_queries:
            try:
                results = collection.query(
                    query_texts=[query],
                    n_results=top_k,
                    include=['documents', 'metadatas', 'distances']
                )
                
                if results['documents'] and results['documents'][0]:
                    for i, doc in enumerate(results['documents'][0]):
                        metadata = results['metadatas'][0][i] if results['metadatas'] else {}
                        distance = results['distances'][0][i] if results['distances'] else 1.0
                        
                        table_name = metadata.get('table_name', 'Unknown')
                        confidence = max(0, 1 - distance)  # Convert distance to confidence
                        
                        # Check if this table already exists in candidates
                        existing = next((t for t in candidate_tables if t['table_name'] == table_name), None)
                        if existing:
                            # Update confidence if this is better
                            if confidence > existing['confidence_score']:
                                existing['confidence_score'] = confidence
                                existing['reason'] = f"Found relevant content for query: {query}"
                        else:
                            candidate_tables.append({
                                'table_name': table_name,
                                'confidence_score': confidence,
                                'reason': f"Contains content matching: {query}",
                                'schema_preview': doc[:200] + "..." if len(doc) > 200 else doc
                            })
                            
            except Exception as e:
                logger.warning(f"Error searching with query '{query}': {e}")
                continue
        
        # Sort by confidence and remove duplicates
        candidate_tables = sorted(candidate_tables, key=lambda x: x['confidence_score'], reverse=True)
        
        # Filter out very low confidence results
        candidate_tables = [t for t in candidate_tables if t['confidence_score'] > 0.2]
        
        logger.info(f"Found {len(candidate_tables)} candidate tables")
        return candidate_tables[:top_k]  # Return top results
        
    except Exception as e:
        logger.error(f"Error in semantic search: {e}")
        return []

def get_all_table_names(config_path):
    """Get all available table names from the vector store."""
    config = yaml.safe_load(open(config_path))
    
    chroma_client = chromadb.PersistentClient(
        path=config['persist_dir'],
        settings=Settings(is_persistent=True),
        tenant=DEFAULT_TENANT,
        database=DEFAULT_DATABASE
    )
    
    try:
        collection = chroma_client.get_collection(
            name=config['collection_name'],
            embedding_function=embedding_functions.SentenceTransformerEmbeddingFunction(
                model_name=config['embedding_model']
            )
        )
        
        # Get all documents to extract table names
        all_data = collection.get(include=['metadatas'])
        table_names = set()
        
        if all_data['metadatas']:
            for metadata in all_data['metadatas']:
                if 'table_name' in metadata:
                    table_names.add(metadata['table_name'])
        
        return list(table_names)
        
    except Exception as e:
        logger.error(f"Error getting table names: {e}")
        return []