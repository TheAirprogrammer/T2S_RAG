# from tools.db_utils import update_vector_store

# if __name__ == "__main__":
#     print("Initializing vector store...")
#     update_vector_store("config/settings.yaml", "data/adventureworks_exported.db")
#     print("Vector store initialized with adventureworks_schema collection.")


import chromadb
import yaml

config = yaml.safe_load(open("config/settings.yaml"))
client = chromadb.PersistentClient(path=config['persist_dir'])
collections = client.list_collections()
print([c.name for c in collections])