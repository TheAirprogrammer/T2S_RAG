import chromadb
import yaml

config = yaml.safe_load(open("config/settings.yaml"))
client = chromadb.PersistentClient(path=config['persist_dir'])
collections = client.list_collections()
print([c.name for c in collections])