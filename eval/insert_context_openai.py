import os
import json
import time

from hirag import HiRAG, QueryParam

import yaml
# Load configuration from YAML file
with open('config.yaml', 'r') as file:
    config = yaml.safe_load(file)

# Extract configurations
OPENAI_EMBEDDING_MODEL = config['openai']['embedding_model']
OPENAI_MODEL = config['openai']['model']
OPENAI_API_KEY = config['openai']['api_key']

os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
from datetime import datetime

EXP_MODE = "hi_causal"  # select hi or hi_causal for indexing

DATASET = "mix"
file_path = f"./eval/datasets/{DATASET}/{DATASET}_unique_contexts.json"
run_name = f"./eval/{DATASET}_{EXP_MODE}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

graph_func = HiRAG(
    working_dir=run_name, 
    enable_hierachical_mode=True, 
    embedding_func_max_async=8,
    enable_naive_rag=True,
    
    addon_params={
                        "neo4j_url": config['hirag']['neo4j_url'],
                        "neo4j_auth": config['hirag']['neo4j_auth'],
                        "causal_max_edges": config['hirag'].get("causal_max_edges", None),  # optional
                        "index_mode": EXP_MODE,  
                    }
    )

with open(file_path, mode="r", encoding="utf-8") as f:
    unique_contexts = json.load(f)
    graph_func.insert(unique_contexts)