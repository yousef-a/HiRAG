import os
import json
import time
import sys
sys.path.append("../")
from hirag import HiRAG, QueryParam
os.environ["OPENAI_API_KEY"] = "***"

DATASET = "cs"
file_path = f"./datasets/{DATASET}/{DATASET}_unique_contexts.json"

graph_func = HiRAG(
    working_dir=f"./datasets/{DATASET}/{DATASET}_work_dir_updatedp",
    enable_hierachical_mode=True,
    embedding_func_max_async=8,
    enable_naive_rag=False,
)

with open(file_path, mode="r") as f:
    unique_contexts = json.load(f)
    graph_func.insert(unique_contexts)
