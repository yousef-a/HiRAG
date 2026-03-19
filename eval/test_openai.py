import os
import json
import argparse
from tqdm import tqdm
import sys
import yaml
   
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from hirag import HiRAG, QueryParam


with open('../config.yaml', 'r') as file:
    config = yaml.safe_load(file)
    
#os.environ["OPENAI_API_KEY"] = "***"

OPENAI_MODEL = config['openai']['model']
OPENAI_API_KEY = config['openai']['api_key']
OPENAI_URL = config['openai']['base_url']

os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", "--maxqueries", type=int, default="130")
    parser.add_argument("-d", "--dataset", type=str, default="mix")
    parser.add_argument("-m", "--mode", type=str, default="hi", help="hi / naive / hi_global / hi_local / hi_bridge / hi_nobridge / hi_rerank / hi_causal")
    args = parser.parse_args()
    
    if args.mode == "naive":
        mode = True
    elif args.mode == "global" or "local":
        mode = False
        
    MAX_QUERIES = args.maxqueries
    DATASET = args.dataset
    
    input_path = f"./datasets/{DATASET}/{DATASET}_query_answers.jsonl"
    output_path1 = f"./datasets/{DATASET}/Causal04/{DATASET}_hi_result.jsonl"
    output_path2 = f"./datasets/{DATASET}/Causalrerank02/{DATASET}_causalrerank_result.jsonl"
# ✅ Check if output folder exists 
    output_folder = os.path.dirname(output_path2)
    # same folder for both outputs 
    if not os.path.exists(output_folder): 
        print(f"Error: Output folder '{output_folder}' does not exist.") 
        sys.exit(1) # stop the process immediately
    
    graph_func = HiRAG(
        working_dir=f"./datasets/{DATASET}/{DATASET}_work_dir_causal_2",
        enable_hierachical_mode=True, 
        embedding_func_max_async=8,
        enable_naive_rag=mode)

    query_list = []
    with open(input_path, encoding="utf-8", mode="r") as f:      # get context
        lines = f.readlines()
        for item in lines:
            item_dict = json.loads(item)
            query_list.append(item_dict["query"])
    query_list = query_list[:MAX_QUERIES]
    answer1_list = []
    answer2_list = []
    print(f"Perform {args.mode} search:")
    for query in tqdm(query_list):
        tqdm.write(f"Q: {query}")
        answer1 = graph_func.query(query=query, param=QueryParam(mode="hi", top_k=20))
        answer2 = graph_func.query(query=query, param=QueryParam(mode="hi_rerank", top_k=20))
        tqdm.write(f"A: {answer1} \n ################################################################################################")
        answer1_list.append(answer1)
        tqdm.write(f"A: {answer2} \n ################################################################################################")
        answer2_list.append(answer2)

    
    result_to_write = []
    for query, answer1 in zip(query_list, answer1_list):
        result_to_write.append({"query": query, "answer": answer1})
    with open(output_path1, "w") as f:
        for item in result_to_write:
            f.write(json.dumps(item) + "\n")
        
    result2_to_write = []
    for query, answer2 in zip(query_list, answer2_list):
        result2_to_write.append({"query": query, "answer": answer2})
    with open(output_path2, "w") as f:
        for item in result2_to_write:
            f.write(json.dumps(item) + "\n")
