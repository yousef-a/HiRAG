import os
import json
from collections import defaultdict
from IPython.core.debugger import set_trace
import re
from collections import Counter
from sentence_transformers import SentenceTransformer, util
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer, util
from openai import OpenAI
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.translate.meteor_score import meteor_score
from nltk.tokenize import word_tokenize
from rouge_score import rouge_scorer
import nltk
import argparse

def normalize_text(text: str) -> str:
    """Lowercase, remove punctuation, normalize whitespace."""
    text = re.sub(r'[^\w\s]', '', text.strip().lower())
    return ' '.join(text.split())

def ensure_string(x):
    """Ensure candidate answers are strings."""
    return str(x)

def unwrap_gold_answer(gold_answer):
    """Gold answers are always lists with one element → unwrap to string."""
    if isinstance(gold_answer, list) and len(gold_answer) == 1:
        return gold_answer[0]
    return str(gold_answer)

def compute_em(pred: str, true: str) -> float:
    """Exact Match score."""
    pred = normalize_text(ensure_string(pred))
    true = normalize_text(ensure_string(true))
    return 1.0 if pred == true else 0.0

from rank_bm25 import BM25Okapi

def compute_bm25(pred: str, true: str) -> float:
    """Compute BM25 similarity score between candidate and gold answer."""
    pred_tokens = normalize_text(pred).split()
    true_tokens = normalize_text(true).split()
    bm25 = BM25Okapi([true_tokens])
    score = bm25.get_scores(pred_tokens)[0]
    # Normalize BM25 score to [0,1] for consistency
    return score / (score + 1)

def compute_f1(pred: str, true: str) -> float:
    """F1 score based on token overlap."""
    pred_tokens = normalize_text(ensure_string(pred)).split()
    true_tokens = normalize_text(ensure_string(true)).split()

    if not pred_tokens or not true_tokens:
        return 0.0

    pred_counter = Counter(pred_tokens)
    true_counter = Counter(true_tokens)

    common = pred_counter & true_counter
    overlap = sum(common.values())

    precision = overlap / len(pred_tokens)
    recall = overlap / len(true_tokens)

    if precision + recall == 0:
        return 0.0
    return 2 * (precision * recall) / (precision + recall)

def evaluate_jsonl_winners(file_path):
    categories = ["Comprehensiveness", "Empowerment", "Diversity", "Query Relevance", "Overall Winner"]
    counts = {cat: defaultdict(int) for cat in categories}
    totals = defaultdict(int)

    if not os.path.exists(file_path):
        print(f"Error: File '{file_path}' not found.")
        return {}

    try:
        with open(file_path, "r", encoding="utf-8") as f:
            lines = [line for line in f if line.strip()]
    except Exception as e:
        print(f"Error opening file: {e}")
        return {}

    if not lines:
        print(f"Warning: File '{file_path}' is empty.")
        return {}

    total_answers = len(lines)
    half = total_answers // 2

    for i, line in enumerate(lines, start=1):
        data = json.loads(line)

        for cat in categories:
            if cat in data and "Winner" in data[cat]:
                winner = data[cat]["Winner"]

                if i > half:
                    actual_win = "Answer 2" if winner == "Answer 1" else "Answer 1"
                else:
                    actual_win = winner

                counts[cat][actual_win] += 1
                totals[cat] += 1

    results = {}
    for cat in categories:
        if totals[cat] > 0:
            results[cat] = {
                "Counts": dict(counts[cat]),
                "Total": totals[cat],
                "Percentages": {winner: (count / totals[cat]) * 100 for winner, count in counts[cat].items()}
            }
        else:
            results[cat] = {"Counts": {}, "Total": 0, "Percentages": {}}

    return results

def summarize_results(summary, f1, f2):
    print(f"{'Category':<20}{'Answer 1 Wins':<15}{'Answer 2 Wins':<15}{'Total':<10}{f1+' %':<15}{f2+' %':<15}")
    print("-" * 90)
    for cat, stats in summary.items():
        a1 = stats["Counts"].get("Answer 1", 0)
        a2 = stats["Counts"].get("Answer 2", 0)
        total = stats["Total"]
        p1 = stats["Percentages"].get("Answer 1", 0)
        p2 = stats["Percentages"].get("Answer 2", 0)
        print(f"{cat:<20}{a1:<15}{a2:<15}{total:<10}{p1:<15.2f}{p2:<15.2f}")

def evaluate_answers(query, gold_answer, answer1, answer2):
    """
    Evaluate two predicted answers against a gold answer using BLEU, ROUGE, METEOR.
    Returns individual scores and an aggregate score for each answer,
    plus binary flags (ans1, ans2) indicating which answer won.
    """

    # BLEU setup
    reference = [gold_answer.split()]
    pred1 = answer1.split()
    pred2 = answer2.split()
    smoothie = SmoothingFunction().method1

    bleu1 = sentence_bleu(reference, pred1, smoothing_function=smoothie)
    bleu2 = sentence_bleu(reference, pred2, smoothing_function=smoothie)

    f1_score1 = compute_f1(answer1, gold_answer)
    f1_score2 = compute_f1(answer2, gold_answer)
    
    # ROUGE setup
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    rouge1 = scorer.score(gold_answer, answer1)
    rouge2 = scorer.score(gold_answer, answer2)

    # METEOR
    meteor1 = meteor_score([word_tokenize(gold_answer)], word_tokenize(answer1))
    meteor2 = meteor_score([word_tokenize(gold_answer)], word_tokenize(answer2))

    # Aggregate score
    agg1 = (bleu1 + rouge1['rougeL'].fmeasure + meteor1) / 3
    agg2 = (bleu2 + rouge2['rougeL'].fmeasure + meteor2) / 3

    # Winner flags
    if agg1 > agg2:
        ans1, ans2 = 1, 0
    else:
        ans1, ans2 = 0, 1

    # #Side-by-side printout
    # print(f"{'Metric':<15}{'Answer1':<15}{'Answer2':<15}") 
    # print("-" * 45) 
    # print(f"{'BLEU':<15}{bleu1:<15.4f}{bleu2:<15.4f}") 
    # print(f"{'ROUGE-1 F1':<15}{rouge1['rouge1'].fmeasure:<15.4f}{rouge2['rouge1'].fmeasure:<15.4f}") 
    # print(f"{'ROUGE-2 F1':<15}{rouge1['rouge2'].fmeasure:<15.4f}{rouge2['rouge2'].fmeasure:<15.4f}") 
    # print(f"{'ROUGE-L F1':<15}{rouge1['rougeL'].fmeasure:<15.4f}{rouge2['rougeL'].fmeasure:<15.4f}") 
    # print(f"{'METEOR':<15}{meteor1:<15.4f}{meteor2:<15.4f}") 
    # print(f"{'Aggregate':<15}{agg1:<15.4f}{agg2:<15.4f}")

    return {
        "Answer1": {"BLEU": bleu1, "ROUGE": rouge1, "METEOR": meteor1, "Aggregate": agg1, "f1score": f1_score1},
        "Answer2": {"BLEU": bleu2, "ROUGE": rouge2, "METEOR": meteor2, "Aggregate": agg2, "f1score": f1_score2 }
    }, ans1, ans2


# 🔹 Batch evaluation across your three files
import json

def batch_evaluate_with_scores(hi_rerank_result_path, hi_result_path, query_answers_path):
    # Check if all files exist
    for path in [hi_rerank_result_path, hi_result_path, query_answers_path]:
        if not os.path.exists(path):
            print(f"Error: File '{path}' not found.")
            return None

    try:
        with open(hi_rerank_result_path, "r", encoding="utf-8") as f1, \
             open(hi_result_path, "r", encoding="utf-8") as f2, \
             open(query_answers_path, "r", encoding="utf-8") as fg:

            answers1 = [json.loads(line) for line in f1 if line.strip()]
            answers2 = [json.loads(line) for line in f2 if line.strip()]
            golds = [json.loads(line) for line in fg if line.strip()]

    except Exception as e:
        print(f"Error opening or reading files: {e}")
        return None

    total_ans1, total_ans2 = 0, 0

    # Accumulators for scores
    bleu1_sum, bleu2_sum = 0, 0
    rouge1_sum, rouge2_sum = 0, 0
    rouge2_sum, rouge2_sum2 = 0, 0   # rouge-2
    rougeL_sum1, rougeL_sum2 = 0, 0
    meteor1_sum, meteor2_sum = 0, 0
    agg1_sum, agg2_sum = 0, 0

    count = 0

    for a1, a2, g in zip(answers1, answers2, golds):
        query = g["query"]
        gold_answer = g["answer"][0] if isinstance(g["answer"], list) else g["answer"]
        answer1 = a1["answer"]
        answer2 = a2["answer"]

        results, ans1, ans2 = evaluate_answers(query, gold_answer, answer1, answer2)
        total_ans1 += ans1
        total_ans2 += ans2

        # Collect scores
        bleu1_sum += results["Answer1"]["BLEU"]
        bleu2_sum += results["Answer2"]["BLEU"]
        rouge1_sum += results["Answer1"]["ROUGE"]["rouge1"].fmeasure
        rouge2_sum += results["Answer2"]["ROUGE"]["rouge1"].fmeasure
        rouge2_sum2 += results["Answer2"]["ROUGE"]["rouge2"].fmeasure
        rougeL_sum1 += results["Answer1"]["ROUGE"]["rougeL"].fmeasure
        rougeL_sum2 += results["Answer2"]["ROUGE"]["rougeL"].fmeasure
        meteor1_sum += results["Answer1"]["METEOR"]
        meteor2_sum += results["Answer2"]["METEOR"]
        agg1_sum += results["Answer1"]["Aggregate"]
        agg2_sum += results["Answer2"]["Aggregate"]

        count += 1

    # Final averages
    print("\n=== Final Tally ===")
    print(f"Answer 1 Wins: {total_ans1}")
    print(f"Answer 2 Wins: {total_ans2}")
    total = total_ans1 + total_ans2
    print(f"Answer 1 %: {total_ans1/total*100:.2f}")
    print(f"Answer 2 %: {total_ans2/total*100:.2f}")

    print("\n=== Final Average Scores ===")
    print(f"{'Metric':<15}{'Answer1':<15}{'Answer2':<15}")
    print("-" * 45)
    print(f"{'BLEU':<15}{bleu1_sum/count:<15.4f}{bleu2_sum/count:<15.4f}")
    print(f"{'ROUGE-1 F1':<15}{rouge1_sum/count:<15.4f}{rouge2_sum/count:<15.4f}")
    print(f"{'ROUGE-L F1':<15}{rougeL_sum1/count:<15.4f}{rougeL_sum2/count:<15.4f}")
    print(f"{'METEOR':<15}{meteor1_sum/count:<15.4f}{meteor2_sum/count:<15.4f}")
    print(f"{'Aggregate':<15}{agg1_sum/count:<15.4f}{agg2_sum/count:<15.4f}")

def batch_evaluate_wins(hi_rerank_result_path, hi_result_path, query_answers_path, f1, f2):
    # Check if all files exist
    for path in [hi_rerank_result_path, hi_result_path, query_answers_path]:
        if not os.path.exists(path):
            print(f"Error: File '{path}' not found.")
            return None

    try:
        with open(hi_rerank_result_path, "r", encoding="utf-8") as f1, \
             open(hi_result_path, "r", encoding="utf-8") as f2, \
             open(query_answers_path, "r", encoding="utf-8") as fg:

            answers1 = [json.loads(line) for line in f1 if line.strip()]
            answers2 = [json.loads(line) for line in f2 if line.strip()]
            golds = [json.loads(line) for line in fg if line.strip()]

    except Exception as e:
        print(f"Error opening or reading files: {e}")
        return None

    metrics = ["BLEU", "ROUGE-1", "ROUGE-2", "ROUGE-L", "METEOR", "Aggregate", "f1score"]
    wins = {m: {"Answer1": 0, "Answer2": 0} for m in metrics}

    for a1, a2, g in zip(answers1, answers2, golds):
        query = g["query"]
        gold_answer = g["answer"][0] if isinstance(g["answer"], list) else g["answer"]
        answer1 = a1["answer"]
        answer2 = a2["answer"]

        results, _, _ = evaluate_answers(query, gold_answer, answer1, answer2)

        # Compare each metric
        if results["Answer1"]["BLEU"] > results["Answer2"]["BLEU"]:
            wins["BLEU"]["Answer1"] += 1
        else:
            wins["BLEU"]["Answer2"] += 1

        if results["Answer1"]["ROUGE"]["rouge1"].fmeasure > results["Answer2"]["ROUGE"]["rouge1"].fmeasure:
            wins["ROUGE-1"]["Answer1"] += 1
        else:
            wins["ROUGE-1"]["Answer2"] += 1

        if results["Answer1"]["ROUGE"]["rouge2"].fmeasure > results["Answer2"]["ROUGE"]["rouge2"].fmeasure:
            wins["ROUGE-2"]["Answer1"] += 1
        else:
            wins["ROUGE-2"]["Answer2"] += 1

        if results["Answer1"]["ROUGE"]["rougeL"].fmeasure > results["Answer2"]["ROUGE"]["rougeL"].fmeasure:
            wins["ROUGE-L"]["Answer1"] += 1
        else:
            wins["ROUGE-L"]["Answer2"] += 1

        if results["Answer1"]["METEOR"] > results["Answer2"]["METEOR"]:
            wins["METEOR"]["Answer1"] += 1
        else:
            wins["METEOR"]["Answer2"] += 1

        if results["Answer1"]["Aggregate"] > results["Answer2"]["Aggregate"]:
            wins["Aggregate"]["Answer1"] += 1
        else:
            wins["Aggregate"]["Answer2"] += 1

        if results["Answer1"]["f1score"] > results["Answer2"]["f1score"]:
            wins["f1score"]["Answer1"] += 1
        else:
            wins["f1score"]["Answer2"] += 1
            
    # Print final tally with percentages
    print("\n=== Final Win Counts per Metric ===")
    print(f"{'Metric':<12}{str(f1)+' Wins':<15}{str(f2)+' Wins':<15}{str(f1)+' %':<12}{str(f2)+' %':<12}")
    print("-" * 70)
    for m in metrics:
        a1 = wins[m]["Answer1"]
        a2 = wins[m]["Answer2"]
        total = a1 + a2
        p1 = (a1 / total * 100) if total > 0 else 0
        p2 = (a2 / total * 100) if total > 0 else 0
        print(f"{m:<12}{a1:<15}{a2:<15}{p1:<12.2f}{p2:<12.2f}")

def main():
    parser = argparse.ArgumentParser(description="Evaluate JSONL winners")

    parser.add_argument("-f", "--folder", type=str, required=True,
                        help="Folder name containing evaluation files")
    parser.add_argument("-opt", "--option", type=int, required=True,
                        choices=[1, 2, 3],
                        help="Evaluation options: 1 - rerank vs hi, 2 - causal vs hi, 3 - causalrerank vs hi")

    args = parser.parse_args()

    mapping = {
        1: ("rerank", "hi"),
        2: ("causal", "hi"),
        3: ("causalrerank", "hi"),
    }

    f1, f2 = mapping[args.option]
    print(f"{f1} {f2}")

    return args.folder, f1, f2

def run_evaluations(folder_name, f1, f2):
    for DATASET in ["mix", "agriculture", "cs"]:
        print(f"\n================================ Summary of {DATASET} Results ====================================")

        for API in ["openai", "deepseek", "glm"]:
            eval_path = f"./datasets/{DATASET}/{folder_name}/{DATASET}_{f1}_{f2}_result_{API}.jsonl"
            print(eval_path)
            summary = evaluate_jsonl_winners(eval_path)
            if summary:
                summarize_results(summary, f1, f2)

        # print("\n================ Objective Metrics Summary rate ==================")
        # hi_rerank_result_path = f"./datasets/{DATASET}/{folder_name}/{DATASET}_{f1}_result.jsonl"
        # hi_result_path = f"./datasets/{DATASET}/{folder_name}/{DATASET}_{f2}_result.jsonl"
        # query_answers_path = f"./datasets/{DATASET}/{DATASET}_query_answers.jsonl"
        # batch_evaluate_wins(hi_rerank_result_path, hi_result_path, query_answers_path, f1, f2)

if __name__ == "__main__":
    folder_name, f1, f2 = main()
    run_evaluations(folder_name, f1, f2)
