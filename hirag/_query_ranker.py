import asyncio
import numpy as np
import json
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer, CrossEncoder
from transformers import AutoTokenizer
import logging
from typing import List, Dict, Any
# -------------------------------
# Global Config
# -------------------------------
ENTITY_TYPES = ["person", "role", "technology", "organization", "event", "location", "concept"]

# Initialize models once globally
semantic_model = SentenceTransformer('all-MiniLM-L6-v2')
cross_encoder_model = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
tokenizer = AutoTokenizer.from_pretrained("cross-encoder/ms-marco-MiniLM-L-6-v2")

# -------------------------------
# Utility
# -------------------------------
def json_check(clusters):
    if isinstance(clusters, list): 
        return json.dumps(clusters)
    elif isinstance(clusters, str): 
        return clusters
    else:
        return '[]'

def json_check_clean(clusters):
    if isinstance(clusters, list) or isinstance(clusters, dict):
        # Already a Python object → dump safely
        return json.dumps(clusters, ensure_ascii=False)
    elif isinstance(clusters, str):
        try:
            # Try to parse the string as JSON
            parsed = json.loads(clusters)
            return json.dumps(parsed, ensure_ascii=False)
        except json.JSONDecodeError:
            logging.error("Invalid JSON string in clusters")
            return '[]'
    else:
        # Fallback
        return '[]'


def normalize_scores(scores):
    if scores is None or len(scores) == 0:
        return []
    min_val, max_val = min(scores), max(scores)
    if max_val == min_val:
        return [0.0 for _ in scores]
    return [(s - min_val) / (max_val - min_val) for s in scores]

def softmax(x):
    e_x = np.exp(x - np.max(x))  # stability trick
    return e_x / e_x.sum()

# -------------------------------
# Async scoring functions
# -------------------------------
async def bm25_score(query, corpus):
    tokenized_corpus = [doc.split() for doc in corpus]
    bm25 = BM25Okapi(tokenized_corpus)
    scores = bm25.get_scores(query.split())
    return normalize_scores(scores)

async def graph_scoring(nodes_data):
    max_degree = max(n.get('rank', 0) for n in nodes_data)
    return [(n.get('rank', 0) / max_degree) if max_degree > 0 else 0.0
            for n in nodes_data]

async def metaboost_score(query, nodes_data):
    # Encode query and entity types
    query_emb = semantic_model.encode([query], convert_to_numpy=True)[0]
    type_embs = semantic_model.encode(ENTITY_TYPES, convert_to_numpy=True)

    # Cosine similarity
    sims = np.dot(type_embs, query_emb) / (np.linalg.norm(type_embs, axis=1) * np.linalg.norm(query_emb))

    # Softmax normalization → probability distribution
    type_probs = softmax(sims)

    # Assign boost scores to each node based on its entity_type
    scores = []
    for n in nodes_data:
        etype = n.get('entity_type', '').strip('"').lower()
        if etype in ENTITY_TYPES:
            idx = ENTITY_TYPES.index(etype)
            scores.append(float(type_probs[idx]))
        else:
            scores.append(0.0)  # unknown type gets no boost
    return scores

#-----------------
# Bi-encoder for long passages
#-----------------
# Initialize once
async def embed_long_text_async(text, chunk_size=400, overlap=50, agg="mean"):
    """
    Async: Chunk long text, embed each chunk, and aggregate into a single vector.

    Args:
        text (str): The document text.
        chunk_size (int): Max words per chunk.
        overlap (int): Overlap between chunks.
        agg (str): Aggregation method: 'mean', 'max', or 'sum'.

    Returns:
        np.ndarray: Aggregated embedding vector for the document.
    """
    # Step 1: Chunk by words (safe for any encoder)
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        start += chunk_size - overlap

    # Step 2: Embed each chunk asynchronously
    chunk_embs = await asyncio.to_thread(
        semantic_model.encode, chunks, convert_to_numpy=True
    )

    # Step 3: Aggregate
    if agg == "mean":
        doc_emb = np.mean(chunk_embs, axis=0)
    elif agg == "max":
        doc_emb = np.max(chunk_embs, axis=0)
    elif agg == "sum":
        doc_emb = np.sum(chunk_embs, axis=0)
    else:
        raise ValueError("Unsupported aggregation method")

    return doc_emb


async def embed_documents_async(documents, chunk_size=400, overlap=50, agg="mean"):
    """
    Async embedding for multiple documents.
    """
    tasks = [embed_long_text_async(doc, chunk_size, overlap, agg) for doc in documents]
    return await asyncio.gather(*tasks)


#-----------------
# Cross encoder for long passage
#-----------------
def chunk_text(text, chunk_size=450, overlap=50):
    # logging.info(f"Start chunking")
    tokens = tokenizer.tokenize(text)
    # logging.info(f"Total doc token count: {len(tokens)}")
    chunks = []
    start = 0
    while start < len(tokens):
        end = min(start + chunk_size, len(tokens))
        chunk_tokens = tokens[start:end]
        chunk = tokenizer.convert_tokens_to_string(tokens[start:end])
        # logging.info(f"Chunk token count: {len(chunk_tokens)}")
        # logging.info(f"Length of chunk: {len(chunk)}")
        chunks.append(chunk)
        start += chunk_size - overlap

    return chunks

async def cross_encoder_rank_for_passages(query, documents):
    doc_scores = []
    cross_enc_iter = 0
    for doc in documents:
        cross_enc_iter+=1
        # Step 1: Chunk document
        chunks = chunk_text(doc)

        # Step 2: Score each query+chunk pair
        pairs = [(query, chunk) for chunk in chunks]
        scores = cross_encoder_model.predict(pairs)
        doc_score = max(scores)
        doc_scores.append(doc_score)

    # Step 4: Normalize across documents
    return normalize_scores(doc_scores)

async def cross_encoder_rank_for_text_units(query, documents, min_score_threshold=-9999, agg="mean"):
    doc_scores = []

    for doc in documents:
        chunks = chunk_text(doc)
        pairs = [(query, chunk) for chunk in chunks]
        
        scores = cross_encoder_model.predict(pairs)

        # Aggregate chunk scores into one document score
        if agg == "mean":
            doc_score = float(np.mean(scores))
        elif agg == "max":
            doc_score = float(np.max(scores))
        elif agg == "sum":
            doc_score = float(np.sum(scores))
        else:
            raise ValueError("Unsupported aggregation method")

        doc_scores.append({"score": doc_score})

    # # Decide threshold (global mean if not provided)
    # threshold = min_score_threshold if min_score_threshold is not None else float(np.mean([d["score"] for d in doc_scores]))

    # # Filter by threshold
    # results = [d for d in doc_scores if d["score"] >= threshold]

    # # Sort by score descending
    # results_sorted = sorted(results, key=lambda x: x["score"], reverse=True)

    return doc_scores

async def cross_encoder_rank(query, descriptions):
    pairs = [(query, d) for d in descriptions]
    scores = cross_encoder_model.predict(pairs)
    return normalize_scores(scores)

# -------------------------------
# Combine scores
# -------------------------------
def combine_scores(semantic_scores, bm25_scores, graph_scores, meta_scores,
                   weights=(0.34, 0.33, 0.20, 0.13)):
    combined = []
    for i in range(len(semantic_scores)):
        score = (weights[0] * semantic_scores[i] +
                 weights[1] * bm25_scores[i] +
                 weights[2] * graph_scores[i] +
                 weights[3] * meta_scores[i])
        combined.append(score)
    return combined

def combine_scores_adaptive_weights(semantic_scores, bm25_scores, graph_scores, meta_scores):

    # Stack signals into array [n_samples, n_signals]
    logging.info(
        "semantic_scores: " +
        ", ".join([f"{w:.4f}" for w in semantic_scores])
    )
    logging.info(
        "bm25_scores : " +
        ", ".join([f"{w:.4f}" for w in bm25_scores])
    ) 
    signals = np.array([semantic_scores, bm25_scores, graph_scores, meta_scores])

    # Compute variance for each signal
    variances = np.var(signals, axis=1)

    # Avoid division by zero: replace 0 variance with small epsilon
    variances = np.where(variances == 0, 1e-8, variances)

    # Inverse variance weighting
    inv_var = 1.0 / variances
    weights = inv_var / np.sum(inv_var)

#    logging.info(f"Adaptive Weights: {weights}")
    logging.info(
        "Multisignal Adaptive Weights: " +
        ", ".join([f"{w:.4f}" for w in weights])
    )
    
    # Weighted combination
    combined = []
    for i in range(len(semantic_scores)):
        score = (weights[0] * semantic_scores[i] +
                 weights[1] * bm25_scores[i] +
                 weights[2] * graph_scores[i] +
                 weights[3] * meta_scores[i])
        combined.append(score)

    return combined


def combine_scores_adaptive_weights_2signals(semantic_scores, bm25_scores):
    """
    Combine semantic and BM25 scores using adaptive weights
    based on inverse variance of each signal.
    """

    signals = [semantic_scores, bm25_scores]
    variances = np.array([np.var(s) for s in signals])

    # Mask out zero-variance signals
    mask = variances > 0
    inv_var = np.zeros_like(variances)
    inv_var[mask] = 1.0 / variances[mask]

    # Normalize weights
    if inv_var.sum() == 0:
        weights = np.ones_like(inv_var) / len(inv_var)
    else:
        weights = inv_var / inv_var.sum()
        
    logging.info(f"Adaptive Weights: {weights}")
    # Weighted combination
    combined = [
        weights[0] * semantic_scores[i] +
        weights[1] * bm25_scores[i]
        for i in range(len(semantic_scores))
    ]

    return combined

# -------------------------------
# Final integration
# -------------------------------
def final_metadata_integration(nodes_data, bm25_scores, graph_scores,
                               meta_scores, semantic_scores, ce_scores,
                               combined_scores, ce_weight=0.5):
    results = []
    for i, node in enumerate(nodes_data):
        final_score = (ce_weight * ce_scores[i]) + ((1 - ce_weight) * combined_scores[i])
        results.append({
            "entity_name": node.get('entity_name', ''),
            "description": node.get('description', ''),
            "entity_type": node.get('entity_type', ''),
            "source_id": node.get('source_id', ''),
            "distance": semantic_scores[i],
            "clusters": json_check(node.get('clusters', [])),
            "rank": node.get('rank', 0),
            "bm25_score": bm25_scores[i],
            "graph_score": graph_scores[i],
            "meta_score": meta_scores[i],
            "cross_enc_score": ce_scores[i],
            "multi_signal_score": combined_scores[i],
            "final_score": final_score
        })
    return sorted(results, key=lambda x: x['final_score'], reverse=True)

# -------------------------------
# Async pipeline runner (MAIN)
# -------------------------------
async def rerank_pipeline_entities(query, nodes_data, top_k=20, ce_weight=0.5):
    # Step 1: Prepare descriptions and semantic scores
    descriptions = [n['description'] for n in nodes_data]
    semantic_scores = [n['distance'] for n in nodes_data]  # already provided

    # Step 2: Run async tasks
    bm25_task = bm25_score(query, descriptions)
    graph_task = graph_scoring(nodes_data)
    meta_task = metaboost_score(query, nodes_data)

    bm25_scores, graph_scores, meta_scores = await asyncio.gather(
        bm25_task, graph_task, meta_task
    )

    # Step 3: Combine scores
    #combined_scores = combine_scores(semantic_scores, bm25_scores, graph_scores, meta_scores)
    combined_scores = combine_scores_adaptive_weights(semantic_scores, bm25_scores, graph_scores, meta_scores)
    
    multi_score_nodes = []
    for i, n in enumerate(nodes_data):
        enriched = dict(n)
        enriched["bm25_score"] = bm25_scores[i]
        enriched["graph_score"] = graph_scores[i]
        enriched["meta_score"] = meta_scores[i]
        enriched["multi_signal_score"] = combined_scores[i]
        multi_score_nodes.append(enriched)

    # Step 4: Sort by multi-signal score and keep top 2*top_k
    multi_score_nodes = sorted(
        multi_score_nodes,
        key=lambda x: x["multi_signal_score"],
        reverse=True
    )[:top_k * 2]

    # Step 5: Cross-encoder scores
    ce_descriptions = [n['description'] for n in multi_score_nodes]
    ce_scores = await cross_encoder_rank(query, ce_descriptions)

    # Step 6: Attach CE scores
    enriched_nodes = []
    for i, n in enumerate(multi_score_nodes):
        enriched_for_ce = dict(n)
        enriched_for_ce["ce_score"] = ce_scores[i]
        enriched_nodes.append(enriched_for_ce)

    # Step 7: Sort by CE score and keep top_k
    ce_entities = sorted(enriched_nodes, key=lambda x: x["ce_score"], reverse=True)[:top_k]
    
    # Step 8: Get adaptive weights
    multi_score = np.array([n["multi_signal_score"] for n in ce_entities], dtype=float)
    cross_score = np.array([n["ce_score"] for n in ce_entities], dtype=float)

    # Stack signals into array [n_samples, n_signals]
    signals = np.array([multi_score, cross_score])

    # Compute variance for each signal
    variances = np.var(signals, axis=1)

    # Avoid division by zero: replace 0 variance with small epsilon
    variances = np.where(variances == 0, 1e-8, variances)

    # Inverse variance weighting
    inv_var = 1.0 / variances
    weights = inv_var / np.sum(inv_var)

    # logging.info(f"Cross Encoder Adaptive Weights: {weights:.4f}")
    logging.info(
        "CommunityReport_Cross Encoder Adaptive Weights: " +
        ", ".join([f"{w:.4f}" for w in weights])
    )

    reranked_entities = []
    # Step 9: Hybrid final score
    for h in ce_entities:
        enriched_for_hybrid_score = dict(h)
        enriched_for_hybrid_score["final_score"] = (
            (weights[1] * h["ce_score"]) + (weights[0] * h["multi_signal_score"])
        )
        reranked_entities.append(enriched_for_hybrid_score)

    # Step 10: Sort by final score
    reranked_entities = sorted(reranked_entities, key=lambda x: x["final_score"], reverse=True)

    return reranked_entities

    
    # return final_metadata_integration(nodes_data, bm25_scores, graph_scores,
    #                                   meta_scores, semantic_scores, ce_scores,
    #                                   combined_scores)

# -------------------------------
# Async pipeline runner for communities
# -------------------------------
async def rerank_pipeline_communities(query, community_reports, ce_weight=0.6):
    """
    Rerank community reports based on report_string and report_json.summary.
    """
    descriptions = []
    for c in community_reports:
        # Normalize findings to always be a JSON string
        json_check_string = c["report_json"].get("findings", [])
        # logging.info(json_check_string)

        # logging.info(f"Full report json : {c['report_json']}")
        # logging.info("Completed")         
        #findings_str = json_check(c["report_json"].get("findings", []))
        findings_str = json_check_clean(json_check_string)
        
        findings = json.loads(findings_str)
    
        findings_text = " ".join([f["summary"] + " " + f["explanation"] for f in findings])
        description = (
            f"{c['report_string']} "
            f"{c['report_json'].get('summary', '')} "
            f"{c['report_json'].get('rating_explanation', '')} "
            f"{findings_text}"
        )
        descriptions.append(description)


    # Semantic scores (embedding similarity with query)
    query_emb = semantic_model.encode([query], convert_to_numpy=True)[0]
    desc_embs = await embed_documents_async(descriptions)
    
    semantic_scores = [
        float(np.dot(query_emb, d) / (np.linalg.norm(query_emb) * np.linalg.norm(d)))
        for d in desc_embs
    ]
    semantic_scores = normalize_scores(semantic_scores)

    # BM25 scores
    bm25_scores = await bm25_score(query, descriptions)

    # Combine scores (graph/meta placeholders set to zero)
    combined_scores = combine_scores(
        semantic_scores,
        bm25_scores,
        [0] * len(community_reports),
        [0] * len(community_reports),
    )

    for i, c in enumerate(community_reports):
        c['combined_score'] = combined_scores[i]

    # Cross encoder scores
    ce_results = await cross_encoder_rank_for_text_units(query, descriptions)

    # Extract just the scores 
    raw_scores = [item["score"] for item in ce_results] 
    
    # Normalize them 
    ce_scores = normalize_scores(raw_scores) 
    
    # Assign back to each community 
    for i, c in enumerate(community_reports): 
        c['ce_score'] = ce_scores[i]

    # Final weighted score
    for i, c in enumerate(community_reports):
        c['final_score'] = (ce_weight * c['ce_score'] +
                            (1 - ce_weight) * c['combined_score'])

    # After computing final_score for each community
    ranked_reports = sorted(
        community_reports,
        key=lambda c: c['final_score'],
        reverse=True  # highest score first
    )
    
    return ranked_reports


# -------------------------------
# Final integration for communities
# -------------------------------
def final_community_integration(community_reports, bm25_scores,
                                semantic_scores, ce_scores,
                                combined_scores, ce_weight=0.5):
    results = []
    for i, c in enumerate(community_reports):
        # Compute final score
        final_score = (ce_weight * ce_scores[i]) + ((1 - ce_weight) * combined_scores[i])

        # Copy original community dict so we don’t lose fields
        enriched = dict(c)  # shallow copy of all original keys

        # Add scoring signals
        enriched.update({
            "semantic_score": semantic_scores[i],
            "bm25_score": bm25_scores[i],
            "cross_enc_score": ce_scores[i],
            "multi_signal_score": combined_scores[i],
            "final_score": final_score
        })

        results.append(enriched)

    # Sort by final_score
    return sorted(results, key=lambda x: x["final_score"], reverse=True)
    
# -------------------------------
# Async pipeline runner for edges
# -------------------------------
async def rerank_pipeline_edges(query, edges, ce_weight=0.5):
    """
    Rerank edges based on description and src_tgt nodes.
    
    Args:
        query (str): The query string.
        edges (list[dict]): List of edge dicts with keys:
            - src_tgt (list[str])
            - description (str)
            - other metadata
        ce_weight (float): Weight for cross-encoder score in final ranking.
    
    Returns:
        list[dict]: Sorted edges with scores attached.
    """
    # Build descriptions by combining src_tgt + description
    descriptions = [
        f"{' '.join(e.get('src_tgt', []))} {e.get('description','')}"
        for e in edges
    ]

    # Semantic scores
    query_emb = semantic_model.encode([query], convert_to_numpy=True)[0]
    desc_embs = semantic_model.encode(descriptions, convert_to_numpy=True)
    semantic_scores = [
        float(np.dot(query_emb, d) / (np.linalg.norm(query_emb) * np.linalg.norm(d)))
        for d in desc_embs
    ]
    semantic_scores = normalize_scores(semantic_scores)

    # BM25 lexical scores
    bm25_scores = await bm25_score(query, descriptions)

    # Cross-encoder scores
    ce_scores = await cross_encoder_rank(query, descriptions)

    # Combine signals (semantic + bm25)
    combined_scores = combine_scores(
        semantic_scores, bm25_scores,
        [0]*len(edges),  # no graph score
        [0]*len(edges)   # no meta score
    )

    # Final integration
    results = []
    for i, e in enumerate(edges):
        final_score = (ce_weight * ce_scores[i]) + ((1 - ce_weight) * combined_scores[i])
        enriched = dict(e)  # keep all original fields

        # Cast all scores to native Python floats
        enriched.update({
            "semantic_score": float(semantic_scores[i]),
            "bm25_score": float(bm25_scores[i]),
            "cross_enc_score": float(ce_scores[i]),
            "multi_signal_score": float(combined_scores[i]),
            "query_rank": float(final_score)
        })
        results.append(enriched)

    # Sort by query_rank
    return sorted(results, key=lambda x: x["query_rank"], reverse=True)


# -------------------------------
# Async pipeline runner for text units
# -------------------------------
# async def rerank_pipeline_text_units(query, text_units, ce_weight=0.5):
#     """
#     Rerank text units (passages) based on their content.
    
#     Args:
#         query (str): The query string.
#         text_units (list[dict]): List of text unit dicts with keys:
#             - content (str)
#             - tokens, chunk_order_index, full_doc_id, etc.
#         ce_weight (float): Weight for cross-encoder score in final ranking.
    
#     Returns:
#         list[dict]: Sorted text units with scores attached.
#     """
#     # Use content as description
#     descriptions = [tu.get("content", "") for tu in text_units]

#     # Semantic scores
#     query_emb = semantic_model.encode([query], convert_to_numpy=True)[0]
#     #desc_embs = semantic_model.encode(descriptions, convert_to_numpy=True)
#     desc_embs = await embed_documents_async(descriptions)
    
#     semantic_scores = [
#         float(np.dot(query_emb, d) / (np.linalg.norm(query_emb) * np.linalg.norm(d)))
#         for d in desc_embs
#     ]
#     semantic_scores = normalize_scores(semantic_scores)

#     # BM25 lexical scores
#     bm25_scores = await bm25_score(query, descriptions)

#     combined_scores = combine_scores( 
#         semantic_scores, 
#         bm25_scores, 
#         [0]*len(text_units), # no graph score 
#         [0]*len(text_units) # no meta score 
#     )

#     top_indices = [i for i, score in enumerate(combined_scores) if score > 0.75
    
#     top_descriptions = [descriptions[i] for i in top_indices]
#     # Cross-encoder scores
#     ce_results = await cross_encoder_rank_for_text_units(query, top_descriptions)
#     ce_scores = [0.0] * len(text_units)
#     for idx, res in zip(top_indices, ce_results):
#         ce_scores[idx] = res["score"]

#     # Final integration
#     results = []
#     for i, tu in enumerate(text_units):
#         final_score = (ce_weight * ce_scores[i]) + ((1 - ce_weight) * combined_scores[i])
#         enriched = dict(tu)  # keep all original fields

#         # Cast scores to native Python floats
#         enriched.update({
#             "semantic_score": float(semantic_scores[i]),
#             "bm25_score": float(bm25_scores[i]),
#             "cross_enc_score": float(ce_scores[i]),
#             "multi_signal_score": float(combined_scores[i]),
#             "query_rank": float(final_score)
#         })
#         results.append(enriched)

#     # Sort by query_rank
#     return sorted(results, key=lambda x: x["query_rank"], reverse=True)


# -------------------------------
# Async pipeline runner for text units
# -------------------------------
async def rerank_text_units_filtering(query, text_units, ce_weight=0.5):
    """
    Rerank text units (passages) based on their content.
    
    Args:
        query (str): The query string.
        text_units (list[dict]): List of text unit dicts with keys:
            - content (str)
            - tokens, chunk_order_index, full_doc_id, etc.
        ce_weight (float): Weight for cross-encoder score in final ranking.
    
    Returns:
        list[dict]: Sorted text units with scores attached.
    """
    # Use content as description
    descriptions = [tu.get("content", "") for tu in text_units]

    # Semantic scores
    query_emb = semantic_model.encode([query], convert_to_numpy=True)[0]
    desc_embs = await embed_documents_async(descriptions)

    semantic_scores = [
        float(np.dot(query_emb, d) / (np.linalg.norm(query_emb) * np.linalg.norm(d)))
        for d in desc_embs
    ]
    semantic_scores = normalize_scores(semantic_scores)

    # BM25 lexical scores
    bm25_scores = await bm25_score(query, descriptions)

    combined_scores = combine_scores( 
        semantic_scores, 
        bm25_scores, 
        [0]*len(text_units), # no graph score 
        [0]*len(text_units) # no meta score 
    )
    
    top_indices = [i for i, score in enumerate(combined_scores) if score > 0.25]
    
    top_descriptions = [descriptions[i] for i in top_indices]
    
    # Cross-encoder scores
    ce_results = await cross_encoder_rank_for_text_units(query, top_descriptions)
    
    raw_ce_scores = [item["score"] for item in ce_results]
    ce_scores = normalize_scores(raw_ce_scores)
    
    # Attach scores back to text_units
    for i, d in enumerate(ce_results):
        d["final_score"] = ce_scores[i]
        
    # Sort by final_score descending
    text_units = sorted(ce_results, key=lambda x: x["final_score"], reverse=True)
    #logging.info(f"{text_units}")
    return text_units

async def embed_long_text_async_comm(
    query_emb: np.ndarray,
    text: str,
    chunk_size: int = 400,
    overlap: int = 50,
    agg: str = "max"
) -> float:
    """
    Split long text into chunks, compute embeddings, and aggregate similarity scores.
    """
    words = text.split()
    chunks = []
    start = 0
    
    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        start += chunk_size - overlap
    
    # Compute embeddings for all chunks (run in thread to avoid blocking)
    chunk_embs = await asyncio.to_thread(
        semantic_model.encode, chunks, convert_to_numpy=True
    )
    
    # Compute cosine similarity between query and each chunk
    chunk_sims = [
        float(np.dot(query_emb, c) / (np.linalg.norm(query_emb) * np.linalg.norm(c)))
        for c in chunk_embs
    ]
    
    # Aggregate similarity scores
    if agg == "mean":
        return float(np.mean(chunk_sims))
    elif agg == "max":
        return float(np.max(chunk_sims))
    elif agg == "sum":
        return float(np.sum(chunk_sims))
    else:
        raise ValueError(f"Unsupported aggregation method: {agg}")


async def embed_documents_async_comm(
    query_emb: np.ndarray,
    documents: List[str],
    chunk_size: int = 400,
    overlap: int = 50,
    agg: str = "max"
) -> List[float]:
    """
    Process multiple documents in parallel.
    """
    tasks = [
        embed_long_text_async_comm(query_emb, doc, chunk_size, overlap, agg)
        for doc in documents
    ]
    return await asyncio.gather(*tasks)


def parse_findings(findings_str: str) -> List[Dict[str, Any]]:
    """
    Safely parse the findings JSON string from report_json.
    """
    try:
        if not findings_str:
            return []
        findings = json.loads(findings_str)
        return findings if isinstance(findings, list) else []
    except (json.JSONDecodeError, TypeError):
        return []


async def rerank_communities(
    query: str,
    community_reports: List[Dict[str, Any]],
    ce_weight: float = 0.5,
    agg: str = "max"
) -> List[Dict[str, Any]]:
    """
    Rerank community reports using semantic similarity, BM25, and cross-encoder.
    
    Expected input format:
    - community_reports: List of dicts with 'report_string' and 'report_json'
    - report_json: dict with keys: title, summary, rating, rating_explanation, findings
    - findings: JSON string containing array of {summary, explanation_text}
    """
    descriptions = []
    
    for c in community_reports:
        report_json = c.get("report_json", {})
        
        # Parse findings from JSON string
        findings_str = report_json.get("findings", "[]")
        findings = parse_findings(findings_str)
        
        # Build findings text
        findings_text = " ".join([
            f"{f.get('summary', '')} {f.get('explanation_text', '')}"
            for f in findings
        ])
        
        # Combine all text fields for semantic matching
        description = (
            f"{c.get('report_string', '')} "
            # f"{report_json.get('summary', '')} "
            # f"{report_json.get('rating_explanation', '')} "
            # f"{findings_text}"
        )
        descriptions.append(description)
    
    # Compute query embedding
    query_emb = semantic_model.encode([query], convert_to_numpy=True)[0]
    
    # Semantic similarity scores
    semantic_scores = await embed_documents_async_comm(
        query_emb, descriptions, agg=agg
    )
    semantic_scores = normalize_scores(semantic_scores)
    
    # BM25 scores
    bm25_scores = await bm25_score(query, descriptions)
    
    # Combine semantic and BM25 scores
    combined_scores = combine_scores_adaptive_weights_2signals(
        semantic_scores,
        bm25_scores,
    )
    # Assign combined scores to reports
    for i, c in enumerate(community_reports):
        c['combined_score'] = combined_scores[i]
    
    # Cross-encoder reranking

    ce_results = await cross_encoder_rank_for_text_units(query, descriptions)
    raw_ce_scores = [item["score"] for item in ce_results]
    ce_scores = normalize_scores(raw_ce_scores)
    
    # Assign cross-encoder scores
    for i, c in enumerate(community_reports):
        c['ce_score'] = ce_scores[i]

    #change2
    multi_score = np.array([c["combined_score"] for c in community_reports], dtype=float)
    cross_score = np.array([c["ce_score"] for c in community_reports], dtype=float)

    # Stack signals into array [n_samples, n_signals]
    signals = np.array([multi_score, cross_score])

    # Compute variance for each signal
    variances = np.var(signals, axis=1)

    # Avoid division by zero: replace 0 variance with small epsilon
    variances = np.where(variances == 0, 1e-8, variances)

    # Inverse variance weighting
    inv_var = 1.0 / variances
    weights = inv_var / np.sum(inv_var)

    # logging.info(f"CommunityReport_Cross Encoder Adaptive Weights: {weights:.4f}")
    logging.info(
        "CommunityReport_Cross Encoder Adaptive Weights: " +
        ", ".join([f"{w:.4f}" for w in weights])
    )
    
    #change2
    
    # Compute final weighted score
    for i, c in enumerate(community_reports):
        c['final_score'] = (
            weights[1] * c['ce_score'] +
            (1 - weights[1]) * c['combined_score']
        )
    
    # Sort by final score (descending)
    ranked_reports = sorted(
        community_reports,
        key=lambda c: c['final_score'],
        reverse=True
    )
    
    return ranked_reports
# -------------------------------
# Async pipeline runner for text units
# -------------------------------
async def rerank_text_units(query, text_units, ce_weight=0.5):
    """
    Rerank text units (passages) based on their content.
    
    Args:
        query (str): The query string.
        text_units (list[dict]): List of text unit dicts with keys:
            - content (str)
            - tokens, chunk_order_index, full_doc_id, etc.
        ce_weight (float): Weight for cross-encoder score in final ranking.
    
    Returns:
        list[dict]: Sorted text units with scores attached.
    """
    # Use content as description
    descriptions = [tu.get("content", "") for tu in text_units]

    # Semantic scores
    query_emb = semantic_model.encode([query], convert_to_numpy=True)[0]
    desc_embs = await embed_documents_async(descriptions)

    semantic_scores = [
        float(np.dot(query_emb, d) / (np.linalg.norm(query_emb) * np.linalg.norm(d)))
        for d in desc_embs
    ]
    semantic_scores = normalize_scores(semantic_scores)

    # BM25 lexical scores
    bm25_scores = await bm25_score(query, descriptions)
#change1
    combined_scores = combine_scores_adaptive_weights_2signals( 
        semantic_scores, 
        bm25_scores, 
    )

    for i, tu in enumerate(text_units):
        tu['combined_score'] = combined_scores[i]

    # Cross-encoder scores
    ce_results = await cross_encoder_rank_for_text_units(query, descriptions)

    raw_ce_scores = [item["score"] for item in ce_results]
    ce_scores = normalize_scores(raw_ce_scores)

    for i, tu in enumerate(text_units):
        tu['ce_score'] = ce_scores[i]

    #change2
    multi_score = np.array([tu["combined_score"] for tu in text_units], dtype=float)
    cross_score = np.array([tu["ce_score"] for tu in text_units], dtype=float)

    # Stack signals into array [n_samples, n_signals]
    signals = np.array([multi_score, cross_score])

    # Compute variance for each signal
    variances = np.var(signals, axis=1)

    # Avoid division by zero: replace 0 variance with small epsilon
    variances = np.where(variances == 0, 1e-8, variances)

    # Inverse variance weighting
    inv_var = 1.0 / variances
    weights = inv_var / np.sum(inv_var)

    # logging.info(f"TextUnits_Cross Encoder Adaptive Weights: {weights:.4f}")
    # logging.info(
    #     "TextUnits_Cross Encoder Adaptive Weights: " +
    #     ", ".join([f"{w:.4f}" for w in weights])
    # )
    
    #change2

    #change3
    # Compute final weighted score
    for i, tu in enumerate(text_units):
        tu['final_score'] = (
            weights[1] * tu['ce_score'] +
            (1-weights[1]) * tu['combined_score']
        )
    #change3
    # Sort by final score (descending)
    ranked_text_units = sorted(
        text_units,
        key=lambda tu: tu['final_score'],
        reverse=True
    )
    
    return ranked_text_units
    