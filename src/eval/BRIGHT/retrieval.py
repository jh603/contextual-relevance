import os
import json
from datasets import load_dataset
from tqdm import tqdm
import bm25s
import Stemmer

BRIGHT_TOPICS = [
    "biology",
    "earth_science",
    "economics",
    "psychology",
    "robotics",
    "stackoverflow",
    "sustainable_living",
    "leetcode",
    "pony",
    "aops",
    "theoremqa_questions",
    "theoremqa_theorems",
]

SUBSET = "gpt4_reason"


def load_docs(split):
    return load_dataset("xlangai/BRIGHT", "documents")[split]


def load_queries(subset, split):
    return load_dataset("xlangai/BRIGHT", subset)[split]


def retrieval_bm25(queries, documents, excluded_ids):
    stemmer = Stemmer.Stemmer("porter")
    corpus = [doc["content"] for doc in documents]

    corpus_tokens = bm25s.tokenize(corpus, stopwords="en", stemmer=stemmer)
    retriever = bm25s.BM25(method="lucene", k1=0.9, b=0.4)
    retriever.index(corpus_tokens)

    all_scores = {}
    for query in tqdm(queries, desc="BM25 retrieval"):
        query_id = query.get("id")
        query_text = query.get("query")
        if not query_text:
            continue

        query_tokens = bm25s.tokenize(query_text, stopwords="en", stemmer=stemmer)
        results, scores = retriever.retrieve(query_tokens, corpus=documents, k=200)

        top_results = {}
        for doc, score in zip(results[0], scores[0]):
            doc_id = doc.get("id")
            doc_with_score = doc.copy()
            doc_with_score["score"] = float(score)
            top_results[doc_id] = doc_with_score

        ex_set = set(excluded_ids.get(str(query_id), []))
        filtered_results = {
            did: doc for did, doc in top_results.items() if did not in ex_set
        }

        all_scores[str(query_id)] = filtered_results

    return all_scores


def compute_ground_truth(examples, key="gold_ids"):
    ground_truth = {}
    for e in tqdm(examples, desc="Computing ground truth"):
        qid = e["id"]
        ground_truth[qid] = {}
        for gid in e.get(key, []):
            ground_truth[qid][gid] = 1
    return ground_truth


def main():
    os.makedirs("results", exist_ok=True)

    for topic in BRIGHT_TOPICS:
        print(f"Processing topic: {topic}")
        documents = load_docs(topic)
        queries = load_queries(SUBSET, topic)

        excluded_ids = {}

        results = retrieval_bm25(queries, documents, excluded_ids)
        output_file = os.path.join("results", f"{topic}_results.jsonl")
        with open(output_file, "w") as f_out:
            for query in queries:
                query_id = str(query.get("id"))
                query_text = query.get("query")
                if query_id in results:
                    result_obj = {
                        "topic": topic,
                        "query": query_text,
                        "query_id": query_id,
                        "results": results[query_id],
                    }
                    f_out.write(json.dumps(result_obj) + "\n")

        ground_truth = compute_ground_truth(queries, key="gold_ids")
        gt_file = os.path.join("results", f"{topic}_ground_truth.json")
        with open(gt_file, "w") as f_gt:
            json.dump(ground_truth, f_gt, indent=2)
        print(f"Ground truth saved to {gt_file}")


if __name__ == "__main__":
    main()
