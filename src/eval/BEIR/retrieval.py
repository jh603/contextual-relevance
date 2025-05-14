import os
import json
from tqdm import tqdm
import bm25s
import Stemmer

BEIR_DATASETS = [
    "arguana",  # ArguAna
    "climate-fever",  # Climate-FEVER
    "dbpedia-entity",  # DBPedia
    "fiqa",  # FiQA-2018
    "nfcorpus",  # NFCorpus
    "scidocs",  # SCIDOCS
    "scifact",  # SciFact
    "trec-covid",  # TREC-COVID
    "webis-touche2020",  # Touche-2020
]

CACHE_DIR = "../cache/my_datasets"
MAX_QUERIES = 100


def load_jsonl(file_path):
    data = []
    with open(file_path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data


def load_docs(path):
    """
    path: full path to the subfolder or dataset directory
    containing corpus.jsonl
    """
    corpus_file = os.path.join(path, "corpus.jsonl")
    docs = load_jsonl(corpus_file)
    for doc in docs:
        if "title" in doc and doc["title"]:
            doc["text"] = doc["title"] + " " + doc["text"]
    for doc in docs:
        doc["id"] = doc.get("_id")
    return docs


def load_queries(path):
    queries_file_jsonl = os.path.join(path, "queries.jsonl")
    queries_file_tsv = os.path.join(path, "queries.tsv")
    if os.path.exists(queries_file_jsonl):
        queries_raw = load_jsonl(queries_file_jsonl)
        queries = []
        for q in queries_raw:
            queries.append({"id": q.get("_id"), "query": q.get("text")})
        return queries
    elif os.path.exists(queries_file_tsv):
        queries = []
        with open(queries_file_tsv, "r") as f:
            lines = f.readlines()
        if not lines:
            return queries
        header = lines[0].strip().split("\t")
        if "query-id" not in [h.lower() for h in header]:
            raise ValueError("TSV query file does not contain a 'query-id' column.")
        second_field = header[1].lower() if len(header) > 1 else ""
        for line in lines[1:]:
            parts = line.strip().split("\t")
            if not parts or len(parts) < 1:
                continue
            qid = parts[0]
            if second_field == "text" and len(parts) >= 2:
                qtext = parts[1]
            else:
                qtext = qid
            queries.append({"id": qid, "query": qtext})
        return queries
    else:
        raise FileNotFoundError(
            f"Neither queries.jsonl nor queries.tsv found at: {path}"
        )


def load_qrels(path):
    qrels_file = os.path.join(path, "qrels", "test.tsv")
    if not os.path.exists(qrels_file):
        raise FileNotFoundError(f"No test.tsv file at {qrels_file}")
    ground_truth = {}
    with open(qrels_file, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.lower().startswith("query-id"):
                continue
            parts = line.split("\t")
            if len(parts) < 3:
                continue
            qid, docid, score = parts[0], parts[1], parts[2]
            if qid not in ground_truth:
                ground_truth[qid] = {}
            ground_truth[qid][docid] = int(score)
    return ground_truth


def retrieval_bm25(queries, documents, excluded_ids):
    stemmer = Stemmer.Stemmer("porter")
    corpus = [doc["text"] for doc in documents]

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


def process_dataset(dataset):
    print(f"Processing dataset: {dataset}")
    try:
        dataset_path = os.path.join(CACHE_DIR, dataset)
        documents = load_docs(dataset_path)
        queries = load_queries(dataset_path)
    except Exception as e:
        print(f"Failed to load dataset '{dataset}': {e}")
        return [], {}

    test_qrels_path = os.path.join(dataset_path, "qrels", "test.tsv")
    ground_truth = {}
    if os.path.exists(test_qrels_path):
        try:
            ground_truth = load_qrels(dataset_path)
            valid_query_ids = set(ground_truth.keys())
            queries = [q for q in queries if str(q["id"]) in valid_query_ids]
            if len(queries) > MAX_QUERIES:
                queries = queries[:MAX_QUERIES]
        except Exception as e:
            print(f"Error loading test file for dataset '{dataset}': {e}")
    else:
        print(
            f"No test file found for dataset '{dataset}'. Proceeding with all queries."
        )

    excluded_ids = {}
    results_dict = retrieval_bm25(queries, documents, excluded_ids)

    results_list = []
    for query in queries:
        query_id = str(query.get("id"))
        query_text = query.get("query")
        if query_id in results_dict:
            result_obj = {
                "dataset": dataset,
                "query": query_text,
                "query_id": query_id,
                "results": results_dict[query_id],
            }
            results_list.append(result_obj)
    return results_list, ground_truth


def main():
    os.makedirs("results", exist_ok=True)

    for dataset in BEIR_DATASETS:
        results_list, ground_truth = process_dataset(dataset)
        if not results_list:
            continue
        output_file = os.path.join("results", f"{dataset}_results.jsonl")
        with open(output_file, "w") as f_out:
            for item in results_list:
                f_out.write(json.dumps(item) + "\n")
        print(
            f"[{dataset}] Results saved to {output_file}. Number of queries: {len(results_list)}"
        )

        if ground_truth:
            gt_file = os.path.join("results", f"{dataset}_ground_truth.json")
            with open(gt_file, "w") as f_gt:
                json.dump(ground_truth, f_gt, indent=2)
            print(f"[{dataset}] Ground truth saved to {gt_file}")


if __name__ == "__main__":
    main()
