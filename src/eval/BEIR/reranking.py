import os
import json
import glob
import random
import numpy as np
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from typing import Dict

from src.reranking_algorithm import Reranking

CACHE_DIR = os.getenv("BEIR_CACHE_DIR", "../cache/my_datasets")

FILE_TO_RERANK = None

EXPERIMENTS = [
    ("heapify", {"snapshot_interval": 1}),
    (
        "uniform",
        {
            "iterations": 100,
            "forced_explore_iterations": 100,
            "snapshot_interval": 1,
            "batch_size": 10,
        },
    ),
    (
        "ts_setrank_75_25",
        {
            "iterations": 100,
            "forced_explore_iterations": 75,
            "snapshot_interval": 1,
            "batch_size": 10,
        },
    ),
    (
        "ts_setrank_50_50",
        {
            "iterations": 100,
            "forced_explore_iterations": 50,
            "snapshot_interval": 1,
            "batch_size": 10,
        },
    ),
    (
        "ts_setrank_25_75",
        {
            "iterations": 100,
            "forced_explore_iterations": 25,
            "snapshot_interval": 1,
            "batch_size": 10,
        },
    ),
    (
        "ts_setrank_0_100",
        {
            "iterations": 100,
            "forced_explore_iterations": 0,
            "snapshot_interval": 1,
            "batch_size": 10,
        },
    ),
]
SEEDS = [42, 43, 44]


def load_jsonl(file_path: str) -> list:
    data = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data


def load_docs(path: str) -> list:
    corpus_file = os.path.join(path, "corpus.jsonl")
    docs = load_jsonl(corpus_file)
    for doc in docs:
        if doc.get("title"):
            doc["text"] = doc["title"] + " " + doc["text"]
        doc["id"] = doc.get("_id")
    return docs


def load_queries(path: str) -> list:
    queries_file = os.path.join(path, "queries.jsonl")
    if not os.path.exists(queries_file):
        raise FileNotFoundError(f"queries.jsonl not found in {path}")
    raw = load_jsonl(queries_file)
    return [{"id": q.get("_id"), "query": q.get("text")} for q in raw]


def load_beir_data(dataset: str, subfolder: str = None):
    if dataset == "cqadupstack" and subfolder:
        ds_path = os.path.join(CACHE_DIR, dataset, subfolder)
    else:
        ds_path = os.path.join(CACHE_DIR, dataset)

    docs = load_docs(ds_path)
    queries = load_queries(ds_path)

    query_dict = {q["id"]: q["query"] for q in queries}
    doc_dict = {d["id"]: d for d in docs}
    content_to_doc = {d["text"]: d for d in docs}
    return query_dict, doc_dict, content_to_doc


def load_qrels(path: str) -> Dict[str, Dict[str, int]]:
    """Load a BEIR-style ground_truth.json into {qid:{docid:rel}}."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def process_beir_query(
    record,
    reranker,
    query_dict,
    doc_dict,
    content_to_doc,
    policy_name,
    cfg,
    golds_map: Dict[str, Dict[str, int]],
):
    qid = record.get("query_id")
    if qid not in query_dict:
        return None
    query_text = query_dict[qid]
    top_ids = list(record.get("results", {}).keys())[:100]
    docs = [
        {"id": did, "text": doc_dict[did]["text"], "content": doc_dict[did]["text"]}
        for did in top_ids
        if did in doc_dict
    ]
    if not docs:
        return None

    if policy_name == "heapify":
        golds = golds_map.get(qid, {})
        ranking, stats, history, _ = reranker.rank_heapify(
            topic=record.get("dataset"),
            query=query_text,
            docs=docs,
            golds=golds,
            snapshot_interval=cfg["snapshot_interval"],
        )
        record["reranked_ids"] = ranking
        record["heapify_stats"] = stats
        record["heapify_history"] = history
    else:
        golds = golds_map.get(qid, {})

        ranking, stats, history, llm_acc = reranker.rank(
            topic=record.get("dataset"),
            query=query_text,
            docs=docs,
            scores=[],
            golds=golds,
            iterations=cfg["iterations"],
            forced_explore_iterations=cfg["forced_explore_iterations"],
            batch_size=cfg["batch_size"],
            snapshot_interval=cfg["snapshot_interval"],
        )

        record["reranked_ids"] = ranking
        record["llm_stats"] = stats
        record["llm_accuracy"] = llm_acc
        record["history"] = history

    return record


def run_policy_beir(
    infile: str,
    output_dir: Path,
    query_dict,
    doc_dict,
    content_to_doc,
    policy_name: str,
    cfg: dict,
    golds_map: Dict[str, Dict[str, int]],
    seed: int,
    model_path: str,
):
    random.seed(seed)
    np.random.seed(seed)
    reranker = Reranking(model=model_path, temperature=0.6)

    with open(infile, "r", encoding="utf-8") as f:
        records = [json.loads(line) for line in f]

    base = Path(infile).stem.replace("_results", "")
    outpath = Path(output_dir) / f"{base}_{policy_name}_seed{seed}.jsonl"
    outpath.parent.mkdir(parents=True, exist_ok=True)

    if outpath.exists():
        print(f"Skipping {policy_name} seed={seed}, exists: {outpath}")
        return

    with ThreadPoolExecutor(max_workers=20) as executor, open(
        outpath, "w", encoding="utf-8"
    ) as fout:
        futures = {
            executor.submit(
                process_beir_query,
                record,
                reranker,
                query_dict,
                doc_dict,
                content_to_doc,
                policy_name,
                cfg,
                golds_map,
            ): record
            for record in records
        }
        for fut in tqdm(
            as_completed(futures), total=len(futures), desc=f"{policy_name} seed={seed}"
        ):
            res = fut.result()
            if res:
                fout.write(json.dumps(res) + "\n")
    print(f"Wrote results to {outpath}")


def main():
    model_dir = os.getenv("RERANK_MODEL", "Qwen/Qwen2.5-7B-Instruct")
    results_dir = Path("results")

    if FILE_TO_RERANK:
        files = [FILE_TO_RERANK]
    else:
        files = sorted(glob.glob(str(results_dir / "*_results.jsonl")))

    qrels_map: Dict[str, Dict[str, Dict[str, int]]] = {}
    for fp in files:
        topic = Path(fp).stem.split("_")[0]
        if topic in qrels_map:
            continue
        ground = results_dir / f"{topic}_ground_truth.json"
        qrels_map[topic] = load_qrels(str(ground))

    for file_path in files:
        print(f"\n== Processing file: {file_path} ==")
        with open(file_path, "r", encoding="utf-8") as f:
            first = json.loads(f.readline())
        topic = first.get("dataset")
        if not topic:
            print(f"No dataset field in {file_path}, skipping")
            continue
        subfolder = first.get("subfolder") if topic == "cqadupstack" else None

        query_dict, doc_dict, content_to_doc = load_beir_data(
            topic, subfolder=subfolder
        )

        for policy_name, cfg in EXPERIMENTS:
            for seed in SEEDS:
                run_policy_beir(
                    infile=file_path,
                    output_dir=results_dir,
                    query_dict=query_dict,
                    doc_dict=doc_dict,
                    content_to_doc=content_to_doc,
                    policy_name=policy_name,
                    cfg=cfg,
                    golds_map=qrels_map[topic],
                    seed=seed,
                    model_path=model_dir,
                )


if __name__ == "__main__":
    main()
