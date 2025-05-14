import json
import random
import numpy as np
from pathlib import Path
from datasets import load_dataset
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

from src.reranking_algorithm import Reranking

EXPERIMENTS = [
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
    ("heapify", {"snapshot_interval": 1}),
]

SEEDS = [42, 43, 44]


def load_bright_data(topic, max_doc_length=10_000):
    docs_ds = load_dataset("xlangai/BRIGHT", "documents")[topic]
    qs_ds = load_dataset("xlangai/BRIGHT", "examples")[topic]
    if max_doc_length:
        for d in docs_ds:
            if len(d["content"]) > max_doc_length:
                d["content"] = d["content"][:max_doc_length] + "…"
    doc_dict = {d["id"]: d for d in docs_ds}
    query_dict = {q["id"]: q["query"] for q in qs_ds}
    return doc_dict, query_dict


def load_qrels(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def process_query(
    topic, record, reranker, doc_dict, query_dict, qrels, cfg, policy_name
):
    qid = record["query_id"]
    if qid not in query_dict:
        return None
    query_text = query_dict[qid]
    golds = qrels.get(qid, {})

    top_ids = list(record["results"].keys())[:100]
    docs = [
        {"id": did, "content": doc_dict[did]["content"]}
        for did in top_ids
        if did in doc_dict
    ]
    if not docs:
        return None

    if policy_name == "heapify":
        ranking, stats, history, acc_history = reranker.rank_heapify(
            topic=topic,
            query=query_text,
            docs=docs,
            golds=golds,
            snapshot_interval=cfg["snapshot_interval"],
        )
        return {
            **record,
            "reranked_ids": ranking,
            "heapify_stats": stats,
            "heapify_history": history,
        }
    else:
        ranking, stats, history, llm_accuracy = reranker.rank(
            topic=topic,
            query=query_text,
            docs=docs,
            scores=[],
            golds=golds,
            iterations=cfg["iterations"],
            forced_explore_iterations=cfg["forced_explore_iterations"],
            batch_size=cfg["batch_size"],
            snapshot_interval=cfg["snapshot_interval"],
        )
        return {
            **record,
            "reranked_ids": ranking,
            "llm_stats": stats,
            "llm_accuracy": llm_accuracy,
            "history": history,
        }


def run_policy(topic, infile, doc_dict, query_dict, qrels, policy_name, cfg, seed):
    random.seed(seed)
    np.random.seed(seed)

    reranker = Reranking(model="Qwen/Qwen2.5-7B-Instruct", temperature=0.6)

    with open(infile, "r", encoding="utf-8") as f:
        records = [json.loads(line) for line in f]

    outpath = infile.replace("_results.jsonl", f"_{policy_name}_seed{seed}.jsonl")
    out_recs = []
    with ThreadPoolExecutor(max_workers=64) as executor, open(
        outpath, "w", encoding="utf-8"
    ) as fout:

        futures = {
            executor.submit(
                process_query,
                topic,
                rec,
                reranker,
                doc_dict,
                query_dict,
                qrels,
                cfg,
                policy_name,
            ): rec
            for rec in records
        }
        for fut in tqdm(
            as_completed(futures), total=len(futures), desc=f"{policy_name} seed={seed}"
        ):
            res = fut.result()
            if res:
                fout.write(json.dumps(res) + "\n")
                out_recs.append(res)

    print(f"Wrote {len(out_recs)} records to {outpath}")
    return outpath


def main():
    results_dir = Path("results")
    for infile in sorted(results_dir.glob("*_results.jsonl")):
        topic = infile.stem.replace("_results", "")
        qrels_f = results_dir / f"{topic}_ground_truth.json"
        if not qrels_f.exists():
            continue

        print(f"\n== Processing topic '{topic}' ==")
        doc_dict, query_dict = load_bright_data(topic)
        qrels = load_qrels(qrels_f)

        for policy_name, cfg in EXPERIMENTS:
            for seed in SEEDS:
                outpath = results_dir / f"{topic}_{policy_name}_seed{seed}.jsonl"
                if outpath.exists():
                    print(f"Skipping {policy_name} seed={seed}, exists")
                    continue
                run_policy(
                    topic=topic,
                    infile=str(infile),
                    doc_dict=doc_dict,
                    query_dict=query_dict,
                    qrels=qrels,
                    policy_name=policy_name,
                    cfg=cfg,
                    seed=seed,
                )


if __name__ == "__main__":
    main()
