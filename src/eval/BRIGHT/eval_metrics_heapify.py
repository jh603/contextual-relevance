import argparse
import json
import math
import logging
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import pytrec_eval


def dcg(rels: list[int], k: int = 10) -> float:
    return sum((2**r - 1) / math.log2(i + 2) for i, r in enumerate(rels[:k]))


def round_sig(x: float, sig: int = 3) -> float:
    if math.isnan(x):
        return x
    return float(f"{x:.{sig}g}")


def load_qrels(path: Path) -> dict[str, dict[str, int]]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_final_ndcg(res_file: Path, qrels: dict[str, dict[str, int]], k: int) -> float:
    per_query = {}
    for ln in res_file.read_text(encoding="utf-8").splitlines():
        obj = json.loads(ln)
        qid = str(obj.get("query_id"))

        hist = obj.get("history") or obj.get("heapify_history") or []
        if "ranking" in obj:
            final_ranking = obj["ranking"]
        elif hist:
            final_ranking = hist[-1].get("ranking", [])
        else:
            continue

        gains = [qrels[qid].get(str(doc_id), 0) for doc_id in final_ranking[:k]]
        ideal = sorted(qrels[qid].values(), reverse=True)[:k]
        ideal += [0] * (k - len(ideal))
        idcg = dcg(ideal, k) or 1.0

        per_query[qid] = dcg(gains, k) / idcg

    return float(np.mean(list(per_query.values()))) if per_query else float("nan")


def compute_baseline_ndcg(results_file: Path, qrels_file: Path, k: int = 10) -> float:
    qrels = json.loads(qrels_file.read_text(encoding="utf-8"))
    results = {}
    for line in results_file.open("r", encoding="utf-8"):
        obj = json.loads(line)
        qid = str(obj["query_id"])
        rr = obj["results"]
        ranked = sorted(
            rr.items(), key=lambda x: float(x[1].get("score", 0)), reverse=True
        )
        results[qid] = {d: float(dt.get("score", 0)) for d, dt in ranked[:100]}

    ndcg_str = f"ndcg_cut.{k}"
    evaluator = pytrec_eval.RelevanceEvaluator(qrels, {ndcg_str})
    scores = evaluator.evaluate(results)
    per_q = [metrics[f"ndcg_cut_{k}"] for metrics in scores.values()]
    return float(np.mean(per_q)) if per_q else float("nan")


def main():
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--input-dir",
        "-i",
        default="results_heapify_weak_bm25",
        help="where to find *_heapify_seed*.jsonl, *_results.jsonl and *_ground_truth.json",
    )
    ap.add_argument("--k", "-k", type=int, default=10, help="cutoff k for nDCG@k")
    ap.add_argument(
        "--output-csv",
        "-o",
        default="heapify_ndcg_table.csv",
        help="where to save the summary CSV",
    )
    args = ap.parse_args()

    input_dir = Path(args.input_dir)
    k = args.k

    orig_scores: dict[str, dict[int, float]] = defaultdict(dict)
    final_scores: dict[str, dict[int, float]] = defaultdict(dict)

    for fp in sorted(input_dir.glob("*_heapify_seed*.jsonl")):
        topic, seed_str = fp.stem.split("_heapify_seed")
        seed = int(seed_str)
        qrels_file = input_dir / f"{topic}_ground_truth.json"
        results_file = input_dir / f"{topic}_results.jsonl"
        if not qrels_file.exists() or not results_file.exists():
            logging.warning("Skipping %s seed %d: missing files", topic, seed)
            continue

        orig_raw = compute_baseline_ndcg(results_file, qrels_file, k)
        qrels = load_qrels(qrels_file)
        final_raw = load_final_ndcg(fp, qrels, k)

        orig_scores[topic][seed] = orig_raw
        final_scores[topic][seed] = final_raw

        logging.info(
            "Topic %s seed %d → orig=%s, final=%s",
            topic,
            seed,
            f"{round_sig(orig_raw,3):.4f}",
            f"{round_sig(final_raw,3):.4f}",
        )

    rows = []
    for topic in sorted(orig_scores):
        o = np.array(list(orig_scores[topic].values()))
        f = np.array(list(final_scores[topic].values()))

        orig_mean_raw = float(o.mean())
        final_mean_raw = float(f.mean())
        sd_raw = float(f.std(ddof=1)) if len(f) > 1 else 0.0

        rows.append(
            {
                "topic": topic,
                "original_ndcg": round_sig(orig_mean_raw, 3),
                "final_ndcg": round_sig(final_mean_raw, 3),
                "ndcg_sd": round_sig(sd_raw, 3),
            }
        )

    # overall “All” row
    seed_sets = [set(s.keys()) for s in final_scores.values()]
    common_seeds = set.intersection(*seed_sets) if seed_sets else set()
    overall_orig_raw, overall_final_raw = [], []

    for s in sorted(common_seeds):
        overall_orig_raw.append(np.mean([orig_scores[t][s] for t in orig_scores]))
        overall_final_raw.append(np.mean([final_scores[t][s] for t in final_scores]))

    if overall_final_raw:
        rows.append(
            {
                "topic": "All",
                "original_ndcg": round_sig(float(np.mean(overall_orig_raw)), 3),
                "final_ndcg": round_sig(float(np.mean(overall_final_raw)), 3),
                "ndcg_sd": round_sig(float(np.std(overall_final_raw, ddof=1)), 3),
            }
        )

    df = pd.DataFrame(rows)

    df.to_csv(args.output_csv, index=False, float_format="%.4f")

    print(f"\nSummary nDCG@{k} per topic: original, final ±SD (3 sig figs), padded:")
    print(df.to_string(index=False, float_format=lambda x: f"{float(x):.4f}"))
    print(f"\nSaved summary table to {args.output_csv}")


if __name__ == "__main__":
    main()
