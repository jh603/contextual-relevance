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


def load_final_stats(
    res_file: Path, qrels: dict[str, dict[str, int]], k: int
) -> tuple[float, float]:
    per_ndcg = {}
    per_calls = []
    for ln in res_file.read_text(encoding="utf-8").splitlines():
        obj = json.loads(ln)
        qid = str(obj.get("query_id"))
        stats = obj.get("heapify_stats", {})
        num_calls = stats.get(
            "num_llm_calls", len(obj.get("history") or obj.get("heapify_history") or [])
        )
        per_calls.append(num_calls)
        if "ranking" in obj:
            final_ranking = obj["ranking"]
        elif hist := obj.get("history") or obj.get("heapify_history"):
            final_ranking = hist[-1].get("ranking", [])
        else:
            continue

        gains = [qrels[qid].get(str(doc_id), 0) for doc_id in final_ranking[:k]]
        ideal = sorted(qrels[qid].values(), reverse=True)[:k]
        ideal += [0] * (k - len(ideal))
        idcg = dcg(ideal, k) or 1.0
        per_ndcg[qid] = dcg(gains, k) / idcg

    mean_ndcg = float(np.mean(list(per_ndcg.values()))) if per_ndcg else float("nan")
    mean_calls = float(np.mean(per_calls)) if per_calls else float("nan")
    return mean_ndcg, mean_calls


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
        default="results_heapify",
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
    calls_scores: dict[str, dict[int, float]] = defaultdict(dict)

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
        final_raw, avg_calls = load_final_stats(fp, qrels, k)

        orig_scores[topic][seed] = orig_raw
        final_scores[topic][seed] = final_raw
        calls_scores[topic][seed] = avg_calls

        logging.info(
            "Topic %s seed %d → orig=%s, final=%s, avg_llm_calls=%.2f",
            topic,
            seed,
            f"{round_sig(orig_raw,3):.4f}",
            f"{round_sig(final_raw,3):.4f}",
            avg_calls,
        )

    rows = []
    for topic in sorted(orig_scores):
        o = np.array(list(orig_scores[topic].values()))
        f = np.array(list(final_scores[topic].values()))
        c = np.array(list(calls_scores[topic].values()))

        rows.append(
            {
                "topic": topic,
                "original_ndcg": round_sig(float(o.mean()), 3),
                "final_ndcg": round_sig(float(f.mean()), 3),
                "ndcg_sd": round_sig(float(f.std(ddof=1)) if len(f) > 1 else 0.0, 3),
                "avg_llm_calls": round_sig(float(c.mean()), 3),
            }
        )

    seed_sets = [set(s.keys()) for s in final_scores.values()]
    common_seeds = set.intersection(*seed_sets) if seed_sets else set()
    if common_seeds:
        overall_o = [
            np.mean([orig_scores[t][s] for t in orig_scores])
            for s in sorted(common_seeds)
        ]
        overall_f = [
            np.mean([final_scores[t][s] for t in final_scores])
            for s in sorted(common_seeds)
        ]
        overall_c = [
            np.mean([calls_scores[t][s] for t in calls_scores])
            for s in sorted(common_seeds)
        ]

        rows.append(
            {
                "topic": "All",
                "original_ndcg": round_sig(float(np.mean(overall_o)), 3),
                "final_ndcg": round_sig(float(np.mean(overall_f)), 3),
                "ndcg_sd": round_sig(float(np.std(overall_f, ddof=1)), 3),
                "avg_llm_calls": round_sig(float(np.mean(overall_c)), 3),
            }
        )

    df = pd.DataFrame(rows)
    df.to_csv(args.output_csv, index=False, float_format="%.4f")

    print(
        f"\nSummary @nDCG@{k} per topic: orig, final ±SD, avg LLM calls (3 sig figs, padded):"
    )
    print(df.to_string(index=False, float_format=lambda x: f"{float(x):.4f}"))
    print(f"\nSaved summary table to {args.output_csv}")


if __name__ == "__main__":
    main()
