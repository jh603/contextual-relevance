from __future__ import annotations

from collections import defaultdict
import argparse
import json
import logging
import math
import random
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import pytrec_eval
from scipy.stats import ttest_1samp, sem

BRIGHT_TOPICS: set[str] = {
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
}
DATASET_SET = set(BRIGHT_TOPICS)

random.seed(42)
np.random.seed(42)
mpl.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["DejaVu Serif"],
        "font.size": 11,
        "figure.dpi": 300,
        "axes.titlesize": 14,
        "axes.labelsize": 12,
        "axes.linewidth": 1.2,
        "axes.grid": True,
        "grid.linestyle": "--",
        "grid.alpha": 0.5,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "lines.linewidth": 2,
        "lines.markersize": 6,
        "legend.fontsize": 10,
        "legend.frameon": True,
        "legend.framealpha": 0.8,
        "legend.edgecolor": "black",
    }
)

MOVING_WINDOW = 15


def dcg(rels, k: int = 10):
    return sum((2**r - 1) / math.log2(i + 2) for i, r in enumerate(rels[:k]))


def load_qrels(p: Path):
    return json.loads(p.read_text())


def load_history(fp: Path):
    d: dict[str, list] = {}
    for ln in fp.read_text().splitlines():
        obj = json.loads(ln)
        qid = obj.get("query_id")
        if qid is not None:
            d[str(qid)] = obj.get("history", [])
    return d


def parse_stem(stem: str, topics: set[str] = DATASET_SET) -> Tuple[str, str, str]:
    rest, _, seed = stem.rpartition("_seed")
    if not rest or not seed:
        raise ValueError(stem)
    for topic in sorted(topics, key=len, reverse=True):
        prefix = topic + "_"
        if rest.startswith(prefix):
            return topic, rest[len(prefix) :], seed
        if rest == topic:
            return topic, "", seed
    raise ValueError(f"{stem} – topic not in BEIR_DATASETS")


def aggregate_metrics(hist: dict, qrels: dict, k: int = 10) -> pd.DataFrame:
    rows: list[dict] = []
    for qid, snaps in hist.items():
        qid = str(qid)
        gains_dict = {str(d): rel for d, rel in qrels.get(qid, {}).items()}
        ideal = sorted(gains_dict.values(), reverse=True)[:k]
        ideal += [0] * (k - len(ideal))
        idcg = dcg(ideal, k) or 1.0
        for s in snaps:
            ranking = s.get("ranking", [])
            gains = [gains_dict.get(str(d), 0) for d in ranking[:k]]
            rows.append(
                {
                    "query_id": qid,
                    "ranking": ranking,
                    "step": s.get("step", np.nan),
                    "ndcg": dcg(gains, k) / idcg,
                    "cum_reg": s.get("regret", {}).get("cumulative", np.nan),
                    "accuracy": s.get("batch_accuracy", np.nan),
                    "cum_accuracy": s.get("cumulative_accuracy", np.nan),
                }
            )
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    num_cols = [
        c
        for c in ("step", "ndcg", "cum_reg", "accuracy", "cum_accuracy")
        if c in df.columns
    ]
    df[num_cols] = df[num_cols].apply(pd.to_numeric, errors="coerce")
    return (
        df.groupby("step")
        .agg(
            ndcg_mean=("ndcg", "mean"),
            ndcg_std=("ndcg", "std"),
            cum_reg_mean=("cum_reg", "mean"),
            cum_reg_std=("cum_reg", "std"),
            acc_mean=("accuracy", "mean"),
            acc_std=("accuracy", "std"),
            cum_acc_mean=("cum_accuracy", "mean"),
            cum_acc_std=("cum_accuracy", "std"),
        )
        .reset_index()
    )


def aggregate_runs(dfs: List[pd.DataFrame]) -> pd.DataFrame:
    dfs = [d for d in dfs if not d.empty]
    if not dfs:
        return pd.DataFrame()
    n_seeds = len(dfs)
    for i, df in enumerate(dfs):
        df["run"] = i
    big = pd.concat(dfs, ignore_index=True)
    return (
        big.groupby("step")
        .agg(
            ndcg_mean=("ndcg_mean", "mean"),
            ndcg_sem=("ndcg_mean", lambda x: x.std(ddof=1) / math.sqrt(n_seeds)),
            cum_reg_mean=("cum_reg_mean", "mean"),
            cum_reg_sem=("cum_reg_mean", lambda x: x.std(ddof=1) / math.sqrt(n_seeds)),
            acc_mean=("acc_mean", "mean"),
            acc_sem=("acc_mean", lambda x: x.std(ddof=1) / math.sqrt(n_seeds)),
            cum_acc_mean=("cum_acc_mean", "mean"),
            cum_acc_sem=("cum_acc_mean", lambda x: x.std(ddof=1) / math.sqrt(n_seeds)),
        )
        .reset_index()
    )


def compute_baseline(res_file: Path, qrels_file: Path, k: int = 10):
    if not (res_file.exists() and qrels_file.exists()):
        return None
    qrels = load_qrels(qrels_file)
    retrieval = {}
    for ln in res_file.open():
        obj = json.loads(ln)
        ranked = sorted(
            obj.get("results", {}).items(),
            key=lambda x: float(x[1]["score"]),
            reverse=True,
        )[:100]
        retrieval[str(obj["query_id"])] = {
            d: float(info["score"]) for d, info in ranked
        }
    metric_dot, metric_us = f"ndcg_cut.{k}", f"ndcg_cut_{k}"
    evaluator = pytrec_eval.RelevanceEvaluator(qrels, {metric_dot, metric_us})
    scores = evaluator.evaluate(retrieval)
    first = next(iter(scores.values()))
    key = metric_dot if metric_dot in first else metric_us
    return float(np.mean([v[key] for v in scores.values()]))


def plot_ndcg(ax, dfd, baseline):
    xmax = max(df.step.max() for df in dfd.values())
    for p, df in dfd.items():
        if df.empty:
            continue
        ax.plot(df.step, df.ndcg_mean, label=p, markevery=5)
    if baseline is not None:
        ax.axhline(baseline, ls="--", c="k", label="Baseline")
    ax.set(xlim=(0, xmax), xlabel="LLM calls", ylabel="nDCG@10", title="nDCG vs calls")
    ax.legend(fontsize=6)


def plot_aurc(ax, dfd):
    vals = {
        p: df.cum_reg_mean.iloc[-1] / df.step.iloc[-1]
        for p, df in dfd.items()
        if not df.empty
    }
    ax.bar(range(len(vals)), vals.values())
    ax.set_xticks(range(len(vals)), vals.keys(), rotation=30, ha="right")
    ax.set(title="AURC", ylabel="Avg regret/call")


def plot_smoothed_acc(ax, dfd):
    xmax = max(df.step.max() for df in dfd.values())
    for p, df in dfd.items():
        d = df.dropna(subset=["acc_mean"])
        ax.plot(
            d.step,
            d.acc_mean.rolling(MOVING_WINDOW, min_periods=1).mean(),
            label=p,
            markevery=5,
        )
    ax.set(
        xlim=(0, xmax), xlabel="LLM calls", ylabel="Accuracy", title="Smoothed accuracy"
    )
    ax.legend(fontsize=6)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-dir", default="results_paper")
    ap.add_argument("--output-dir", default="results_paper")
    ap.add_argument("--k", type=int, default=10, help="cutoff for nDCG@k")
    args = ap.parse_args()
    k = args.k

    in_dir, out_dir = Path(args.input_dir), Path(args.output_dir)
    out_dir.mkdir(exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    runs: dict[tuple[str, str], list[Path]] = {}
    for fp in in_dir.glob("*.jsonl"):
        if fp.stem.endswith(("_results", "_ground_truth")):
            continue
        try:
            topic, policy, _ = parse_stem(fp.stem)
        except ValueError as e:
            logging.warning("skip %s – %s", fp.name, e)
            continue
        runs.setdefault((topic, policy), []).append(fp)

    topic_eval: dict[str, dict[str, pd.DataFrame]] = {}
    phase_rows: list[dict] = []

    for (topic, policy), fps in runs.items():
        qrels_file = in_dir / f"{topic}_ground_truth.json"
        qrels = load_qrels(qrels_file)
        idcgs = {}
        for qid, gains in qrels.items():
            ideal = sorted(gains.values(), reverse=True)[:k]
            ideal += [0] * (k - len(ideal))
            idcgs[qid] = dcg(ideal, k) or 1.0

        dfs = [aggregate_metrics(load_history(fp), qrels) for fp in fps]
        agg = aggregate_runs(dfs)
        if agg.empty:
            logging.warning("%s/%s – all seeds empty, skipped", topic, policy)
            continue

        agg.to_csv(out_dir / f"{topic}_{policy}_eval.csv", index=False)
        topic_eval.setdefault(topic, {})[policy] = agg

        for fp in fps:
            for ln in fp.read_text().splitlines():
                snap_hist = json.loads(ln).get("history", [])
                for snap in snap_hist:
                    ph = snap.get("phase")
                    acc = snap.get("batch_accuracy")
                    if acc is not None and ph in ("explore", "exploit"):
                        phase_rows.append(
                            dict(topic=topic, policy=policy, phase=ph, accuracy=acc)
                        )

    for topic, pol_dict in topic_eval.items():
        if not pol_dict:
            continue
        qrels_file = in_dir / f"{topic}_ground_truth.json"
        qrels = load_qrels(qrels_file)
        fig, axs = plt.subplots(3, 2, figsize=(10, 8), constrained_layout=True)

        baseline = compute_baseline(in_dir / f"{topic}_results.jsonl", qrels_file)
        xmax = max(df.step.max() for df in pol_dict.values())
        for policy, df in pol_dict.items():
            axs[0, 0].plot(df.step, df.ndcg_mean, label=policy, markevery=5)
        if baseline is not None:
            axs[0, 0].axhline(baseline, ls="--", c="k", label="Baseline")
        axs[0, 0].set(
            xlim=(0, xmax), xlabel="LLM calls", ylabel="nDCG@10", title="nDCG vs calls"
        )
        axs[0, 0].legend(fontsize=6)

        plot_aurc(axs[0, 1], pol_dict)

        policies = list(pol_dict.keys())
        final_ndcgs = [df.ndcg_mean.iloc[-1] for df in pol_dict.values()]
        tbl = axs[1, 0].table(
            cellText=[[p, f"{nd:.3f}"] for p, nd in zip(policies, final_ndcgs)],
            colLabels=["Policy", f"Final nDCG@{k}"],
            loc="center",
        )
        axs[1, 0].axis("off")
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(8)
        tbl.scale(1, 1.5)

        axs[1, 1].boxplot(
            [df.acc_mean.dropna().values for df in pol_dict.values()],
            tick_labels=policies,
            vert=True,
        )
        axs[1, 1].set_title("Accuracy distribution")
        axs[1, 1].tick_params(axis="x", labelrotation=30)
        for lbl in axs[1, 1].get_xticklabels():
            lbl.set_ha("right")

        fin_means = [df.cum_reg_mean.iloc[-1] for df in pol_dict.values()]
        axs[2, 0].bar(policies, fin_means)
        axs[2, 0].set_title("Final cumulative regret")
        axs[2, 0].tick_params(axis="x", rotation=30)

        per_seed_scores = {p: {} for p in policies}
        for p in policies:
            for seed_i, fp in enumerate(runs[(topic, p)]):
                hist = load_history(fp)
                qs = [qid for qid, snaps in hist.items() if snaps and qid in qrels]
                vals = []
                for qid in qs:
                    last = hist[qid][-1]["ranking"][:k]
                    score = dcg([qrels[qid].get(str(d), 0) for d in last], k) / dcg(
                        sorted(qrels[qid].values(), reverse=True)[:k]
                        + [0] * max(0, k - len(qrels[qid])),
                        k,
                    )
                    vals.append(score)
                per_seed_scores[p][seed_i] = np.mean(vals) if vals else np.nan

        uni = per_seed_scores.get("uniform", {})
        rows = []
        for p in policies:
            if p == "uniform":
                continue
            pol = per_seed_scores[p]
            seed_inds = sorted(set(uni) & set(pol))
            diffs = np.array([pol[i] - uni[i] for i in seed_inds])
            mean_diff = np.nanmean(diffs)
            pval = ttest_1samp(diffs, 0.0, nan_policy="omit").pvalue
            rows.append([p, f"{mean_diff:+.3f}", f"{pval:.3f}"])

        tbl2 = axs[2, 1].table(
            cellText=rows, colLabels=["Policy", f"ΔnDCG@{k}", "p-value"], loc="center"
        )
        tbl2.auto_set_font_size(False)
        tbl2.set_fontsize(8)
        tbl2.scale(1, 1.5)
        axs[2, 1].axis("off")
        fig.suptitle(f"Eval summary – {topic}", y=1.02)
        fig.savefig(out_dir / f"{topic}_summary.png")
        plt.close(fig)

    if topic_eval:
        global_curves: dict[str, pd.DataFrame] = {}
        temp: dict[str, list[pd.DataFrame]] = defaultdict(list)
        for pol_dict in topic_eval.values():
            for policy, df in pol_dict.items():
                temp[policy].append(df[["step", "ndcg_mean"]])
        for policy, dfs in temp.items():
            big = pd.concat(dfs, ignore_index=True)
            global_curves[policy] = big.groupby("step", as_index=False).agg(
                ndcg_mean=("ndcg_mean", "mean")
            )
        fig, ax = plt.subplots(figsize=(8, 5))
        for policy, df in global_curves.items():
            ax.plot(df.step, df.ndcg_mean, label=policy, markevery=5)
        ax.set(
            xlabel="LLM calls",
            ylabel=f"nDCG@{k}",
            title="Cumulative nDCG over LLM calls (average across datasets)",
        )
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(out_dir / "cumulative_ndcg_summary.png")
        plt.close(fig)
        print(
            "Cumulative nDCG summary saved to",
            out_dir / "cumulative_ndcg_summary.png",
        )

    topic_policy_scores: dict[str, dict[str, float]] = defaultdict(dict)
    for (topic, policy), fps in runs.items():
        if topic not in DATASET_SET:
            continue
        qrels_file = in_dir / f"{topic}_ground_truth.json"
        if not qrels_file.exists():
            continue
        qrels = load_qrels(qrels_file)
        vals = []
        for fp in fps:
            hist = load_history(fp)
            per_q_scores = []
            for qid, snaps in hist.items():
                if not snaps:
                    continue
                last = snaps[-1]["ranking"][:k]
                gains = [qrels[qid].get(str(d), 0) for d in last]
                ideal = sorted(qrels[qid].values(), reverse=True)[:k] + [0] * (
                    k - len(qrels[qid])
                )
                idcg = dcg(ideal, k) or 1.0
                per_q_scores.append(dcg(gains, k) / idcg)
            if per_q_scores:
                vals.append(np.mean(per_q_scores))
        if vals:
            topic_policy_scores[topic][policy] = float(np.mean(vals))
    policies = sorted({p for _, p in runs.keys()} - {"uniform"})
    deltas: dict[str, list[float]] = {p: [] for p in policies}
    topics = sorted(topic_policy_scores.keys())
    for topic in topics:
        base = topic_policy_scores[topic].get("uniform", None)
        if base is None:
            continue
        for p in policies:
            val = topic_policy_scores[topic].get(p, None)
            if val is not None:
                deltas[p].append(val - base)
    means = [np.mean(deltas[p]) for p in policies]
    sems = [sem(deltas[p], ddof=1) for p in policies]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(policies, means)
    ax.axhline(0, color="k", linestyle="--", linewidth=1)
    ax.set_ylabel(f"Δ nDCG@{k} over uniform")
    ax.set_title("TS sampling: mean ΔnDCG@10 across datasets")
    ax.tick_params(axis="x", rotation=30)
    for lbl in ax.get_xticklabels():
        lbl.set_ha("right")
    fig.tight_layout()
    fig.savefig(out_dir / "delta_ndcg_summary.png")
    plt.close(fig)
    print("✅  ΔnDCG summary saved to", out_dir / "delta_ndcg_summary.png")

    topics = BRIGHT_TOPICS
    policies = [
        "uniform",
        "ts_setrank_75_25",
        "ts_setrank_50_50",
        "ts_setrank_25_75",
        "ts_setrank_0_100",
    ]

    per_dataset_policy = {t: {p: [] for p in policies} for t in topics}
    for (topic, policy), fps in runs.items():
        if topic not in topics or policy not in policies:
            continue
        qrels_file = in_dir / f"{topic}_ground_truth.json"
        if not qrels_file.exists():
            continue
        qrels = load_qrels(qrels_file)
        for fp in fps:
            hist = load_history(fp)
            scores = []
            for qid, snaps in hist.items():
                if not snaps or qid not in qrels:
                    continue
                last = snaps[-1]["ranking"][:k]
                gains = [qrels[qid].get(str(doc), 0) for doc in last]
                ideal = sorted(qrels[qid].values(), reverse=True)[:k] + [0] * (
                    k - len(qrels[qid])
                )
                idcg = dcg(ideal, k) or 1.0
                scores.append(dcg(gains, k) / idcg)
            if scores:
                per_dataset_policy[topic][policy].append(np.mean(scores))

    print(r"\begin{table}[H]")
    print(r"\centering")
    print(r"\footnotesize")
    print(
        r"\caption{Final reranking performance (nDCG@%d mean $\pm$ std over seeds) by BEIR dataset.}"
        % k
    )
    print(r"\label{tab:beir-datasets}")
    print(r"\begin{tabular}{" + "l" + "c" * len(topics) + "}")
    print(r"\toprule")
    print("Method & " + " & ".join(topics) + r" \\")
    print(r"\midrule")

    for policy in policies:
        row = [policy.replace("_", r"\-")]
        for dataset in topics:
            vals = per_dataset_policy[dataset][policy]
            if vals:
                m, s = np.mean(vals), np.std(vals, ddof=1)
                row.append(f"{m:.3f} $\\pm$ {s:.3f}")
            else:
                row.append("--")
        print(" & ".join(row) + r" \\")

    seed_scores = {policy: defaultdict(list) for policy in policies}

    for (topic, policy), fps in runs.items():
        if policy not in policies:
            continue
        qrels_path = in_dir / f"{topic}_ground_truth.json"
        if not qrels_path.exists():
            continue
        qrels = load_qrels(qrels_path)
        for fp in fps:
            _, _, seed = parse_stem(fp.stem)
            hist = load_history(fp)
            per_q = []
            for qid, snaps in hist.items():
                if not snaps or qid not in qrels:
                    continue
                last = snaps[-1]["ranking"][:k]
                gains = [qrels[qid].get(str(doc), 0) for doc in last]
                ideal = sorted(qrels[qid].values(), reverse=True)[:k]
                ideal += [0] * (k - len(ideal))
                idcg = dcg(ideal, k) or 1.0
                per_q.append(dcg(gains, k) / idcg)
            if per_q:
                seed_scores[policy][seed].append(np.mean(per_q))

    agg = ["All datasets"]
    for policy in policies:
        complete_seed_means = [
            np.mean(vals)
            for seed, vals in seed_scores[policy].items()
            if len(vals) == len(BRIGHT_TOPICS)
        ]
        if complete_seed_means:
            mu = np.mean(complete_seed_means)
            sigma = np.std(complete_seed_means, ddof=1)
            agg.append(f"{mu:.3f} $\\pm$ {sigma:.3f}")
        else:
            agg.append("--")

    print(" & ".join(agg) + r" \\")

    print(r"\bottomrule")
    print(r"\end{tabular}")
    print(r"\end{table}")

    seed_regrets = {policy: defaultdict(list) for policy in policies}
    for (topic, policy), fps in runs.items():
        if policy not in policies:
            continue
        qrels_path = in_dir / f"{topic}_ground_truth.json"
        if not qrels_path.exists():
            continue
        qrels = load_qrels(qrels_path)
        for fp in fps:
            _, _, seed = parse_stem(fp.stem)
            df = aggregate_metrics(load_history(fp), qrels)
            if df.empty:
                continue
            seed_regrets[policy][seed].append(df.cum_reg_mean.iloc[-1])

    print(r"\begin{table}[H]")
    print(r"\centering")
    print(r"\footnotesize")
    print(
        r"\caption{Aggregate cumulative regret (mean $\pm$ std over seeds and datasets).}"
    )
    print(r"\label{tab:agg-regret}")
    print(r"\begin{tabular}{l" + "c" * len(policies) + "}")
    print(r"\toprule")
    header = ["Method"] + [p.replace("_", r"\-") for p in policies]
    print(" & ".join(header) + r" \\")
    print(r"\midrule")

    row = ["All datasets"]
    for policy in policies:
        complete = [
            np.mean(vals)
            for seed, vals in seed_regrets[policy].items()
            if len(vals) == len(BRIGHT_TOPICS)
        ]
        if complete:
            mu = np.mean(complete)
            sigma = np.std(complete, ddof=1)
            row.append(f"{mu:.3f} $\\pm$ {sigma:.3f}")
        else:
            row.append("--")
    print(" & ".join(row) + r" \\")

    print(r"\bottomrule")
    print(r"\end{tabular}")
    print(r"\end{table}")


if __name__ == "__main__":
    main()
