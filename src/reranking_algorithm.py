import math
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from scipy.stats import beta as beta_dist

from src.llms import get_responses
from src.constants import SYSTEM_PROMPT, USER_INSTRUCTIONS, USER_INSTRUCTIONS_CUSTOM


class Reranking:
    def __init__(self, model, system_prompt=None, temperature=0.6, ports=None):
        self.model = model
        self.system_prompt = SYSTEM_PROMPT
        self.instructions = USER_INSTRUCTIONS
        self.temperature = temperature
        self.ports = ports or list(range(5000, 5008))

        self.heap_num_child = 2
        self.k = 10

    def _dcg(self, rels, k=10):
        return sum((2**r - 1) / math.log2(i + 2) for i, r in enumerate(rels[:k]))

    def rank(
        self,
        topic: str,
        query: str,
        docs: list[dict],
        scores: list[float],
        golds: dict,
        iterations: int = 100,
        forced_explore_iterations: int = 20,
        batch_size: int = 10,
        update_interval: int = 1,
        snapshot_interval: int = 10,
    ):
        instructions = USER_INSTRUCTIONS_CUSTOM.get(topic, self.instructions)
        snapshot_interval = snapshot_interval or update_interval
        stats = {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0}

        ids = [d["id"] for d in docs]
        orig_pos = {did: i for i, did in enumerate(ids)}

        if len(scores) == len(ids):
            init_score = dict(zip(ids, scores))
            baseline_ranking = sorted(ids, key=lambda d: init_score[d], reverse=True)
        else:
            baseline_ranking = ids.copy()

        if len(scores) == len(ids):
            mn, mx = min(scores), max(scores)
            eps = 0.5
            beta_params = {
                did: {
                    "alpha": 1.0 + eps * ((s - mn) / (mx - mn) if mx > mn else 0.5),
                    "beta": 1.0
                    + eps * (1 - ((s - mn) / (mx - mn) if mx > mn else 0.5)),
                }
                for did, s in zip(ids, scores)
            }
        else:
            beta_params = {did: {"alpha": 1.0, "beta": 1.0} for did in ids}

        full_rels = sorted(golds.values(), reverse=True)
        ideal_rels = full_rels[: self.k] + [0] * max(0, self.k - len(full_rels))
        idcg = self._dcg(ideal_rels, k=self.k) or 1.0

        history = []
        acc_history = []
        cumulative_regret = 0.0
        last_batch_rels = None
        last_batch_acc = None
        last_phase = None

        def make_snapshot(step, phase: str, force_baseline: bool = False):
            nonlocal cumulative_regret
            if force_baseline:
                ranking = baseline_ranking
            else:
                ranking = sorted(
                    ids,
                    key=lambda d: (
                        -beta_params[d]["alpha"]
                        / (beta_params[d]["alpha"] + beta_params[d]["beta"]),
                        orig_pos[d],
                    ),
                )

            calls = stats["calls"]
            stat_summary = {
                "num_llm_calls": calls,
                "avg_prompt_tokens": stats["prompt_tokens"] / calls if calls else 0.0,
                "avg_completion_tokens": (
                    stats["completion_tokens"] / calls if calls else 0.0
                ),
            }

            rels = [golds.get(d, 0) for d in ranking]
            ndcg = self._dcg(rels, k=self.k) / idcg
            instant = 1.0 - ndcg
            if not force_baseline:
                cumulative_regret += instant

            cum_acc = sum(acc_history) / len(acc_history) if acc_history else None

            return {
                "step": step,
                "phase": phase,
                "ranking": ranking,
                "cumulative_accuracy": cum_acc,
                "batch_relevant_count": last_batch_rels,
                "batch_accuracy": last_batch_acc,
                "stats": stat_summary,
                "regret": {"instant": instant, "cumulative": cumulative_regret},
            }

        history.append(make_snapshot(0, phase="baseline", force_baseline=True))

        last_phase = "explore"
        with ThreadPoolExecutor(max_workers=64) as exe:
            futures = {}
            for _ in range(forced_explore_iterations):
                batch = random.sample(docs, min(batch_size, len(docs)))
                prompt = self._build_prompt(query, batch, instructions)
                futures[
                    exe.submit(
                        get_responses,
                        f"{self.system_prompt}\n\n{prompt}",
                        model=self.model,
                        temperature=self.temperature,
                        port=random.choice(self.ports),
                        n=1,
                    )
                ] = batch

            for fut in tqdm(as_completed(futures), total=len(futures), desc="Forced"):
                resp = fut.result()[0] if fut.result() else {}
                stats["calls"] += 1
                stats["prompt_tokens"] += resp.get("prompt_tokens", 0)
                stats["completion_tokens"] += resp.get("completion_tokens", 0)
                batch = futures[fut]
                picked = self._parse(resp.get("completion", ""), len(batch))

                batch_acc = 0.0
                non_empty = 0
                for idx, d in enumerate(batch, start=1):
                    did = d["id"]
                    picked_lower = [p.lower() for p in picked]
                    y_pred = 1 if f"passage {idx}" in picked_lower else 0
                    beta_params[did]["alpha"] += y_pred
                    beta_params[did]["beta"] += 1 - y_pred

                    y_true = 1 if golds.get(did, 0) > 0 else 0
                    if y_true == 1:
                        non_empty += 1
                        batch_acc += 1 if y_pred == 1 else 0

                last_batch_rels = non_empty
                if non_empty:
                    last_batch_acc = batch_acc / non_empty
                    acc_history.append(last_batch_acc)
                else:
                    last_batch_acc = None

                if stats["calls"] % snapshot_interval == 0:
                    history.append(make_snapshot(stats["calls"], phase=last_phase))

        remaining = iterations - forced_explore_iterations
        groups = math.ceil(remaining / update_interval)
        last_phase = "exploit"
        for g in range(groups):
            with ThreadPoolExecutor(max_workers=64) as exe:
                futures = {}
                cnt = min(update_interval, remaining - g * update_interval)
                for _ in range(cnt):
                    samples = {
                        did: beta_dist.rvs(p["alpha"], p["beta"])
                        for did, p in beta_params.items()
                    }
                    top_ids = sorted(ids, key=lambda d: -samples[d])[:batch_size]
                    batch = [
                        next(doc for doc in docs if doc["id"] == tid) for tid in top_ids
                    ]
                    prompt = self._build_prompt(query, batch, instructions)
                    futures[
                        exe.submit(
                            get_responses,
                            f"{self.system_prompt}\n\n{prompt}",
                            model=self.model,
                            temperature=self.temperature,
                            port=random.choice(self.ports),
                            n=1,
                        )
                    ] = batch

                for fut in tqdm(
                    as_completed(futures),
                    total=len(futures),
                    desc=f"Group {g+1}/{groups}",
                ):
                    resp = fut.result()[0] if fut.result() else {}
                    stats["calls"] += 1
                    stats["prompt_tokens"] += resp.get("prompt_tokens", 0)
                    stats["completion_tokens"] += resp.get("completion_tokens", 0)
                    batch = futures[fut]
                    picked = self._parse(resp.get("completion", ""), len(batch))

                    batch_acc = 0.0
                    non_empty = 0
                    for idx, d in enumerate(batch, start=1):
                        did = d["id"]
                        picked_lower = [p.lower() for p in picked]
                        y_pred = 1 if f"passage {idx}" in picked_lower else 0
                        beta_params[did]["alpha"] += y_pred
                        beta_params[did]["beta"] += 1 - y_pred

                        y_true = 1 if golds.get(did, 0) > 0 else 0
                        if y_true == 1:
                            non_empty += 1
                            batch_acc += 1 if y_pred == 1 else 0

                    last_batch_rels = non_empty
                    if non_empty:
                        last_batch_acc = batch_acc / non_empty
                        acc_history.append(last_batch_acc)
                    else:
                        last_batch_acc = None

                    if stats["calls"] % snapshot_interval == 0:
                        history.append(make_snapshot(stats["calls"], phase=last_phase))

        if stats["calls"] % snapshot_interval != 0:
            history.append(make_snapshot(stats["calls"], phase=last_phase))

        final_snapshot = history[-1]
        return final_snapshot["ranking"], final_snapshot["stats"], history, acc_history

    def rank_heapify(
        self,
        topic: str,
        query: str,
        docs: list[dict],
        golds: dict,
        snapshot_interval: int = 1,
    ):
        instructions = USER_INSTRUCTIONS_CUSTOM.get(topic, self.instructions)
        stats = {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0}
        history = []
        acc_history = []

        canonical = docs.copy()
        orig_pos = {d["id"]: i for i, d in enumerate(canonical)}

        arr = canonical.copy()
        random.shuffle(arr)

        n = len(arr)

        self._heap_golds = golds
        self._heap_acc_history = acc_history

        full_rels = sorted(golds.values(), reverse=True)
        ideal_rels = full_rels[: self.k] + [0] * max(0, self.k - len(full_rels))
        idcg = self._dcg(ideal_rels, k=self.k) or 1.0
        cumulative_regret = 0.0

        def make_snapshot():
            nonlocal cumulative_regret
            calls = stats["calls"]
            ranking = [d["id"] for d in reversed(arr)]
            rels = [golds.get(did, 0) for did in ranking]
            ndcg = self._dcg(rels, k=self.k) / idcg
            instant = 1.0 - ndcg
            cumulative_regret += instant
            cum_acc = sum(acc_history) / len(acc_history) if acc_history else None
            return {
                "step": calls,
                "ranking": ranking,
                "cumulative_accuracy": cum_acc,
                "stats": {
                    "num_llm_calls": calls,
                    "avg_prompt_tokens": (
                        stats["prompt_tokens"] / calls if calls else 0.0
                    ),
                    "avg_completion_tokens": (
                        stats["completion_tokens"] / calls if calls else 0.0
                    ),
                },
                "regret": {"instant": instant, "cumulative": cumulative_regret},
            }

        history.append(make_snapshot())
        self._parallel_build_heap(arr, query, stats, instructions, orig_pos)

        for i in range(n - 1, 0, -1):
            arr[0], arr[i] = arr[i], arr[0]
            self._parallel_heapify_root(arr, i, query, stats, instructions, orig_pos)
            if stats["calls"] % snapshot_interval == 0:
                history.append(make_snapshot())
            if (n - i) >= self.k:
                break

        final_ids = [d["id"] for d in reversed(arr)]
        history.append(make_snapshot())
        return final_ids, history[-1]["stats"], history, acc_history

    def _compare_heap(
        self,
        query: str,
        docs: list[dict],
        stats: dict,
        instructions: str,
        orig_pos: dict,
    ) -> str:
        instr = instructions or self.instructions
        labels = [f"Passage {i+1}" for i in range(len(docs))]
        prompt = f"{instr}\n\nQuestion: {query}\nPassages:\n"
        for label, d in zip(labels, docs):
            prompt += f"Title: {label}\nContent: {d['content']}\n\n"
        resp = get_responses(
            f"{self.system_prompt}\n\n{prompt}",
            model=self.model,
            temperature=self.temperature,
            port=random.choice(self.ports),
            n=1,
        )
        stats["calls"] += 1
        if resp:
            first = resp[0]
            stats["prompt_tokens"] += first.get("prompt_tokens", 0)
            stats["completion_tokens"] += first.get("completion_tokens", 0)
            text = first.get("completion", "")
        else:
            text = ""
        picked = self._parse(text, len(labels))
        picked = [lbl for lbl in picked if lbl in labels]

        def orig_key(label: str) -> int:
            num = int(label.split()[1]) - 1
            doc = docs[num]
            return orig_pos[doc["id"]]

        if picked:
            chosen = min(picked, key=orig_key)
        else:
            chosen = min(labels, key=orig_key)

        idx = labels.index(chosen)
        did = docs[idx]["id"]
        y_true = 1 if self._heap_golds.get(did, 0) > 0 else 0
        batch_acc = float(y_true)
        self._heap_acc_history.append(batch_acc)

        return chosen

    def _parallel_build_heap(
        self,
        arr: list[dict],
        query: str,
        stats: dict,
        instructions: str,
        orig_pos: dict,
    ):

        n = len(arr)
        internal = [i for i in range(n) if self.heap_num_child * i + 1 < n]
        by_depth = {}
        for i in internal:
            d = self._compute_depth(i)
            by_depth.setdefault(d, []).append(i)
        for depth in sorted(by_depth.keys(), reverse=True):
            idxs = by_depth[depth]
            with ThreadPoolExecutor(max_workers=len(idxs)) as exe:
                futures = {}
                for i in idxs:
                    start = self.heap_num_child * i + 1
                    end = min(self.heap_num_child * (i + 1) + 1, n)
                    group = [arr[i]] + arr[start:end]
                    futures[
                        exe.submit(
                            self._compare_heap,
                            query,
                            group,
                            stats,
                            instructions,
                            orig_pos,
                        )
                    ] = (i, list(range(start, end)))
                for fut in as_completed(futures):
                    i, children = futures[fut]
                    label = fut.result()
                    try:
                        num = int(label.split()[1])
                        idx = num - 1
                    except:
                        idx = 0
                    if 0 < idx <= len(children):
                        swap = children[idx - 1]
                        arr[i], arr[swap] = arr[swap], arr[i]

    def _parallel_heapify_root(
        self,
        arr: list[dict],
        n: int,
        query: str,
        stats: dict,
        instructions: str,
        orig_pos: dict,
    ):
        i = 0
        while self.heap_num_child * i + 1 < n:
            start = self.heap_num_child * i + 1
            end = min(self.heap_num_child * (i + 1) + 1, n)
            group = [arr[i]] + arr[start:end]
            label = self._compare_heap(query, group, stats, instructions, orig_pos)
            try:
                num = int(label.split()[1])
                idx = num - 1
            except:
                idx = 0
            swap_to = i if idx == 0 else start + idx - 1
            if swap_to == i:
                break
            arr[i], arr[swap_to] = arr[swap_to], arr[i]
            i = swap_to

    def _build_prompt(self, query, batch, instructions=None):
        instr = instructions or self.instructions
        prompt = f"{instr}\n\nQuestion: {query}\nPassages:\n"
        for i, d in enumerate(batch, start=1):
            prompt += f"Title: Passage {i}\nContent: {d['content']}\n\n"
        return prompt

    def _parse(self, text, batch_size=None):
        if "<answer>" not in text:
            return []
        blk = text.split("<answer>", 1)[1]
        if "</answer>" in blk:
            blk = blk.split("</answer>", 1)[0]
        blk = blk.strip()

        prefix = "Relevant passages:"
        if blk.lower().startswith(prefix.lower()):
            blk = blk[len(prefix) :].strip()
        if not blk or blk.lower().startswith("no relevant passages"):
            return []

        import re

        nums = re.findall(r"passage\s*(\d+)", blk, flags=re.IGNORECASE)

        if batch_size is not None:
            nums = [n for n in nums if 1 <= int(n) <= batch_size]

        return [f"Passage {n}" for n in nums]

    def _compute_depth(self, index: int) -> int:
        d = self.heap_num_child
        return int(math.floor(math.log((d - 1) * index + 1, d)))
