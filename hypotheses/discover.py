"""Generate, refine, match and select hypotheses with 30 group + 50 pair rounds."""

import argparse
import hashlib
import json
import logging
import os
import time
from collections import Counter
from pathlib import Path

import numpy as np

from . import prompts

LOG = logging.getLogger(__name__)


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    )
    temp.replace(path)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read_items(items_path, difficulty_path):
    records = [
        json.loads(line)
        for line in Path(items_path).read_text().splitlines()
        if line.strip()
    ]
    payload = json.loads(Path(difficulty_path).read_text())
    parameters = payload["items"] if isinstance(payload, dict) else payload
    difficulty = {str(r["item_id"]): float(r["difficulty"]) for r in parameters}
    require(len(difficulty) == len(parameters), "Duplicate difficulty item IDs.")
    seen, items = set(), []
    for row in records:
        iid = str(row["item_id"])
        require(iid not in seen, f"Duplicate item ID: {iid}")
        require(
            iid in difficulty and np.isfinite(difficulty[iid]),
            f"Missing/invalid difficulty: {iid}",
        )
        require(
            isinstance(row.get("question"), str) and row["question"].strip(),
            f"Missing question: {iid}",
        )
        seen.add(iid)
        text = row["question"]
        for key in ["answer", "explanation"]:
            if row.get(key) is not None:
                text += f"\n{key.title()}: {row[key]}"
        items.append({"item_id": iid, "text": text, "difficulty": difficulty[iid]})
    require(bool(items), "No items supplied.")
    return items


def make_splits(items, v_size, t_size, seed, supplied=None):
    ids = [r["item_id"] for r in items]
    if supplied is not None:
        require(
            set(supplied) == {"G", "V", "T"}, "Splits must have exactly G, V, T keys."
        )
        result = {k: list(map(str, supplied[k])) for k in ["G", "V", "T"]}
    else:
        require(
            v_size > 0 and t_size > 0 and v_size + t_size < len(items),
            "Invalid split sizes.",
        )
        rng = np.random.default_rng(seed)
        ordered = np.argsort([r["difficulty"] for r in items], kind="stable")
        buckets = [rng.permutation(a).tolist() for a in np.array_split(ordered, 10)]
        remaining = list(range(len(items)))

        def draw(count):
            # Proportional allocation with largest remainders; exact requested size.
            quotas = (
                np.array([len(b) for b in buckets]) * count / sum(map(len, buckets))
            )
            sizes = np.floor(quotas).astype(int)
            for i in np.argsort(-(quotas - sizes), kind="stable")[
                : count - sizes.sum()
            ]:
                sizes[i] += 1
            chosen = []
            for b, n in zip(buckets, sizes):
                chosen.extend(b[:n])
                del b[:n]
            return chosen

        t = draw(t_size)
        v = draw(v_size)
        used = set(t + v)
        result = {
            "G": [ids[i] for i in remaining if i not in used],
            "V": [ids[i] for i in sorted(v)],
            "T": [ids[i] for i in sorted(t)],
        }
    joined = sum(result.values(), [])
    require(
        len(joined) == len(set(joined)) and set(joined) == set(ids),
        "Splits must be disjoint and cover exactly the supplied items.",
    )
    require(all(result.values()), "All three splits must be nonempty.")
    return result


def sample_plan(items, splits, group_rounds=30, pair_rounds=50, min_gap=4.0, seed=0):
    require(
        group_rounds > 0 and pair_rounds > 0 and min_gap > 0,
        "Invalid sampling configuration.",
    )
    by_id = {r["item_id"]: r for r in items}
    ids = splits["G"]
    y = np.array([by_id[i]["difficulty"] for i in ids])
    order = np.argsort(y, kind="stable")
    buckets = np.empty(len(ids), dtype=int)
    buckets[order] = np.arange(len(ids)) * 10 // len(ids)
    require(
        min(np.bincount(buckets, minlength=10)) >= 20,
        "Group sampling needs at least 20 G items in each of ten difficulty buckets.",
    )
    rng = np.random.default_rng(seed)
    combos = [(lo, hi) for lo in range(10) for hi in range(lo + 3, 10)]
    cc, ic = Counter(), Counter()
    plan = []
    for rnd in range(group_rounds):
        w = np.array([(hi - lo) ** 2 / (1 + cc[(lo, hi)]) for lo, hi in combos])
        lo, hi = combos[rng.choice(len(combos), p=w / w.sum())]
        selected = []
        for bucket in [hi, lo]:
            pos = np.flatnonzero(buckets == bucket)
            w = np.array([1 / (1 + ic[int(i)]) for i in pos])
            selected.append(rng.choice(pos, 20, replace=False, p=w / w.sum()).tolist())
        cc[(lo, hi)] += 1
        ic.update(sum(selected, []))
        plan.append(
            {
                "kind": "group",
                "round": rnd + 1,
                "hard": [ids[i] for i in selected[0]],
                "easy": [ids[i] for i in selected[1]],
            }
        )
    rng = np.random.default_rng(seed)
    left, right = np.triu_indices(len(ids), 1)
    keep = np.abs(y[left] - y[right]) >= min_gap
    left, right = left[keep], right[keep]
    h = np.where(y[left] > y[right], left, right)
    e = np.where(y[left] > y[right], right, left)
    require(len(h) > 0, "No pairs meet min-gap; lower --min-gap explicitly.")
    ic = np.zeros(len(ids))
    pc = np.zeros(len(h))
    for rnd in range(pair_rounds):
        weights = (1 + ic[h]) ** -2 * (1 + ic[e]) ** -2 / (1 + pc)
        used, picked = set(), []
        for _ in range(5):
            eligible = np.array(
                [i for i in range(len(h)) if h[i] not in used and e[i] not in used]
            )
            require(
                len(eligible) > 0,
                "Cannot sample five disjoint pairs; lower --min-gap or enlarge G.",
            )
            w = weights[eligible]
            i = int(rng.choice(eligible, p=w / w.sum()))
            picked.append(i)
            used.update([h[i], e[i]])
        for i in picked:
            ic[h[i]] += 1
            ic[e[i]] += 1
            pc[i] += 1
        plan.append(
            {
                "kind": "pair",
                "round": rnd + 1,
                "pairs": [[ids[h[i]], ids[e[i]]] for i in picked],
            }
        )
    return plan


class CachedClient:
    """Content-addressed validated responses; completed calls are reused on restart."""

    def __init__(self, cache, thinking="low", retries=3, transport=None):
        self.cache = Path(cache)
        self.thinking = thinking
        self.retries = retries
        self.transport = transport

    def ask(self, model, system, user, validate):
        request = {
            "model": model,
            "system": system,
            "user": user,
            "thinking": self.thinking,
            "temperature": 0,
            "max_output_tokens": 16384,
        }
        key = hashlib.sha256(json.dumps(request, sort_keys=True).encode()).hexdigest()
        path = self.cache / (key + ".json")
        if path.exists():
            response = json.loads(path.read_text())
            validate(response)
            return response
        feedback = ""
        for attempt in range(self.retries):
            try:
                if self.transport:
                    response = self.transport(model, system, user + feedback)
                else:
                    from google import genai
                    from google.genai import types

                    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
                    try:
                        result = client.models.generate_content(
                            model=model,
                            contents=user + feedback,
                            config=types.GenerateContentConfig(
                                system_instruction=system,
                                temperature=0,
                                response_mime_type="application/json",
                                max_output_tokens=16384,
                                thinking_config=types.ThinkingConfig(
                                    thinking_level=self.thinking
                                ),
                            ),
                        )
                        response = json.loads(result.text)
                    finally:
                        client.close()
                validate(response)
                save(path, response)
                return response
            except Exception as exc:
                if attempt + 1 == self.retries:
                    raise RuntimeError(
                        f"Request failed after {self.retries} attempts; cache retained."
                    ) from exc
                LOG.warning(
                    "Retrying failed %s request (%s)", model, type(exc).__name__
                )
                feedback = "\nPrevious response failed validation. Return complete valid JSON matching the schema."
                time.sleep(min(2**attempt, 8))


def generate(items, plan, client, model, min_gap, context=""):
    by_id = {r["item_id"]: r for r in items}
    raw = []

    def show(i):
        r = by_id[i]
        return f"[IRT difficulty: {r['difficulty']:+.3f}] {r['text']}"

    for batch in plan:
        kind = batch["kind"]
        if kind == "group":
            user = "HARDER group:\n" + "\n\n".join(show(i) for i in batch["hard"])
            user += "\n\nEASIER group:\n" + "\n\n".join(show(i) for i in batch["easy"])
            system = prompts.GROUP
        else:
            user = f"Minimum IRT gap: {min_gap}\n" + "\n\n".join(
                f"Pair {n + 1}\nHARDER: {show(h)}\nEASIER: {show(e)}"
                for n, (h, e) in enumerate(batch["pairs"])
            )
            system = prompts.PAIR

        def validate(r):
            require(
                isinstance(r, dict) and isinstance(r.get("hypotheses"), list),
                "Expected hypotheses list.",
            )
            if kind == "pair":
                require(len(r["hypotheses"]) <= 10, "At most ten pair hypotheses.")
            for row in r["hypotheses"]:
                text = row if kind == "group" else row.get("hypothesis")
                require(
                    isinstance(text, str) and bool(text.strip()), "Empty hypothesis."
                )
                if kind == "pair":
                    require(
                        isinstance(row.get("pairs"), list)
                        and bool(row["pairs"])
                        and all(type(i) is int and 1 <= i <= 5 for i in row["pairs"]),
                        "Invalid pair references.",
                    )

        response = client.ask(
            model, system + "\nDataset context: " + context, user, validate
        )
        raw.extend(
            {
                "text": r.strip() if isinstance(r, str) else r["hypothesis"].strip(),
                "kind": kind,
                "round": batch["round"],
            }
            for r in response["hypotheses"]
        )
        LOG.info(
            "%s round %d: %d candidates",
            kind,
            batch["round"],
            len(response["hypotheses"]),
        )
    require(bool(raw), "No hypotheses generated.")
    return raw


def validate_partition(response, members):
    require(
        isinstance(response, dict) and isinstance(response.get("classes"), list),
        "Missing classes.",
    )
    seen = []
    for row in response["classes"]:
        ids = row.get("member_ids")
        require(isinstance(ids, list) and bool(ids), "Empty class.")
        require(all(i in members for i in ids), "Unknown member ID.")
        seen.extend(ids)
        if row.get("resolution") == "KEEP_ORIGINAL":
            require(
                row.get("representative_id") in ids
                and row.get("replacement_text") is None,
                "Invalid original.",
            )
        else:
            require(
                row.get("resolution") == "REWRITE"
                and len(ids) > 1
                and row.get("representative_id") is None
                and isinstance(row.get("replacement_text"), str)
                and bool(row["replacement_text"].strip()),
                "Invalid rewrite; singletons must retain original.",
            )
    require(
        len(seen) == len(set(seen)) and set(seen) == set(members),
        "Partition must cover every ID exactly once.",
    )


def refine(
    raw, client, model, embedding_model, threshold=0.85, encode=None, strict=False
):
    texts = list(dict.fromkeys(r["text"] for r in raw))
    ids = [f"H{i}" for i in range(len(texts))]
    if len(texts) == 1:
        return texts, []
    if encode is None:
        from sentence_transformers import SentenceTransformer

        encoder = SentenceTransformer(embedding_model)

        def encode(values):
            return encoder.encode(
                values, normalize_embeddings=True, show_progress_bar=False
            )

    vectors = np.asarray(encode(texts), dtype=float)
    require(
        vectors.ndim == 2 and len(vectors) == len(texts) and np.isfinite(vectors).all(),
        "Invalid embeddings.",
    )
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    require((norms > 0).all(), "Zero embedding.")
    vectors = vectors / norms
    a, b = np.where(np.triu(vectors @ vectors.T > threshold, k=1))
    pairs = [
        {
            "pair_id": f"P{k}",
            "a": {"id": ids[i], "text": texts[i]},
            "b": {"id": ids[j], "text": texts[j]},
        }
        for k, (i, j) in enumerate(zip(a, b))
    ]
    decisions = []
    parent = {i: i for i in ids}

    def root(i):
        while parent[i] != i:
            i = parent[i]
        return i

    for start in range(0, len(pairs), 20):
        chunk = pairs[start : start + 20]
        expected = {p["pair_id"] for p in chunk}

        def validate(r):
            require(
                isinstance(r, dict) and isinstance(r.get("decisions"), list),
                "Missing decisions.",
            )
            rows = r["decisions"]
            require(
                len(rows) == len(expected) and {x["pair_id"] for x in rows} == expected,
                "Pair coverage.",
            )
            for row in rows:
                rel = row["relationship"]
                decision = row["decision"]
                require(
                    rel
                    in [
                        "EXACT_EQUIVALENT",
                        "SAME_CORE_MINOR_MODIFIER",
                        "NESTED_OR_SPECIAL_CASE",
                        "RELATED_DISTINCT",
                    ],
                    "Relation.",
                )
                require(decision in ["KEEP_BOTH", "KEEP_REPRESENTATIVE"], "Decision.")
                require(
                    strict
                    or rel not in ["EXACT_EQUIVALENT", "SAME_CORE_MINOR_MODIFIER"]
                    or decision == "KEEP_REPRESENTATIVE",
                    "Equivalent decision.",
                )
                require(
                    rel != "RELATED_DISTINCT" or decision == "KEEP_BOTH",
                    "Distinct decision.",
                )

                if strict:
                    require(
                        (rel == "EXACT_EQUIVALENT")
                        == (decision == "KEEP_REPRESENTATIVE"),
                        "Strict deduplication merges only exact equivalents.",
                    )

        response = client.ask(
            model,
            prompts.RELATION + (prompts.STRICT if strict else ""),
            json.dumps(chunk),
            validate,
        )
        decisions.extend(response["decisions"])
    pair_by_id = {p["pair_id"]: p for p in pairs}
    for d in decisions:
        if d["decision"] == "KEEP_REPRESENTATIVE":
            p = pair_by_id[d["pair_id"]]
            parent[root(p["b"]["id"])] = root(p["a"]["id"])
    components = {}
    for i in ids:
        components.setdefault(root(i), []).append(i)
    by_id = dict(zip(ids, texts))
    final = []
    audit = []
    for members in components.values():
        if len(members) == 1:
            final.append(by_id[members[0]])
            continue
        internal = [
            d
            for d in decisions
            if pair_by_id[d["pair_id"]]["a"]["id"] in members
            and pair_by_id[d["pair_id"]]["b"]["id"] in members
        ]
        payload = {
            "hypotheses": {i: by_id[i] for i in members},
            "pair_evidence": [
                dict(
                    d,
                    a=pair_by_id[d["pair_id"]]["a"]["id"],
                    b=pair_by_id[d["pair_id"]]["b"]["id"],
                )
                for d in internal
            ],
        }

        def check_partition(response):
            validate_partition(response, members)
            if strict:
                require(
                    all(
                        c["resolution"] == "KEEP_ORIGINAL" for c in response["classes"]
                    ),
                    "Strict deduplication must retain originals.",
                )

        response = client.ask(
            model,
            prompts.REFINE + (prompts.STRICT if strict else ""),
            json.dumps(payload),
            check_partition,
        )
        audit.append(response)
        for row in response["classes"]:
            final.append(
                by_id[row["representative_id"]]
                if row["resolution"] == "KEEP_ORIGINAL"
                else row["replacement_text"].strip()
            )
    return list(dict.fromkeys(final)), audit


def match(items, hypotheses, client, model):
    matrix = []
    for n, item in enumerate(items):
        scores = []
        for start in range(0, len(hypotheses), 40):
            chunk = hypotheses[start : start + 40]

            def validate(r):
                require(
                    isinstance(r, dict)
                    and isinstance(r.get("scores"), list)
                    and len(r["scores"]) == len(chunk)
                    and all(type(s) is int and s in (0, 1) for s in r["scores"]),
                    "Expected binary scores in order.",
                )

            system = (
                prompts.MATCH
                + "\nProperties:\n"
                + json.dumps(chunk, ensure_ascii=False)
            )
            scores.extend(
                client.ask(model, system, "Question:\n" + item["text"], validate)[
                    "scores"
                ]
            )
        matrix.append(scores)
        if (n + 1) % 25 == 0:
            LOG.info("Matched %d/%d items", n + 1, len(items))
    return np.asarray(matrix, dtype=int)


def select(matrix, items, splits, top_k=20, repeats=100, seed=0):
    from sklearn.linear_model import Lasso
    from sklearn.model_selection import GridSearchCV, KFold
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    require(top_k > 0 and repeats > 0, "top-k and repeats must be positive.")
    positions = {r["item_id"]: i for i, r in enumerate(items)}
    rows = [positions[i] for i in splits["V"]]
    X = np.asarray(matrix)[rows]
    y = np.array([items[i]["difficulty"] for i in rows])
    size = int(len(rows) * 0.75)
    require(size >= 10, "Selection requires at least 14 V items.")
    frequency = np.zeros(X.shape[1])
    magnitude = np.zeros(X.shape[1])
    alphas = []
    rng = np.random.default_rng(seed)
    for repeat in range(repeats):
        ix = rng.choice(len(rows), size, replace=False)
        # Scaling is fitted inside each CV fold, not before cross-validation.
        search = GridSearchCV(
            make_pipeline(StandardScaler(), Lasso(max_iter=100000)),
            {"lasso__alpha": np.logspace(-4, 0, 25)},
            cv=KFold(5, shuffle=True, random_state=seed + repeat),
            scoring="neg_mean_squared_error",
            error_score="raise",
        )
        search.fit(X[ix], y[ix])
        coef = search.best_estimator_.named_steps["lasso"].coef_
        frequency += np.abs(coef) > 1e-10
        magnitude += np.abs(coef)
        alphas.append(float(search.best_params_["lasso__alpha"]))
    frequency /= repeats
    magnitude /= repeats
    order = sorted(
        range(len(frequency)), key=lambda i: (-frequency[i], -magnitude[i], i)
    )
    chosen = [i for i in order if frequency[i] > 0][:top_k]
    return {
        "fit_split": "V",
        "repeats": repeats,
        "sample_fraction": 0.75,
        "cv_folds": 5,
        "selected_indices": chosen,
        "ranking": order,
        "frequency": frequency.tolist(),
        "mean_abs_coefficient": magnitude.tolist(),
        "chosen_alphas": alphas,
    }


def run(items, splits, plan, client, args, encode=None):
    raw = generate(items, plan, client, args.propose_model, args.min_gap, args.context)
    deduplicated, strict_audit = refine(
        raw,
        client,
        args.propose_model,
        args.embedding_model,
        args.threshold,
        encode,
        strict=True,
    )
    hypotheses, audit = refine(
        [{"text": h} for h in deduplicated],
        client,
        args.propose_model,
        args.embedding_model,
        args.threshold,
        encode,
    )
    matrix = match(items, hypotheses, client, args.match_model)
    selection = select(matrix, items, splits, args.top_k, args.repeats, args.seed)
    chosen = set(selection["selected_indices"])
    ranks = {i: n + 1 for n, i in enumerate(selection["ranking"])}
    result = {
        "schema_version": 1,
        "config": {
            k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()
        },
        "splits": splits,
        "counts": {
            "raw_candidates": len(raw),
            "distinct_candidates": len({r["text"] for r in raw}),
            "refined": len(hypotheses),
            "selected": len(chosen),
        },
        "hypotheses": [
            {
                "hypothesis_id": f"H{i + 1:03d}",
                "text": h,
                "selected": i in chosen,
                "rank": ranks[i],
                "selection_frequency": selection["frequency"][i],
            }
            for i, h in enumerate(hypotheses)
        ],
        "matches": {
            "item_ids": [r["item_id"] for r in items],
            "hypothesis_ids": [f"H{i + 1:03d}" for i in range(len(hypotheses))],
            "values": matrix.tolist(),
        },
        "selection": selection,
    }
    if args.debug:
        result["debug"] = {
            "raw_candidates": raw,
            "refinement": {"strict": strict_audit, "semantic_family": audit},
            "sampling_plan": plan,
        }
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--items", type=Path, required=True)
    parser.add_argument("--difficulty", type=Path, required=True)
    parser.add_argument("--splits", type=Path)
    parser.add_argument("--v-size", type=int, default=400)
    parser.add_argument("--t-size", type=int, default=200)
    parser.add_argument("--group-rounds", type=int, default=30)
    parser.add_argument("--pair-rounds", type=int, default=50)
    parser.add_argument("--min-gap", type=float, default=4.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--propose-model", default="gemini-3.1-pro-preview")
    parser.add_argument("--match-model", default="gemini-3.7-flash")
    parser.add_argument("--embedding-model", default="Qwen/Qwen3-Embedding-4B")
    parser.add_argument("--threshold", type=float, default=0.85)
    parser.add_argument("--context", default="")
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--repeats", type=int, default=100)
    parser.add_argument("--cache", type=Path, default=Path(".cache/hypotheses"))
    parser.add_argument(
        "--output", type=Path, default=Path("output/hypotheses_results.json")
    )
    parser.add_argument("--debug", action="store_true")
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Required for paid model calls; otherwise dry run.",
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    require(
        0 <= args.threshold <= 1 and args.top_k > 0 and args.repeats > 0,
        "Invalid threshold/top-k/repeats.",
    )
    items = read_items(args.items, args.difficulty)
    splits = make_splits(
        items,
        args.v_size,
        args.t_size,
        args.seed,
        json.loads(args.splits.read_text()) if args.splits else None,
    )
    plan = sample_plan(
        items, splits, args.group_rounds, args.pair_rounds, args.min_gap, args.seed
    )
    require(len(splits["V"]) >= 14, "Selection requires at least 14 V items.")
    fingerprint = hashlib.sha256(
        json.dumps(
            {"items": items, "splits": splits, "plan": plan}, sort_keys=True
        ).encode()
    ).hexdigest()
    print(
        json.dumps(
            {
                "splits": {k: len(v) for k, v in splits.items()},
                "generation_calls": len(plan),
                "item_presentations": args.group_rounds * 40 + args.pair_rounds * 10,
                "note": "Refinement and matching calls depend on generated hypothesis count; retries may add cost.",
            },
            indent=2,
        )
    )
    if not args.execute:
        return
    require(not args.output.exists(), "Output exists; choose a new output path.")
    require(
        bool(os.environ.get("GEMINI_API_KEY")), "Set GEMINI_API_KEY before execution."
    )
    result = run(items, splits, plan, CachedClient(args.cache), args)
    result["input_plan_sha256"] = fingerprint
    save(args.output, result)
    print(args.output)


if __name__ == "__main__":
    main()
