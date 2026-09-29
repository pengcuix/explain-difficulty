import json
from types import SimpleNamespace

import numpy as np
import pytest

from hypotheses import prompts
from hypotheses.discover import (
    CachedClient,
    make_splits,
    match,
    read_items,
    refine,
    run,
    sample_plan,
    select,
    validate_partition,
)
from irt.fit import fit, validate_matrix


def items(n=260):
    return [
        {
            "item_id": str(i),
            "text": f"Question {i}",
            "difficulty": float(i % 20) / 2 - 5,
        }
        for i in range(n)
    ]


def test_irt_input():
    _, mask, ids, models = validate_matrix([[0, np.nan], [1, 0], [-1, 1]])
    assert mask.sum() == 4 and ids == ["0", "1", "2"] and models == ["0", "1"]
    for bad in [[[1, 0.5], [0, 1]], [[1, -1], [0, -1]], [[1, np.inf], [0, 1]]]:
        with pytest.raises(ValueError):
            validate_matrix(bad)
    with pytest.raises(ValueError):
        validate_matrix([[1, 0], [0, 1]], ["a", "a"])


def test_irt_fit():
    pytest.importorskip("py_irt")
    x = np.random.default_rng(0).integers(0, 2, size=(20, 8)).astype(float)
    x[0, 0] = np.nan
    result = fit(x, epochs=3)
    assert result["n_observed"] == 159 and len(result["items"]) == 20
    assert all(
        np.isfinite(r["difficulty"]) and r["difficulty_sd"] > 0 for r in result["items"]
    )
    assert len(result["models"]) == 8


def test_splits_plan():
    data = items()
    s = make_splits(data, 30, 30, 0)
    plan = sample_plan(data, s, 30, 50, 4, 0)
    assert len(plan) == 80 and plan == sample_plan(data, s, 30, 50, 4, 0)
    by_id = {r["item_id"]: r for r in data}
    for b in plan:
        if b["kind"] == "group":
            draw = b["hard"] + b["easy"]
            assert len(draw) == 40
        else:
            draw = sum(b["pairs"], [])
            assert len(draw) == 10
            assert all(
                by_id[h]["difficulty"] - by_id[e]["difficulty"] >= 4
                for h, e in b["pairs"]
            )
        assert len(draw) == len(set(draw)) and set(draw) <= set(s["G"])
    with pytest.raises(ValueError):
        make_splits(data, 30, 30, 0, {"G": ["0"], "V": ["0"], "T": ["1"]})
    with pytest.raises(ValueError):
        sample_plan(data, s, 1, 1, 100, 0)


def test_read_join(tmp_path):
    a = tmp_path / "items.jsonl"
    b = tmp_path / "d.json"
    a.write_text(json.dumps({"item_id": "2", "question": "Q", "answer": "A"}) + "\n")
    b.write_text(json.dumps({"items": [{"item_id": "2", "difficulty": 1.5}]}))
    assert read_items(a, b)[0]["text"] == "Q\nAnswer: A"


def test_partition_validation():
    with pytest.raises(ValueError):
        validate_partition({"classes": []}, ["H0"])
    with pytest.raises(ValueError):
        validate_partition(
            {
                "classes": [
                    {
                        "member_ids": ["H0"],
                        "resolution": "REWRITE",
                        "replacement_text": "New",
                        "representative_id": None,
                    }
                ]
            },
            ["H0"],
        )


def test_cached_validation(tmp_path):
    calls = []

    def transport(*args):
        calls.append(args)
        return {"scores": [1]}

    c = CachedClient(tmp_path, transport=transport)

    def check(r):
        assert r == {"scores": [1]}

    assert c.ask("m", "s", "u", check) == c.ask("m", "s", "u", check)
    assert len(calls) == 1


def test_pipeline(tmp_path):
    data = items()
    splits = make_splits(data, 30, 30, 0)
    plan = sample_plan(data, splits, 1, 1, 4, 0)
    calls = []

    def transport(model, system, user):
        calls.append(system)
        if system == prompts.RELATION + prompts.STRICT:
            return {
                "decisions": [
                    {
                        "pair_id": p["pair_id"],
                        "relationship": "SAME_CORE_MINOR_MODIFIER",
                        "decision": "KEEP_BOTH",
                    }
                    for p in json.loads(user)
                ]
            }
        if system.startswith(prompts.GROUP):
            return {"hypotheses": ["Property A", "Property A", "Property B"]}
        if system.startswith(prompts.PAIR):
            return {"hypotheses": [{"hypothesis": "Property C", "pairs": [1]}]}
        if system == prompts.RELATION:
            return {
                "decisions": [
                    {
                        "pair_id": p["pair_id"],
                        "relationship": "SAME_CORE_MINOR_MODIFIER",
                        "decision": "KEEP_REPRESENTATIVE",
                        "reason": "Equivalent",
                    }
                    for p in json.loads(user)
                ]
            }
        if system == prompts.REFINE:
            ids = list(json.loads(user)["hypotheses"])
            return {
                "classes": [
                    {
                        "member_ids": ids,
                        "resolution": "REWRITE",
                        "representative_id": None,
                        "replacement_text": "Shared property",
                    }
                ]
            }
        assert system.startswith(prompts.MATCH)
        i = int(user.split()[-1])
        return {"scores": [int(i % 20 >= 10)]}

    args = SimpleNamespace(
        propose_model="mock",
        match_model="mock",
        embedding_model="mock",
        threshold=0.85,
        min_gap=4,
        context="",
        top_k=1,
        repeats=2,
        seed=0,
        debug=False,
    )
    client = CachedClient(tmp_path, transport=transport)
    result = run(
        data, splits, plan, client, args, encode=lambda t: np.ones((len(t), 2))
    )
    assert result["counts"] == {
        "raw_candidates": 4,
        "distinct_candidates": 3,
        "refined": 1,
        "selected": 1,
    }
    assert len(result["matches"]["values"]) == 260
    assert result["selection"]["fit_split"] == "V" and "debug" not in result
    n = len(calls)
    assert (
        run(data, splits, plan, client, args, encode=lambda t: np.ones((len(t), 2)))
        == result
    )
    assert len(calls) == n
    # Neither generation nor selection can use T labels/targets.
    changed = [dict(r) for r in data]
    for r in changed:
        if r["item_id"] in splits["T"]:
            r["difficulty"] = 9999
    assert (
        select(result["matches"]["values"], changed, splits, 1, 2, 0)
        == result["selection"]
    )


def test_component_does_not_force_merge(tmp_path):
    def transport(model, system, user):
        if system == prompts.RELATION:
            return {
                "decisions": [
                    {
                        "pair_id": p["pair_id"],
                        "relationship": "EXACT_EQUIVALENT",
                        "decision": "KEEP_REPRESENTATIVE",
                    }
                    for p in json.loads(user)
                ]
            }
        members = list(json.loads(user)["hypotheses"])
        return {
            "classes": [
                {
                    "member_ids": [m],
                    "resolution": "KEEP_ORIGINAL",
                    "representative_id": m,
                    "replacement_text": None,
                }
                for m in members
            ]
        }

    final, _ = refine(
        [{"text": "A"}, {"text": "B"}],
        CachedClient(tmp_path, transport=transport),
        "mock",
        "mock",
        encode=lambda t: np.ones((len(t), 2)),
    )
    assert final == ["A", "B"]


def test_nonbinary_match_rejected(tmp_path):
    client = CachedClient(tmp_path, retries=1, transport=lambda *a: {"scores": [2]})
    with pytest.raises(RuntimeError):
        match(items(1), ["A"], client, "mock")
    assert not list(tmp_path.glob("*.json"))
