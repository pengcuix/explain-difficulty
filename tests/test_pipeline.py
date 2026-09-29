import csv
import hashlib
import json
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from hypotheses import discover, prompts
from hypotheses.discover import (
    CachedClient,
    apply_selection,
    main,
    make_splits,
    match,
    read_items,
    refine,
    run,
    sample_plan,
    save,
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


def test_pipeline(tmp_path, monkeypatch):
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
        min_gap=4.0,
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
    }
    assert len(result["matches"]["values"]) == 260
    assert "selection" not in result and "debug" not in result
    n = len(calls)
    assert (
        run(data, splits, plan, client, args, encode=lambda t: np.ones((len(t), 2)))
        == result
    )
    assert len(calls) == n
    selected = apply_selection(result, top_k=1, repeats=2)
    assert selected["counts"]["selected"] == 1
    assert selected["selection"]["fit_split"] == "V"
    assert "selection" not in result
    # Neither generation nor selection can use T labels/targets.
    changed = [dict(r) for r in data]
    for r in changed:
        if r["item_id"] in splits["T"]:
            r["difficulty"] = 9999
    assert (
        select(result["matches"]["values"], changed, splits, 1, 2, 0)
        == selected["selection"]
    )
    # Exercise the paid-call entry point using only the same cached mock client.
    prepared, output = tmp_path / "prepared.json", tmp_path / "generated.json"
    save(prepared, {"items": data, "splits": splits})
    monkeypatch.setenv("GEMINI_API_KEY", "unused-test-placeholder")
    monkeypatch.setattr(discover, "CachedClient", lambda cache: client)
    monkeypatch.setattr(
        discover, "run", lambda *a: run(*a, encode=lambda t: np.ones((len(t), 2)))
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "discover",
            "generate",
            "--input",
            str(prepared),
            "--group-rounds",
            "1",
            "--pair-rounds",
            "1",
            "--propose-model",
            "mock",
            "--match-model",
            "mock",
            "--embedding-model",
            "mock",
            "--output",
            str(output),
            "--execute",
        ],
    )
    main()
    generated = json.loads(output.read_text())
    assert generated["counts"] == result["counts"]
    assert generated["matches"] == result["matches"]
    assert "selection" not in generated and len(calls) == n


@pytest.mark.parametrize("extension", ["json", "jsonl", "csv"])
def test_question_formats_and_order(tmp_path, extension):
    rows = [
        {
            "item_id": "001",
            "question": "Q, first?",
            "answer": "A",
            "explanation": "First\nsecond",
        },
        {"item_id": "002", "question": "Second?", "answer": "", "explanation": ""},
    ]
    path = tmp_path / f"items.{extension}"
    if extension == "json":
        save(path, rows[::-1])
    elif extension == "jsonl":
        path.write_text("\n".join(json.dumps(r) for r in rows[::-1]))
    else:
        with path.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows[::-1])
    difficulty = tmp_path / "irt.json"
    save(
        difficulty,
        {
            "items": [
                {"item_id": r["item_id"], "difficulty": i} for i, r in enumerate(rows)
            ]
        },
    )
    result = read_items(path, difficulty)
    assert [r["item_id"] for r in result] == ["001", "002"]
    assert result[0]["text"] == "Q, first?\nAnswer: A\nExplanation: First\nsecond"
    assert result[1]["text"] == "Second?"


def test_items_must_cover_irt(tmp_path):
    path, difficulty = tmp_path / "items.json", tmp_path / "irt.json"
    save(
        difficulty,
        [{"item_id": "0", "difficulty": 1}, {"item_id": "1", "difficulty": 2}],
    )
    for rows, message in [
        ([{"item_id": "0", "question": "Q"}], "cover exactly"),
        ([{"item_id": "0", "question": "Q"}] * 2, "Duplicate"),
        ([{"item_id": "2", "question": "Q"}], "Missing/invalid difficulty"),
        ([{"question": "Q"}], "Missing item_id"),
    ]:
        save(path, rows)
        with pytest.raises(ValueError, match=message):
            read_items(path, difficulty)


def test_prepare_and_generate_dry_run(tmp_path, monkeypatch, capsys):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    data = items()
    matrix_path = tmp_path / "responses.npy"
    np.save(matrix_path, np.random.default_rng(0).integers(0, 2, (260, 8)))
    question_path, difficulty_path = tmp_path / "items.json", tmp_path / "irt.json"
    save(
        question_path,
        [{"item_id": r["item_id"], "question": r["text"]} for r in data[::-1]],
    )
    save(
        difficulty_path,
        {
            "items": data,
            "matrix_shape": [260, 8],
            "response_sha256": hashlib.sha256(matrix_path.read_bytes()).hexdigest(),
        },
    )
    prepared_path = tmp_path / "prepared.json"
    prepare_args = [
        "prepare",
        "--items",
        str(question_path),
        "--difficulty",
        str(difficulty_path),
        "--responses",
        str(matrix_path),
        "--v-size",
        "30",
        "--t-size",
        "30",
        "--output",
        str(prepared_path),
    ]
    monkeypatch.setattr(sys, "argv", ["discover", *prepare_args])
    main()
    prepared = json.loads(prepared_path.read_text())
    assert prepared["items"] == data
    assert {k: len(v) for k, v in prepared["splits"].items()} == {
        "G": 200,
        "V": 30,
        "T": 30,
    }
    capsys.readouterr()
    output = tmp_path / "hypotheses.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "discover",
            "generate",
            "--input",
            str(prepared_path),
            "--output",
            str(output),
        ],
    )
    main()
    dry_run = json.loads(capsys.readouterr().out)
    assert dry_run["generation_calls"] == 80 and dry_run["item_presentations"] == 1700
    assert not output.exists()
    # A same-shape but different response matrix must not pass provenance checks.
    np.save(matrix_path, np.zeros((260, 8)))
    bad_output = tmp_path / "bad-prepared.json"
    monkeypatch.setattr(sys, "argv", ["discover", *prepare_args[:-1], str(bad_output)])
    with pytest.raises(ValueError, match="checksum differs"):
        main()
    assert not bad_output.exists()


def test_separate_selection_cli(tmp_path, monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    data = items(40)
    result = {
        "items": data,
        "splits": make_splits(data, 20, 10, 0),
        "hypotheses": [{"hypothesis_id": "H001", "text": "Property"}],
        "counts": {"raw_candidates": 1, "distinct_candidates": 1, "refined": 1},
        "matches": {
            "item_ids": [r["item_id"] for r in data],
            "hypothesis_ids": ["H001"],
            "values": [[int(r["difficulty"] > 0)] for r in data],
        },
    }
    output, cache = tmp_path / "result.json", tmp_path / "cache"
    save(output, result)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "discover",
            "select",
            "--input",
            str(output),
            "--repeats",
            "2",
            "--top-k",
            "1",
            "--cache",
            str(cache),
        ],
    )
    main()
    selected = json.loads(output.read_text())
    assert selected["hypotheses"][0]["selected"]
    assert selected["matches"] == result["matches"]
    assert selected["selection"]["repeats"] == 2
    backups = list(cache.glob("selection-input-*.json"))
    assert len(backups) == 1 and json.loads(backups[0].read_text()) == result
    # Refuse misaligned rows rather than silently selecting on wrong targets.
    result["matches"]["item_ids"].reverse()
    save(output, result)
    before = output.read_bytes()
    with pytest.raises(ValueError, match="rows do not align"):
        main()
    assert output.read_bytes() == before


def test_selection_configuration():
    data = items(40)
    splits = make_splits(data, 20, 10, 0)
    matrix = [[int(r["difficulty"] > 0)] for r in data]
    result = select(matrix, data, splits, repeats=1, sample_fraction=0.5, cv_folds=2)
    assert result["sample_fraction"] == 0.5 and result["cv_folds"] == 2
    with pytest.raises(ValueError, match="binary"):
        select([[2]] * 40, data, splits, repeats=1)
    with pytest.raises(ValueError, match="subsample"):
        select(matrix, data, splits, sample_fraction=0.1)


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
