"""Fit a hierarchical 1PL model to a question-by-model response matrix."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def validate_matrix(matrix, item_ids=None, model_ids=None):
    matrix = np.asarray(matrix, dtype=float)
    if matrix.ndim != 2 or min(matrix.shape) < 2:
        raise ValueError("Expected at least two items and two models (items x models).")
    observed = ~np.isnan(matrix) & (matrix != -1)
    if not np.isin(matrix[observed], [0, 1]).all():
        raise ValueError("Responses must be 0, 1, -1, or NaN; -1/NaN mean missing.")
    if not observed.any(axis=0).all() or not observed.any(axis=1).all():
        raise ValueError(
            "Remove all-missing rows/columns and their IDs before fitting."
        )
    ids = []
    for supplied, size in [(item_ids, matrix.shape[0]), (model_ids, matrix.shape[1])]:
        values = (
            list(map(str, supplied))
            if supplied is not None
            else list(map(str, range(size)))
        )
        if len(values) != size or len(set(values)) != size:
            raise ValueError(
                "IDs must be unique and match the corresponding matrix dimension."
            )
        ids.append(values)
    return matrix, observed, *ids


def fit(matrix, item_ids=None, model_ids=None, epochs=500, seed=0, lr=0.1):
    import pyro
    from py_irt.config import IrtConfig
    from py_irt.dataset import Dataset
    from py_irt.training import IrtModelTrainer

    if epochs < 1 or not 0 < lr <= 1:
        raise ValueError("epochs must be positive and lr must be in (0, 1].")
    matrix, observed, item_ids, model_ids = validate_matrix(matrix, item_ids, model_ids)
    rows, cols = np.nonzero(observed)
    data = Dataset(
        item_ids=set(item_ids),
        subject_ids=set(model_ids),
        item_id_to_ix={v: i for i, v in enumerate(item_ids)},
        ix_to_item_id=dict(enumerate(item_ids)),
        subject_id_to_ix={v: i for i, v in enumerate(model_ids)},
        ix_to_subject_id=dict(enumerate(model_ids)),
        observation_items=rows.tolist(),
        observation_subjects=cols.tolist(),
        observations=matrix[observed].tolist(),
        training_example=[True] * len(rows),
    )
    pyro.clear_param_store()
    pyro.set_rng_seed(seed)
    config = IrtConfig(
        model_type="1pl",
        priors="hierarchical",
        epochs=epochs,
        lr=lr,
        seed=seed,
        log_every=epochs,
    )
    trainer = IrtModelTrainer(data_path=None, config=config, dataset=data)
    trainer.train(epochs=epochs, device="cpu")
    store = pyro.get_param_store()
    values = {
        k: store[k].detach().cpu().numpy()
        for k in ["loc_diff", "scale_diff", "loc_ability", "scale_ability"]
    }
    if not all(np.isfinite(v).all() for v in values.values()):
        raise RuntimeError(
            "Non-finite posterior parameters; check data and learning rate."
        )
    return {
        "schema_version": 1,
        "config": {
            "model": "1pl",
            "priors": "hierarchical",
            "epochs": epochs,
            "seed": seed,
            "lr": lr,
            "parameter_source": "final_variational_iterate",
        },
        "matrix_shape": list(matrix.shape),
        "n_observed": int(observed.sum()),
        "items": [
            {
                "item_id": v,
                "difficulty": float(values["loc_diff"][i]),
                "difficulty_sd": float(values["scale_diff"][i]),
            }
            for i, v in enumerate(item_ids)
        ],
        "models": [
            {
                "model_id": v,
                "ability": float(values["loc_ability"][i]),
                "ability_sd": float(values["scale_ability"][i]),
            }
            for i, v in enumerate(model_ids)
        ],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--responses", type=Path, required=True)
    parser.add_argument("--item-ids", type=Path)
    parser.add_argument("--model-ids", type=Path)
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--output", type=Path, default=Path("output/irt_results.json"))
    args = parser.parse_args()
    if args.output.exists():
        parser.error(
            "Output already exists; choose another path to preserve the previous fit."
        )
    result = fit(
        np.load(args.responses, allow_pickle=False),
        json.loads(args.item_ids.read_text()) if args.item_ids else None,
        json.loads(args.model_ids.read_text()) if args.model_ids else None,
        args.epochs,
        args.seed,
        args.lr,
    )
    with args.responses.open("rb") as stream:
        digest = hashlib.sha256()
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    result["response_sha256"] = digest.hexdigest()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temp = args.output.with_suffix(".tmp")
    temp.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    temp.replace(args.output)
    print(args.output)


if __name__ == "__main__":
    main()
