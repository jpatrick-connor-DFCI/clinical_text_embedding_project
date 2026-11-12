"""Seeded random search with fixed splits, fingerprints and validation-only ranking."""

import argparse
import json
import math
from pathlib import Path
import time

import numpy as np
import pandas as pd

from .config import Config
from .dataset import StaticTransform, digest
from .train import (add_data_arguments, atomic_csv, atomic_json, atomic_torch,
                    data_from_args, fit, json_safe, load_checkpoint, parse_landmarks)


SEARCH_FIELDS = {"d_model", "n_heads", "n_layers", "dim_ff", "dropout", "lr", "weight_decay",
                 "batch_size", "risk_set_max_controls", "max_notes", "time_scale_days",
                 "landmark_scale_days", "gap_cap_days"}
INTEGER_FIELDS = {"d_model", "n_heads", "n_layers", "dim_ff", "batch_size", "risk_set_max_controls", "max_notes"}


def validate_search_space(space):
    if not isinstance(space, dict) or set(space) - SEARCH_FIELDS:
        raise ValueError(f"Search fields must be a subset of {sorted(SEARCH_FIELDS)}")
    for name, spec in space.items():
        if isinstance(spec, list):
            if not spec or not all(isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x) for x in spec):
                raise ValueError(f"{name}: candidate lists must contain finite numbers")
            if name in INTEGER_FIELDS and not all(isinstance(x, int) for x in spec):
                raise ValueError(f"{name}: use integer candidates")
        elif isinstance(spec, dict):
            if set(spec) != {"type", "low", "high"} or spec["type"] not in {"uniform", "loguniform", "int"}:
                raise ValueError(f"{name}: expected type/low/high distribution")
            low, high = spec["low"], spec["high"]
            if not all(isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x) for x in (low, high)) or low > high:
                raise ValueError(f"{name}: invalid distribution bounds")
            if spec["type"] == "loguniform" and low <= 0:
                raise ValueError(f"{name}: loguniform bounds must be positive")
            if spec["type"] == "int" and not all(isinstance(x, int) for x in (low, high)):
                raise ValueError(f"{name}: int bounds must be integers")
            if (name in INTEGER_FIELDS) != (spec["type"] == "int"):
                raise ValueError(f"{name}: integer fields require int distributions; continuous fields require uniform/loguniform")
        else:
            raise ValueError(f"{name}: expected candidate list or distribution")


def sample_plan(base_config, space, num_trials, search_seed):
    validate_search_space(space)
    if num_trials < 1 or search_seed < 0:
        raise ValueError("num_trials must be positive and search_seed nonnegative")
    rng = np.random.default_rng(search_seed)
    configs, rejected = [], []
    for trial in range(num_trials):
        for attempt in range(1000):
            values = base_config.to_dict()
            for name in sorted(space):
                spec = space[name]
                if isinstance(spec, list):
                    value = spec[int(rng.integers(len(spec)))]
                elif spec["type"] == "int":
                    value = int(rng.integers(spec["low"], spec["high"] + 1))
                elif spec["type"] == "uniform":
                    value = float(rng.uniform(spec["low"], spec["high"]))
                else:
                    value = float(np.exp(rng.uniform(np.log(spec["low"]), np.log(spec["high"]))))
                values[name] = value
            values["seed"] = (base_config.seed + 7919 * trial) % (2 ** 32)
            try:
                config = Config.from_dict(values)
            except (TypeError, ValueError) as error:
                rejected.append({"trial_id": trial, "attempt": attempt, "error": str(error), "config": values})
                continue
            configs.append(config)
            break
        else:
            raise ValueError("Could not draw a compatible configuration in 1000 attempts; inspect the search space")
    return configs, rejected


def run_search(data, base_config, space, num_trials, search_seed, output_dir, resume=False):
    """Never evaluates the test split. Interrupted trials resume at epoch boundaries."""
    base_config.validate()
    configs, rejected = sample_plan(base_config, space, num_trials, search_seed)
    fingerprint = digest({"data": data.fingerprint, "base_config": base_config.to_dict(),
                          "space": space, "search_seed": search_seed, "schema": 1})
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = output / "search_manifest.json"
    if manifest_path.exists():
        existing = json.loads(manifest_path.read_text())
        if existing["search_fingerprint"] != fingerprint:
            raise ValueError("Search resume fingerprint mismatch: data, splits, base config, seed or space changed")
        if not resume:
            raise ValueError("Search already exists; use --resume or a new output directory")
        if num_trials < existing["num_trials"]:
            raise ValueError("Cannot shrink an existing search")
        if existing["trial_configs"] != [c.to_dict() for c in configs[:existing["num_trials"]]]:
            raise ValueError("Reconstructed trial plan does not match the saved search")
    elif any(output.iterdir()):
        raise ValueError("Search output directory is nonempty without a valid manifest")
    atomic_json({"search_fingerprint": fingerprint, "data_fingerprint": data.fingerprint,
                 "num_trials": num_trials, "search_seed": search_seed,
                 "base_config": base_config.to_dict(), "search_space": space,
                 "trial_configs": [c.to_dict() for c in configs],
                 "rejected_configurations": rejected}, manifest_path)
    records = []
    successful = []
    for trial_id, config in enumerate(configs):
        trial_dir = output / "trials" / f"trial_{trial_id:04d}"
        trial_dir.mkdir(parents=True, exist_ok=True)
        record_path = trial_dir / "trial_result.json"
        record = json.loads(record_path.read_text()) if record_path.exists() else None
        if record is not None and record["status"] == "success":
            checkpoint = load_checkpoint(record["checkpoint_path"])
            if checkpoint["data_fingerprint"] != data.fingerprint or checkpoint["run_fingerprint"] != record["run_fingerprint"]:
                raise ValueError("Successful trial checkpoint provenance mismatch")
            expected = config.to_dict()
            expected["input_dim"] = data.input_dim
            expected["static_dim"] = StaticTransform.fit(data.patients, data.split_ids("train")).output_dim
            if checkpoint["config"] != expected or record["config"] != expected:
                raise ValueError("Successful trial configuration differs from its saved plan")
            if record["validation_score"] != checkpoint["best_validation_score"]:
                raise ValueError("Successful trial validation score differs from its checkpoint")
            print(f"Reusing successful trial {trial_id}", flush=True)
        else:
            record = {"trial_id": trial_id, "status": "running", "config": config.to_dict(),
                      "validation_score": None, "checkpoint_path": "", "runtime_seconds": 0., "error": ""}
            atomic_json(record, record_path)
            _write_results(records + [record], output, base_config.landmarks)
            started = time.monotonic()
            try:
                result = fit(data, config, trial_dir, resume=(trial_dir / "run_manifest.json").exists())
                record.update(result)
                record["config"] = result["config"]
            except Exception as error:
                record.update(status="failed", error=f"{type(error).__name__}: {error}",
                              runtime_seconds=time.monotonic() - started)
            atomic_json(record, record_path)
        records.append(record)
        if record["status"] == "success":
            successful.append(record)
        _write_results(records, output, base_config.landmarks)
    if not successful:
        raise RuntimeError("All search trials failed; inspect hparam_search_results.csv and trial_result.json")
    # Stable tie-break: earliest trial. Only the validation metric enters ranking.
    best = max(successful, key=lambda row: (row["validation_score"], -row["trial_id"]))
    state = load_checkpoint(best["checkpoint_path"])
    atomic_torch(state, output / "best_model.pt")
    atomic_json(best["config"], output / "best_config.json")
    summary = {"search_fingerprint": fingerprint, "selected_trial": best["trial_id"],
               "validation_score": best["validation_score"], "landmark_metrics": best["landmark_metrics"],
               "checkpoint_path": str((output / "best_model.pt").resolve()),
               "successful_trials": len(successful), "failed_trials": len(records) - len(successful),
               "test_evaluated": False}
    atomic_json(summary, output / "search_summary.json")
    return summary


def _write_results(records, output, landmarks):
    rows = []
    for record in records:
        row = {k: record.get(k) for k in ("trial_id", "status", "validation_score", "best_epoch",
                                         "checkpoint_path", "runtime_seconds", "error", "run_fingerprint")}
        row["config_json"] = json.dumps(record["config"], sort_keys=True)
        for metric in record.get("landmark_metrics", []):
            prefix = f"landmark_{metric['landmark']:g}"
            row[prefix + "_c_index"] = metric["c_index"]
            row[prefix + "_n_eligible"] = metric["n_eligible"]
            row[prefix + "_n_events"] = metric["n_events"]
            row[prefix + "_n_deaths"] = metric["n_deaths"]
        rows.append(row)
    atomic_csv(pd.DataFrame(rows), output / "hparam_search_results.csv")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    add_data_arguments(parser)
    parser.add_argument("--config")
    parser.add_argument("--landmarks", type=parse_landmarks)
    parser.add_argument("--search_space_json", required=True)
    parser.add_argument("--num_trials", type=int, default=30)
    parser.add_argument("--search_seed", type=int, default=42)
    parser.add_argument("--n_epochs", type=int)
    parser.add_argument("--patience", type=int)
    parser.add_argument("--device")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    config = Config.from_json(args.config) if args.config else Config()
    if args.landmarks is not None:
        config.landmarks = args.landmarks
    if args.n_epochs is not None:
        config.num_epochs = args.n_epochs
    if args.patience is not None:
        config.patience = args.patience
    if args.device is not None:
        config.device = args.device
    space = json.loads(Path(args.search_space_json).read_text())
    result = run_search(data_from_args(args), config, space, args.num_trials,
                        args.search_seed, args.output_dir, args.resume)
    print(json.dumps(json_safe(result), indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
