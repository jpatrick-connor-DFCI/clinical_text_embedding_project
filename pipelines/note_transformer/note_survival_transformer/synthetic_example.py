"""Generate four portable input files and optionally run a complete CPU demo."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .config import Config
from .dataset import load_data
from .hparam_search import run_search
from .predict import final_test_evaluation, inference
from .train import atomic_csv, atomic_json, fit


def generate_synthetic(output_dir, n_patients=48, input_dim=8, seed=123):
    if n_patients < 18 or input_dim < 2:
        raise ValueError("Synthetic example requires at least 18 patients and 2 embedding dimensions")
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    names = ("notes.npy", "notes.csv", "outcomes.csv", "splits.csv", "patients.csv")
    if any((output / name).exists() for name in names):
        raise ValueError("Synthetic inputs already exist; use a new directory")
    rng = np.random.default_rng(seed)
    signal = np.linspace(-1.5, 1.5, n_patients)
    rng.shuffle(signal)
    note_rows, embeddings, outcomes, splits, patients = [], [], [], [], []
    n_train = int(n_patients * .6)
    n_val = int(n_patients * .2)
    for i, z in enumerate(signal):
        patient = f"synthetic_{i:03d}"
        # Event time follows a latent baseline signal encoded in each available note.
        event_time = float(12 * np.exp(-1.2 * z) + rng.uniform(.1, 1.))
        event, death = 1, 0
        if i % 13 == 0:
            event_time, event, death = min(event_time, 5.), 0, 1
        elif i % 7 == 0:
            event_time, event = min(event_time, 10.), 0
        if i == 1:
            event_time, event, death = -3., 1, 0  # deliberately no eligible demo landmark
        outcomes.append({"patient_id": patient, "time": event_time, "event": event, "death_event": death})
        split = "train" if i < n_train else "val" if i < n_train + n_val else "test"
        splits.append({"patient_id": patient, "split": split})
        patients.append({"patient_id": patient, "baseline_measurement": z + rng.normal(0, .1),
                         "optional_measurement": np.nan if i % 9 == 0 else rng.normal()})
        if i % 11 == 0:
            continue  # patients with no notes remain in the cohort
        for k, day in enumerate([-5.5, -1.5, 1.25, 4.75]):
            if day > event_time:
                continue
            vector = rng.normal(0, .2, input_dim).astype(np.float32)
            vector[0] = z + rng.normal(0, .02)
            vector[1] = np.tanh(z)
            note_rows.append({"patient_id": patient, "note_id": f"note_{k}",
                              "days_from_index": day, "embedding_row": len(embeddings)})
            embeddings.append(vector)
    np.save(output / "notes.npy", np.stack(embeddings).astype(np.float32))
    # Deliberately shuffled CSV metadata tests deterministic sorting at load time.
    pd.DataFrame(note_rows).sample(frac=1, random_state=seed).to_csv(output / "notes.csv", index=False)
    for filename, rows in (("outcomes.csv", outcomes), ("splits.csv", splits), ("patients.csv", patients)):
        pd.DataFrame(rows).to_csv(output / filename, index=False)
    atomic_json({"embedding_model_id": "synthetic-v1", "index_origin": "arbitrary synthetic baseline",
                 "time_unit": "days", "availability_time_verified": True,
                 "n_patients": n_patients, "input_dim": input_dim, "seed": seed}, output / "provenance.json")
    return output


def run_demo(output_dir):
    output = Path(output_dir)
    data_dir = generate_synthetic(output / "data")
    data = load_data(data_dir / "notes.npy", data_dir / "notes.csv", data_dir / "outcomes.csv",
                     data_dir / "splits.csv", data_dir / "patients.csv", embedding_model_id="synthetic-v1")
    config = Config(d_model=8, n_heads=2, n_layers=1, dim_ff=16, dropout=0., max_notes=8,
                    landmarks=(-2., .5, 2.), num_epochs=3, patience=3, lr=.01,
                    batch_size=4, risk_set_max_controls=8, time_scale_days=10.,
                    landmark_scale_days=10., num_threads=1)
    space = {"d_model": [8, 12], "n_heads": [2], "n_layers": [1], "dim_ff": [16, 24],
             "dropout": [0.0], "lr": [.005, .01], "batch_size": [4],
             "risk_set_max_controls": [8], "max_notes": [8]}
    atomic_json(config.to_dict(), output / "smoke_config.json")
    atomic_json(space, output / "smoke_search_space.json")
    search = run_search(data, config, space, 2, 42, output / "search")
    resumed = run_search(data, config, space, 2, 42, output / "search", resume=True)
    if search["selected_trial"] != resumed["selected_trial"]:
        raise AssertionError("Search resume changed model selection")
    single = fit(data, config, output / "single_run")
    test = final_test_evaluation(search["checkpoint_path"], data, output / "test", "cpu")
    new_data = load_data(data_dir / "notes.npy", data_dir / "notes.csv", patients_path=data_dir / "patients.csv",
                         embedding_model_id="synthetic-v1")
    predictions = inference(search["checkpoint_path"], new_data, [-2., .5, 2.])
    atomic_csv(predictions, output / "outcome_free_predictions.csv")
    report = {"status": "PASS", "search": search, "single_run_checkpoint": single["checkpoint_path"],
              "test": test, "outcome_free_prediction_rows": len(predictions),
              "predictions_path": str((output / "outcome_free_predictions.csv").resolve()),
              "search_resume_verified": True}
    atomic_json(report, output / "demo_results.json")
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--run_demo", action="store_true")
    parser.add_argument("--n_patients", type=int, default=48)
    parser.add_argument("--input_dim", type=int, default=8)
    parser.add_argument("--seed", type=int, default=123)
    args = parser.parse_args(argv)
    if args.run_demo:
        result = run_demo(args.output_dir)
    else:
        result = {"data_directory": str(generate_synthetic(args.output_dir, args.n_patients, args.input_dim, args.seed).resolve())}
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
