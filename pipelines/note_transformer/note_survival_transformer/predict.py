"""Outcome-free inference and an explicit, separate held-out test evaluation."""

import argparse
from dataclasses import replace
import json
from pathlib import Path

import torch

from .config import Config
from .dataset import StaticTransform, load_data
from .evaluate import evaluate_model, predict_patients
from .model import NoteSurvivalTransformer
from .train import atomic_csv, atomic_json, json_safe, load_checkpoint, parse_landmarks, seed_everything


def restore_model(checkpoint, device="cpu", num_threads=None):
    """Restore a model with optional runtime settings, leaving checkpoint metadata intact."""
    state = load_checkpoint(checkpoint)
    overrides = {"device": device}
    if num_threads is not None:
        overrides["num_threads"] = num_threads
    config = replace(Config.from_dict(state["config"]), **overrides)
    config.validate(True)
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise ValueError("CUDA requested but unavailable")
    seed_everything(config.seed, config.num_threads)
    model = NoteSurvivalTransformer(config).to(device)
    model.load_state_dict(state["model_state_dict"])
    model.eval()
    return model, config, StaticTransform(**state["static_transform"]), state


def _validate_embeddings(data, config, state):
    if data.input_dim != config.input_dim:
        raise ValueError("Embedding dimension differs from checkpoint")
    trained_id = state["embedding_model_id"]
    if trained_id != "unspecified" and data.embedding_model_id != trained_id:
        raise ValueError("Embedding model/version differs from checkpoint; supply the correct --embedding_model_id")


def inference(checkpoint, data, landmarks=None, device="cpu", patient_ids=None):
    model, config, static, state = restore_model(checkpoint, device)
    _validate_embeddings(data, config, state)
    landmarks = config.landmarks if landmarks is None else tuple(landmarks)
    # Validate finite/unique anchors while allowing a new inference schedule.
    config = replace(config, landmarks=landmarks).validate(True)
    ids = data.patient_ids if patient_ids is None else patient_ids
    if not set(ids).issubset(set(data.patient_ids)):
        raise ValueError("Prediction patient IDs are absent from input metadata")
    return predict_patients(model, data, ids, landmarks, config, static, state["model_id"])


def final_test_evaluation(checkpoint, data, output_dir, device=None, num_threads=None):
    """Evaluate the saved checkpoint with its original split fingerprint and optional runtime threads."""
    state = load_checkpoint(checkpoint)
    if data.fingerprint != state["data_fingerprint"]:
        raise ValueError("Final test inputs/splits differ from the model-selection data fingerprint")
    model, config, static, state = restore_model(
        checkpoint, device or state["config"]["device"], num_threads=num_threads)
    score, metrics, predictions = evaluate_model(model, data, data.split_ids("test"), config, static, state["model_id"])
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    atomic_csv(predictions, output / "test_predictions.csv")
    report = {"weighted_test_c_index": score, "landmark_metrics": metrics,
              "model_id": state["model_id"], "checkpoint_path": str(Path(checkpoint).resolve()),
              "data_fingerprint": data.fingerprint}
    atomic_json(report, output / "test_metrics.json")
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--note_embeddings_path", required=True)
    parser.add_argument("--note_metadata_path", required=True)
    parser.add_argument("--patients_path")
    parser.add_argument("--patient_ids_path", help="Optional patient_id CSV to include new patients without notes")
    parser.add_argument("--embedding_model_id", default="unspecified")
    parser.add_argument("--landmarks", type=parse_landmarks)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output_csv")
    parser.add_argument("--evaluate_test", action="store_true")
    parser.add_argument("--outcomes_path")
    parser.add_argument("--splits_path")
    parser.add_argument("--output_dir")
    args = parser.parse_args(argv)
    if args.evaluate_test:
        if not args.outcomes_path or not args.splits_path or not args.output_dir:
            parser.error("--evaluate_test requires --outcomes_path, --splits_path and --output_dir")
        if args.landmarks is not None or args.patient_ids_path:
            parser.error("Final test evaluation uses the checkpoint's fixed landmarks and original splits")
    elif not args.output_csv:
        parser.error("Inference requires --output_csv")
    elif args.outcomes_path or args.splits_path:
        parser.error("Outcome-free inference does not accept outcomes/splits; use --evaluate_test")
    data = load_data(args.note_embeddings_path, args.note_metadata_path,
                     args.outcomes_path, args.splits_path, args.patients_path, args.patient_ids_path,
                     args.embedding_model_id)
    if args.evaluate_test:
        result = final_test_evaluation(args.checkpoint, data, args.output_dir, args.device)
        print(json.dumps(json_safe(result), indent=2, allow_nan=False))
    else:
        predictions = inference(args.checkpoint, data, args.landmarks, args.device)
        atomic_csv(predictions, args.output_csv)
        print(json.dumps({"n_predictions": len(predictions), "output_csv": str(Path(args.output_csv).resolve())}))


if __name__ == "__main__":
    main()
