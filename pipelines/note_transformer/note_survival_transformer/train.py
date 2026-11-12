"""Single-configuration fitting, atomic checkpoints and deterministic epoch resume."""

import argparse
from dataclasses import replace
import json
import math
import os
from pathlib import Path
import random
import tempfile
import time

import numpy as np
import torch

from .config import Config
from .dataset import StaticTransform, digest, load_data
from .evaluate import evaluate_model
from .losses import sampled_cox_loss
from .model import NoteSurvivalTransformer
from .risk_sets import RiskSetSampler


def json_safe(value):
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, (float, np.floating)) and not math.isfinite(value):
        return None
    if isinstance(value, np.generic):
        return value.item()
    return value


def _atomic(path, writer):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # Some PyTorch zip writers reject dot-prefixed extensionless filenames.
    fd, temporary = tempfile.mkstemp(prefix="pending_", suffix=path.suffix, dir=path.parent)
    os.close(fd)
    try:
        writer(temporary)
        with open(temporary, "rb") as stream:
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def atomic_json(value, path):
    _atomic(path, lambda temporary: Path(temporary).write_text(
        json.dumps(json_safe(value), indent=2, sort_keys=True, allow_nan=False) + "\n"))


def atomic_torch(value, path):
    _atomic(path, lambda temporary: torch.save(value, temporary))


def atomic_csv(frame, path):
    _atomic(path, lambda temporary: frame.to_csv(temporary, index=False))


def load_checkpoint(path):
    # Checkpoints contain optimizer/RNG metadata: load only trusted package artifacts.
    state = torch.load(path, map_location="cpu", weights_only=False)
    if state.get("schema_version") != 1 or state.get("package") != "note_survival_transformer":
        raise ValueError("Unsupported checkpoint schema")
    return state


def seed_everything(seed, num_threads=1):
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(num_threads)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.benchmark = False
    if hasattr(torch.backends, "cuda"):
        torch.backends.cuda.matmul.allow_tf32 = False


def _rng_state():
    return {"python": random.getstate(), "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None}


def _restore_rng(state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if state["cuda"] is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])


def train_epoch(model, data, sampler, static_transform, optimizer, config, epoch):
    """One optimizer step per batch_size groups; backprop one group at a time.

    This computes the exact mean of the groups' sampled losses without holding
    every group's dense inputs or autograd graph simultaneously in memory.
    A singleton risk set has zero loss and is included in the group mean.
    """
    model.train()
    groups = iter(sampler.epoch_groups(config.seed + 1009 * (epoch + 1)))
    total, count = 0., 0
    while True:
        chunk = []
        for _ in range(config.batch_size):
            group = next(groups, None)
            if group is None:
                break
            chunk.append(group)
        if not chunk:
            break
        optimizer.zero_grad(set_to_none=True)
        for group in chunk:
            batch, _ = data.batch(group.member_ids, group.landmark, config, static_transform, config.device)
            scores = model(**batch)
            loss = sampled_cox_loss(scores, group.weights)
            if not torch.isfinite(loss):
                raise RuntimeError("Nonfinite sampled Cox loss")
            (loss / len(chunk)).backward()
            total += float(loss.detach())
            count += 1
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(),
                                             config.max_grad_norm or float("inf"))
        if not torch.isfinite(norm):
            raise RuntimeError("Nonfinite gradient norm")
        optimizer.step()
    if not count:
        raise ValueError("No eligible primary events for training")
    return total / count


def fit(data, config, output_dir, resume=False, stop_after_epoch=None):
    """Fit using train/validation only. Test IDs are never evaluated here.

    stop_after_epoch is an integration/testing hook for an orderly interruption;
    resume restarts at the next completed-epoch boundary. A killed partial epoch
    is replayed from the preceding atomic last_model checkpoint.
    """
    if data.outcomes is None or data.splits is None:
        raise ValueError("Training requires outcomes and patient-level splits")
    static = StaticTransform.fit(data.patients, data.split_ids("train"))
    if config.input_dim not in (0, data.input_dim) or config.static_dim not in (0, static.output_dim):
        raise ValueError("Configured input/static dimensions do not match data")
    config = replace(config, input_dim=data.input_dim, static_dim=static.output_dim).validate(True)
    if config.device.startswith("cuda") and not torch.cuda.is_available():
        raise ValueError("CUDA requested but unavailable")
    seed_everything(config.seed, config.num_threads)
    sampler = RiskSetSampler(data.outcomes, data.split_ids("train"), config.landmarks,
                             config.risk_set_max_controls, config.use_ipw)
    if len(sampler.event_indices) == 0:
        raise ValueError("No eligible primary events; review landmarks/outcomes")
    if sampler.singleton_events == len(sampler.event_indices):
        raise ValueError("All eligible event groups are singleton risk sets; training has no ranking signal")
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    run_fingerprint = digest({"data": data.fingerprint, "config": config.to_dict(),
                              "static": static.to_dict(), "objective_version": 1})
    manifest = {"run_fingerprint": run_fingerprint, "data_fingerprint": data.fingerprint,
                "config": config.to_dict(), "risk_set_audit": sampler.summary(),
                "embedding_model_id": data.embedding_model_id, "static_transform": static.to_dict()}
    manifest_path = output / "run_manifest.json"
    if manifest_path.exists():
        existing = json.loads(manifest_path.read_text())
        if existing["run_fingerprint"] != run_fingerprint:
            raise ValueError("Resume/input/config fingerprint mismatch; use a new output directory")
        if not resume:
            raise ValueError("Output directory already contains a run; use --resume or a new directory")
    elif resume and any(output.iterdir()):
        raise ValueError("Cannot resume an output directory without a valid manifest")
    atomic_json(manifest, manifest_path)
    atomic_json(config.to_dict(), output / "config.json")
    model = NoteSurvivalTransformer(config).to(config.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, config.num_epochs) if config.scheduler == "cosine" else None
    best_score, best_epoch, best_metrics, stale, history, start_epoch = -float("inf"), -1, [], 0, [], 0
    elapsed_previous = 0.
    last_path = output / "last_model.pt"
    best_path = output / "best_model.pt"
    if resume and last_path.exists():
        state = load_checkpoint(last_path)
        if state["run_fingerprint"] != run_fingerprint:
            raise ValueError("Checkpoint fingerprint mismatch")
        model.load_state_dict(state["model_state_dict"])
        optimizer.load_state_dict(state["optimizer_state_dict"])
        if scheduler is not None:
            scheduler.load_state_dict(state["scheduler_state_dict"])
        _restore_rng(state["rng_state"])
        best_score, best_epoch = state["best_validation_score"], state["best_epoch"]
        best_metrics, stale, history = state["best_landmark_metrics"], state["stale_epochs"], state["history"]
        start_epoch, elapsed_previous = state["epoch"] + 1, state["runtime_seconds"]
        if state["completed"]:
            result = json.loads((output / "result.json").read_text()) if (output / "result.json").exists() else _result(state, best_path)
            if not best_path.exists():
                raise ValueError("Completed run is missing its best checkpoint")
            atomic_json(result, output / "result.json")
            return result
    started = time.monotonic()
    base = {"schema_version": 1, "package": "note_survival_transformer", **manifest,
            "model_id": run_fingerprint[:16]}
    for epoch in range(start_epoch, config.num_epochs):
        loss = train_epoch(model, data, sampler, static, optimizer, config, epoch)
        score, metrics, _ = evaluate_model(model, data, data.split_ids("val"), config, static)
        if not math.isfinite(score):
            raise ValueError("Validation has no computable C-index at any landmark; model selection is undefined")
        improved = score > best_score + config.min_delta
        if improved:
            best_score, best_epoch, best_metrics, stale = score, epoch, metrics, 0
        else:
            stale += 1
        history.append({"epoch": epoch + 1, "train_sampled_loss": loss,
                        "weighted_validation_c_index": score, "landmark_metrics": metrics,
                        "learning_rate": optimizer.param_groups[0]["lr"]})
        if scheduler is not None:
            scheduler.step()
        completed = epoch + 1 == config.num_epochs or (epoch + 1 >= config.min_epochs and stale >= config.patience)
        state = {**base, "epoch": epoch, "model_state_dict": model.state_dict(),
                 "best_validation_score": best_score, "best_epoch": best_epoch,
                 "best_landmark_metrics": best_metrics, "history": history,
                 "stale_epochs": stale, "runtime_seconds": elapsed_previous + time.monotonic() - started,
                 "completed": completed}
        if improved:
            atomic_torch(state, best_path)
        state.update(optimizer_state_dict=optimizer.state_dict(),
                     scheduler_state_dict=scheduler.state_dict() if scheduler else None,
                     rng_state=_rng_state())
        atomic_torch(state, last_path)
        _atomic(output / "training_log.jsonl", lambda temporary: Path(temporary).write_text(
            "".join(json.dumps(json_safe(row), sort_keys=True) + "\n" for row in history)))
        print(json.dumps({"epoch": epoch + 1, "train_loss": loss, "val_c_index": score,
                          "best_val_c_index": best_score}), flush=True)
        if completed:
            result = _result(state, best_path)
            atomic_json(result, output / "result.json")
            return result
        if stop_after_epoch is not None and epoch + 1 >= stop_after_epoch:
            return {"status": "interrupted", "checkpoint_path": str(last_path.resolve())}
    raise RuntimeError("Training did not produce a final result")


def _result(state, best_path):
    return {"status": "success", "config": state["config"], "run_fingerprint": state["run_fingerprint"],
            "validation_score": state["best_validation_score"], "best_epoch": state["best_epoch"] + 1,
            "landmark_metrics": state["best_landmark_metrics"], "risk_set_audit": state["risk_set_audit"],
            "checkpoint_path": str(best_path.resolve()), "runtime_seconds": state["runtime_seconds"]}


def add_data_arguments(parser, require_outcomes=True):
    parser.add_argument("--note_embeddings_path", required=True)
    parser.add_argument("--note_metadata_path", required=True)
    parser.add_argument("--outcomes_path", required=require_outcomes)
    parser.add_argument("--splits_path", required=require_outcomes)
    parser.add_argument("--patients_path")
    parser.add_argument("--embedding_model_id", default="unspecified")


def data_from_args(args):
    return load_data(args.note_embeddings_path, args.note_metadata_path,
                     args.outcomes_path, args.splits_path, args.patients_path,
                     embedding_model_id=args.embedding_model_id)


def parse_landmarks(value):
    try:
        values = tuple(float(x) for x in value.split(","))
    except ValueError as error:
        raise argparse.ArgumentTypeError("landmarks must be comma-separated numbers") from error
    if not values or not all(math.isfinite(x) for x in values) or len(set(values)) != len(values):
        raise argparse.ArgumentTypeError("landmarks must be finite, nonempty and unique")
    return tuple(sorted(values))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    add_data_arguments(parser)
    parser.add_argument("--config")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--landmarks", type=parse_landmarks)
    parser.add_argument("--n_epochs", type=int)
    parser.add_argument("--batch_size", type=int)
    parser.add_argument("--risk_set_max_controls", type=int)
    parser.add_argument("--max_notes", type=int)
    parser.add_argument("--device")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--evaluate_test", action="store_true", help="Explicit final test evaluation after validation selection")
    args = parser.parse_args(argv)
    config = Config.from_json(args.config) if args.config else Config()
    for name in ("landmarks", "batch_size", "risk_set_max_controls", "max_notes", "device"):
        if getattr(args, name) is not None:
            setattr(config, name, getattr(args, name))
    if args.n_epochs is not None:
        config.num_epochs = args.n_epochs
    config.validate()
    data = data_from_args(args)
    result = fit(data, config, args.output_dir, args.resume)
    if args.evaluate_test:
        from .predict import final_test_evaluation
        final_test_evaluation(result["checkpoint_path"], data, Path(args.output_dir) / "test", args.device)
    print(json.dumps(json_safe(result), indent=2))


if __name__ == "__main__":
    main()
