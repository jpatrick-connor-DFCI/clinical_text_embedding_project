"""Landmark-specific cohort evaluation; raw risks are never pooled across landmarks."""

import math
import numpy as np
import pandas as pd
import torch
from lifelines.utils import concordance_index

from .losses import full_breslow_loss


def safe_c_index(time, log_risk, event):
    time, score, event = np.asarray(time), np.asarray(log_risk), np.asarray(event)
    if len(time) < 2:
        return float("nan"), "fewer_than_two_patients"
    if not np.isfinite(time).all() or not np.isfinite(score).all():
        raise ValueError("C-index inputs must be finite")
    if not np.isin(event, [0, 1]).all():
        raise ValueError("C-index event indicators must be binary")
    if not event.sum():
        return float("nan"), "no_primary_events"
    try:
        return float(concordance_index(time, -score, event)), None
    except ZeroDivisionError:
        return float("nan"), "no_comparable_pairs"


def weighted_validation_score(metrics):
    available = [m for m in metrics if math.isfinite(m["c_index"]) and m["n_eligible"] > 0]
    if not available:
        return float("nan")
    return sum(m["c_index"] * m["n_eligible"] for m in available) / sum(m["n_eligible"] for m in available)


@torch.no_grad()
def predict_patients(model, data, patient_ids, landmarks, config, static_transform,
                     model_id, retrospective=False):
    """Outcome-free prediction or eligible retrospective first-event prediction."""
    model.eval()
    rows = []
    if retrospective and data.outcomes is None:
        raise ValueError("Retrospective prediction requires outcomes")
    for landmark in sorted(landmarks):
        ids = sorted(patient_ids)
        if retrospective:
            ids = [p for p in ids if float(data.outcomes.loc[p, "time"]) > landmark]
        for start in range(0, len(ids), config.eval_batch_size):
            patients = ids[start:start + config.eval_batch_size]
            batch, counts = data.batch(patients, landmark, config, static_transform, config.device)
            scores = model(**batch).cpu().numpy()
            if not np.isfinite(scores).all():
                raise RuntimeError("Model produced nonfinite log-risk predictions")
            for patient, score, count in zip(patients, scores, counts):
                row = {"patient_id": patient, "landmark": float(landmark),
                       "log_risk_event": float(score), "n_notes_used": int(count), "model_id": model_id}
                if retrospective:
                    label = data.outcomes.loc[patient]
                    row.update(time_from_index=float(label["time"]),
                               time_from_landmark=float(label["time"] - landmark),
                               event=int(label["event"]), death_event=int(label["death_event"]))
                rows.append(row)
    columns = ["patient_id", "landmark", "log_risk_event", "n_notes_used", "model_id"]
    if retrospective:
        columns += ["time_from_index", "time_from_landmark", "event", "death_event"]
    return pd.DataFrame(rows, columns=columns)


def metrics_from_predictions(predictions, landmarks):
    metrics = []
    for landmark in sorted(landmarks):
        frame = predictions[predictions["landmark"] == landmark]
        time = frame["time_from_landmark"].to_numpy(dtype=float)
        event = frame["event"].to_numpy(dtype=int)
        score = frame["log_risk_event"].to_numpy(dtype=float)
        c, reason = safe_c_index(time, score, event)
        loss = float(full_breslow_loss(torch.tensor(score), torch.tensor(time),
                                      torch.tensor(event))) if event.sum() else float("nan")
        metrics.append({"landmark": float(landmark), "c_index": c, "c_index_reason": reason,
                        "n_eligible": len(frame), "n_events": int(event.sum()),
                        "n_deaths": int(frame["death_event"].sum()), "full_breslow_loss": loss})
    return metrics


def evaluate_model(model, data, patient_ids, config, static_transform, model_id="validation"):
    predictions = predict_patients(model, data, patient_ids, config.landmarks, config,
                                   static_transform, model_id, retrospective=True)
    metrics = metrics_from_predictions(predictions, config.landmarks)
    return weighted_validation_score(metrics), metrics, predictions
