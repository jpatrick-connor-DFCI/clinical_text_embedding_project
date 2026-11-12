"""Validated file contracts and landmark histories, independent of outcomes."""

from dataclasses import dataclass
import hashlib
import json
import numpy as np
import pandas as pd
import torch


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                    allow_nan=False).encode()).hexdigest()


def file_digest(path):
    h = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _read_table(path, required):
    frame = pd.read_csv(path, dtype={"patient_id": "string", "note_id": "string"})
    missing = set(required) - set(frame.columns)
    if missing:
        raise ValueError(f"{path}: missing columns {sorted(missing)}")
    for name in ("patient_id", "note_id"):
        if name in frame:
            invalid = frame[name].isna() | frame[name].str.strip().eq("")
            if invalid.any() or frame[name].str.strip().ne(frame[name]).any():
                raise ValueError(f"{path}: {name} must contain nonempty, unpadded strings")
            frame[name] = frame[name].astype(str)
    return frame


def _numeric(series, name, allow_missing=False):
    values = pd.to_numeric(series, errors="raise").to_numpy(dtype=np.float64)
    if np.isinf(values).any() or (not allow_missing and np.isnan(values).any()):
        raise ValueError(f"{name} must be finite")
    return values


def _binary(series, name):
    values = pd.to_numeric(series.astype(str).str.lower().replace(
        {"true": "1", "false": "0"}), errors="raise").to_numpy(dtype=float)
    if not np.isin(values, [0, 1]).all():
        raise ValueError(f"{name} must contain only 0/1")
    return values.astype(np.int64)


@dataclass
class StaticTransform:
    columns: list
    medians: list
    means: list
    scales: list

    @classmethod
    def fit(cls, patients, train_ids):
        columns = [c for c in patients.columns if c != "patient_id"] if patients is not None else []
        if not columns:
            return cls([], [], [], [])
        indexed = patients.set_index("patient_id").reindex(train_ids)
        train = np.column_stack([_numeric(indexed[c], c, True) for c in columns])
        medians = np.array([np.median(x[np.isfinite(x)]) if np.isfinite(x).any() else 0.
                            for x in train.T])
        filled = np.where(np.isnan(train), medians, train)
        means = filled.mean(axis=0)
        scales = filled.std(axis=0)
        scales[scales < 1e-8] = 1.
        return cls(columns, medians.tolist(), means.tolist(), scales.tolist())

    @property
    def output_dim(self):
        return 2 * len(self.columns)  # standardized values plus per-feature missingness

    def transform(self, patients, patient_ids):
        if not self.columns:
            return {p: np.empty(0, dtype=np.float32) for p in patient_ids}
        if patients is None or set(patients.columns) != {"patient_id", *self.columns}:
            raise ValueError("patients.csv must match the checkpoint's static columns exactly")
        indexed = patients.set_index("patient_id").reindex(patient_ids)
        raw = np.column_stack([_numeric(indexed[c], c, True) for c in self.columns])
        missing = np.isnan(raw)
        filled = np.where(missing, np.asarray(self.medians), raw)
        scaled = (filled - np.asarray(self.means)) / np.asarray(self.scales)
        result = np.concatenate([scaled, missing.astype(float)], axis=1).astype(np.float32)
        if not np.isfinite(result).all():
            raise ValueError("Static preprocessing produced nonfinite float32 values")
        return dict(zip(patient_ids, result))

    def to_dict(self):
        return {"columns": self.columns, "medians": self.medians,
                "means": self.means, "scales": self.scales}


class NoteData:
    """Memory-mapped vectors and deterministic per-patient metadata indices."""

    def __init__(self, embeddings, notes, patient_ids, outcomes=None, splits=None,
                 patients=None, fingerprint="", embedding_model_id="unspecified"):
        self.embeddings = embeddings
        self.notes = notes.sort_values(["patient_id", "days_from_index", "note_id"],
                                     kind="mergesort").reset_index(drop=True)
        self.patient_ids = sorted(patient_ids)
        self.outcomes = outcomes.set_index("patient_id") if outcomes is not None else None
        self.splits = splits
        self.patients = patients
        self.fingerprint = fingerprint
        self.embedding_model_id = embedding_model_id
        self.input_dim = embeddings.shape[1]
        self._histories = {}
        for patient, frame in self.notes.groupby("patient_id", sort=False):
            self._histories[patient] = (frame["days_from_index"].to_numpy(dtype=np.float64),
                                        frame["embedding_row"].to_numpy(dtype=np.int64))

    def split_ids(self, split):
        if self.splits is None:
            raise ValueError("This operation requires patient-level splits")
        return sorted(self.splits.loc[self.splits["split"] == split, "patient_id"].tolist())

    def history(self, patient_id, landmark, max_notes):
        days, rows = self._histories.get(patient_id, (np.empty(0), np.empty(0, dtype=int)))
        end = np.searchsorted(days, landmark, side="right")
        start = max(0, end - max_notes)  # cutoff FIRST, then most recent history
        kept_days = days[start:end].copy()
        vectors = np.array(self.embeddings[rows[start:end]], dtype=np.float32, copy=True)
        return vectors, kept_days

    def batch(self, patient_ids, landmark, config, static_transform, device="cpu"):
        histories = [self.history(p, landmark, config.max_notes) for p in patient_ids]
        length = max(1, max((len(days) for _, days in histories), default=0))
        x = np.zeros((len(patient_ids), length, self.input_dim), dtype=np.float32)
        # float64 raw timestamps avoid collapsing fractional cutoff boundaries.
        days = np.zeros((len(patient_ids), length), dtype=np.float64)
        mask = np.zeros((len(patient_ids), length), dtype=bool)
        counts = []
        for row, (vectors, timestamps) in enumerate(histories):
            n = len(timestamps)
            x[row, :n] = vectors
            days[row, :n] = timestamps
            mask[row, :n] = True
            counts.append(n)
        static = static_transform.transform(self.patients, patient_ids)
        static_array = np.stack([static[p] for p in patient_ids]) if static_transform.output_dim else None
        return {"x_notes": torch.as_tensor(x, device=device),
                "days": torch.as_tensor(days, device=device),
                "landmark": torch.full((len(patient_ids),), float(landmark), dtype=torch.float64, device=device),
                "attention_mask": torch.as_tensor(mask, device=device),
                "static_covariates": torch.as_tensor(static_array, device=device) if static_array is not None else None}, counts


def load_data(note_embeddings_path, note_metadata_path, outcomes_path=None,
              splits_path=None, patients_path=None, patient_ids_path=None,
              embedding_model_id="unspecified"):
    paths = {"embeddings": note_embeddings_path, "notes": note_metadata_path}
    notes = _read_table(note_metadata_path, ["patient_id", "note_id", "days_from_index", "embedding_row"])
    if "embedding_model_id" in notes:
        encoder_ids = notes["embedding_model_id"].dropna().astype(str).unique()
        if notes["embedding_model_id"].isna().any() or len(encoder_ids) != 1:
            raise ValueError("Note vectors must use one common embedding model/version")
        if embedding_model_id not in ("unspecified", encoder_ids[0]):
            raise ValueError("Embedding metadata disagrees with --embedding_model_id")
        embedding_model_id = encoder_ids[0]
    if notes.duplicated(["patient_id", "note_id"]).any():
        raise ValueError("Duplicate patient/note IDs")
    notes["days_from_index"] = _numeric(notes["days_from_index"], "days_from_index")
    rows = _numeric(notes["embedding_row"], "embedding_row")
    if not np.equal(rows, np.floor(rows)).all():
        raise ValueError("embedding_row must be an integer")
    embeddings = np.load(note_embeddings_path, mmap_mode="r", allow_pickle=False)
    if embeddings.dtype != np.float32 or embeddings.ndim != 2 or embeddings.shape[1] < 1:
        raise ValueError("notes.npy must be float32 [N_notes, D] with D > 0")
    if len(notes) != embeddings.shape[0] or set(rows) != set(range(len(notes))):
        raise ValueError("embedding_row must be a permutation of all notes.npy rows")
    notes["embedding_row"] = rows.astype(np.int64)
    for start in range(0, len(embeddings), 4096):
        if not np.isfinite(embeddings[start:start + 4096]).all():
            raise ValueError("Note embeddings must be finite")
    patients = _read_table(patients_path, ["patient_id"]) if patients_path else None
    if patients is not None:
        paths["patients"] = patients_path
        if patients["patient_id"].duplicated().any():
            raise ValueError("Duplicate patient rows")
        for column in patients.columns:
            if column != "patient_id":
                _numeric(patients[column], column, True)
    ids = set(notes["patient_id"])
    if patients is not None:
        ids.update(patients["patient_id"])
    if patient_ids_path:
        ids_table = _read_table(patient_ids_path, ["patient_id"])
        if ids_table["patient_id"].duplicated().any():
            raise ValueError("Duplicate prediction patient IDs")
        ids.update(ids_table["patient_id"])
        paths["patient_ids"] = patient_ids_path
    outcomes = None
    splits = None
    if outcomes_path:
        paths["outcomes"] = outcomes_path
        outcomes = _read_table(outcomes_path, ["patient_id", "time", "event"])
        unexpected = set(outcomes.columns) - {"patient_id", "time", "event", "death_event"}
        if unexpected:
            raise ValueError(f"Unsupported outcome columns {sorted(unexpected)}; delayed entry and recurrent events are unsupported")
        if outcomes["patient_id"].duplicated().any():
            raise ValueError("Outcomes require exactly one first-event/censoring row per patient")
        outcomes["time"] = _numeric(outcomes["time"], "time")
        outcomes["event"] = _binary(outcomes["event"], "event")
        outcomes["death_event"] = _binary(outcomes["death_event"], "death_event") if "death_event" in outcomes else 0
        if ((outcomes["event"] + outcomes["death_event"]) > 1).any():
            raise ValueError("Primary and death cause indicators must be mutually exclusive")
        outcome_ids = set(outcomes["patient_id"])
        if ids - outcome_ids:
            raise ValueError("Note/patient metadata contains patients absent from outcomes")
        ids = outcome_ids
    if splits_path:
        paths["splits"] = splits_path
        splits = _read_table(splits_path, ["patient_id", "split"])
        if splits["patient_id"].duplicated().any():
            raise ValueError("Patient split isolation violated: duplicate patient assignments")
        if not splits["split"].isin(["train", "val", "test"]).all():
            raise ValueError("split must be train, val or test")
        if set(splits["patient_id"]) != ids:
            raise ValueError("Splits must cover the patient population exactly")
        if outcomes is None:
            raise ValueError("Training splits require outcomes")
        if not {"train", "val", "test"}.issubset(set(splits["split"])):
            raise ValueError("Each split must contain at least one patient")
    if not ids:
        raise ValueError("No patients; use patients.csv or patient_ids_path for patients with no notes")
    fingerprint = digest({"schema": 1, "files": {name: file_digest(path) for name, path in paths.items()},
                          "embedding_model_id": embedding_model_id})
    return NoteData(embeddings, notes, ids, outcomes, splits, patients,
                    fingerprint, embedding_model_id)
