"""Prepare the note-level inputs for the note-survival transformer (overall survival).

Maps the repo's note store and death_met survival cohort onto the transformer
package's five-file contract (see ``pipelines/note_transformer/docs/DATA_FORMAT.md``):

    notes.npy, notes.csv, outcomes.csv, splits.csv, [patients.csv], prepare_manifest.json

Cohorts
-------
text  Replays the Cox text-model cohort (embedding prediction df joined to the cancer-type
      df, ``tt_death > 0``, finite features) in Cox row order, so the 20% test split can be
      reproduced exactly (``train_test_split(test_size=0.2, stratify=event, random_state=1234)``)
      and verified against ``full_cohort/death/text_ipcw_reference.csv.gz``.
all   Every ``death_met_surv_df`` patient with a valid ``tt_death``; uses its own split.

Note availability
-----------------
``days_from_index`` is fractional elapsed days computed from the raw ``NOTE_DATETIME`` and the
anchor date (``FIRST_TREATMENT_START_DT`` or ``SEQUENCING_DT``) in microseconds, never from the
integer ``NOTE_TIME_REL_*`` columns (those truncate post-landmark hours into the landmark day).
The anchor columns are date-only, so same-day notes with any time of day land at a positive
fractional offset; no within-day availability claim is made.

Usage (from the repo root):

    python -m pipelines.note_transformer.prepare_os_inputs --anchor treatment
    python -m pipelines.note_transformer.prepare_os_inputs --max-patients 300   # smoke set
"""

import argparse
import ast
import json
import os
import shutil
import struct
import sys
import time

import numpy as np
import polars as pl
import zstandard as zstd
from sklearn.model_selection import train_test_split

from anchors import ANCHORS, DEFAULT_ANCHOR, age_col
from config import NOTES_PATH, SURV_PATH
from schemes import embedding_file, full_cohort_event_dir, scheme_results_dir
from pipelines.training.slurm_array_utils import (
    build_full_prediction_df,
    filter_event_rows,
    load_cancer_type_df,
    validate_cox_inputs,
    _feature_file,
)

SCHEME = "death_met"
EVENT = "death"
TT_COL = "tt_death"
SCHEMA_VERSION = 1
COX_TEST_FRAC = 0.2
COX_TEST_SEED = 1234
NOTE_TYPES = ["Clinician", "Imaging", "Pathology"]
EMBEDDING_MODEL_ID = "Simonlee711/Clinical_ModernBERT"
MICROSECONDS_PER_DAY = 86_400_000_000.0
TIMESTAMP_POLICY = "fractional_days_from_raw_timestamps_v1"
ANCHOR_RAW_COL = {"treatment": "FIRST_TREATMENT_START_DT", "sequencing": "SEQUENCING_DT"}
NOTES_META_FILE = "full_clinical_notes_embeddings_metadata.parquet"
NOTES_ARRAY_FILE = "full_clinical_notes_embeddings_as_array.npy.zst"
MANIFEST_NAME = "prepare_manifest.json"
BASE_FILES = ["notes.npy", "notes.csv", "outcomes.csv", "splits.csv", MANIFEST_NAME]


# --------------------------------------------------------------------------- paths


def _suffixed_surv_file(anchor: str) -> str:
    return os.path.basename(embedding_file(SCHEME, anchor)).replace(
        "_embedding_prediction_df", "_surv_df"
    )


def surv_path(anchor: str) -> str:
    return os.path.join(SURV_PATH, _suffixed_surv_file(anchor))


def notes_meta_path() -> str:
    return os.path.join(NOTES_PATH, NOTES_META_FILE)


def notes_array_path() -> str:
    return os.path.join(NOTES_PATH, NOTES_ARRAY_FILE)


def ipcw_reference_path(anchor: str) -> str:
    return os.path.join(full_cohort_event_dir(SCHEME, EVENT, anchor), "text_ipcw_reference.csv.gz")


def default_inputs_dir(anchor: str, cohort: str, max_patients: int | None) -> str:
    name = "inputs" if cohort == "text" else "inputs_all"
    if max_patients:
        name = f"inputs_smoke{max_patients}" if cohort == "text" else f"inputs_all_smoke{max_patients}"
    return os.path.join(scheme_results_dir(SCHEME, anchor), "note_transformer", name)


# --------------------------------------------------------------------------- cohorts


def build_cox_text_cohort(anchor: str) -> tuple[pl.DataFrame, list[str]]:
    """Replay the Cox text-model cohort in Cox row order.

    Mirrors ``pipelines/training/run_full_cohort_event.py``: build the prediction df,
    keep valid ``tt_death`` rows, drop constant columns, then require all features finite.
    Returns ``(cohort, static_columns)`` with ``DFCI_MRN, death, tt_death`` plus the static
    covariates that survived the constant-column filter.
    """
    full_df, type_cols, embed_cols, _ = build_full_prediction_df(SCHEME, anchor)
    df = filter_event_rows(full_df, EVENT)
    base_vars = ["GENDER", age_col(anchor)]
    feature_cols = base_vars + type_cols + embed_cols
    df, dropped = validate_cox_inputs(df, EVENT, TT_COL, feature_cols, label=f"{SCHEME}:{EVENT}")
    feature_cols = [c for c in feature_cols if c not in dropped]
    required = feature_cols + [EVENT, TT_COL]
    df = df.filter(pl.all_horizontal([pl.col(c).cast(pl.Float64, strict=False).is_finite() for c in required]))
    static_cols = [c for c in base_vars + type_cols if c in df.columns and c not in dropped]
    return df.select(["DFCI_MRN", EVENT, TT_COL, *static_cols]), static_cols


def build_all_cohort(anchor: str) -> tuple[pl.DataFrame, list[str]]:
    """Every death_met_surv_df patient with a valid tt_death; cancer types left-joined."""
    surv = pl.read_parquet(surv_path(anchor))
    df = filter_event_rows(surv, EVENT)
    cancer_type_df, type_cols = load_cancer_type_df(anchor)
    df = df.join(cancer_type_df.select(["DFCI_MRN"] + type_cols), on="DFCI_MRN", how="left")
    static_cols = [c for c in ["GENDER", age_col(anchor), *type_cols] if c in df.columns]
    return df.select(["DFCI_MRN", EVENT, TT_COL, *static_cols]), static_cols


def subsample_cohort(cohort: pl.DataFrame, n: int, seed: int) -> pl.DataFrame:
    """Deterministic event-stratified subsample for smoke runs (cohort order preserved)."""
    if n >= cohort.height:
        return cohort
    idx = np.arange(cohort.height)
    keep, _ = train_test_split(
        idx, train_size=n, stratify=cohort[EVENT].cast(pl.Int64).to_numpy(), random_state=seed
    )
    return cohort[np.sort(keep)]


def split_patients(cohort: pl.DataFrame, val_frac: float, seed: int) -> dict[str, np.ndarray]:
    """Cox test split first (fixed 20%, seed 1234), then val carved from train+val."""
    idx = np.arange(cohort.height)
    event = cohort[EVENT].cast(pl.Int64).to_numpy()
    idx_trainval, idx_test = train_test_split(
        idx, test_size=COX_TEST_FRAC, stratify=event, random_state=COX_TEST_SEED
    )
    idx_train, idx_val = train_test_split(
        idx_trainval, test_size=val_frac, stratify=event[idx_trainval], random_state=seed
    )
    return {"train": idx_train, "val": idx_val, "test": idx_test}


def cox_trainval_ids(anchor: str) -> set[int] | None:
    """Train+val MRNs recorded by the Cox run (test row ``reference_ids``), or None if absent."""
    path = ipcw_reference_path(anchor)
    if not os.path.exists(path):
        return None
    ref = pl.read_csv(path, infer_schema_length=0)
    row = ref.filter(pl.col("eval_data") == "test")
    if row.height != 1:
        raise ValueError(f"{path}: expected exactly one eval_data=='test' row, found {row.height}")
    return {int(x) for x in json.loads(row["reference_ids"][0])}


def assert_split_matches_cox(
    cohort: pl.DataFrame, split_idx: dict[str, np.ndarray], anchor: str, allow_mismatch: bool
) -> str:
    """Return the split_check status; raise on a mismatch unless allowed."""
    ref = cox_trainval_ids(anchor)
    if ref is None:
        print(f"[note] {ipcw_reference_path(anchor)} not found; Cox split not verified.")
        return "reference_absent"
    mrns = cohort["DFCI_MRN"].to_numpy()
    ours = {int(m) for m in mrns[np.concatenate([split_idx["train"], split_idx["val"]])]}
    if ours == ref:
        return "matched_cox_ipcw_reference"
    msg = (
        f"Train+val split does not match the Cox IPCW reference: ours={len(ours)}, "
        f"reference={len(ref)}, only_ours={len(ours - ref)}, only_reference={len(ref - ours)}. "
        "The cohort replay or the polars join order has drifted from the Cox run."
    )
    if allow_mismatch:
        print(f"[warn] {msg} (--allow-split-mismatch)")
        return "mismatch_allowed"
    raise ValueError(msg + " Pass --allow-split-mismatch to proceed without the identical-test claim.")


# --------------------------------------------------------------------------- notes


def load_note_metadata(anchor: str, mrns: list[int]) -> tuple[pl.DataFrame, dict]:
    """Note rows for the cohort with fractional ``days_from_index`` (float64)."""
    path = notes_meta_path()
    anchor_raw = ANCHOR_RAW_COL[anchor]
    needed = ["DFCI_MRN", "NOTE_TYPE", "EMBEDDING_INDEX", "NOTE_DATETIME", anchor_raw]
    schema = pl.read_parquet_schema(path)
    missing = [c for c in needed if c not in schema]
    if missing:
        raise ValueError(
            f"{path} is missing raw timestamp/index columns {missing}. Regenerate the note metadata "
            "with pipelines.preprocessing.knit_embeddings; the integer NOTE_TIME_REL_* columns are "
            "deliberately not used as a fallback because they truncate to whole days."
        )
    meta = (
        pl.scan_parquet(path)
        .select(needed)
        .with_columns(pl.col("DFCI_MRN").cast(pl.Int64, strict=False))
        .filter(pl.col("NOTE_TYPE").is_in(NOTE_TYPES) & pl.col("DFCI_MRN").is_in(mrns))
        .collect()
    )
    n_selected = meta.height
    meta = meta.filter(pl.col("NOTE_DATETIME").is_not_null() & pl.col(anchor_raw).is_not_null())
    n_dropped = n_selected - meta.height
    if n_dropped:
        print(f"[note] dropped {n_dropped:,}/{n_selected:,} cohort notes with a missing timestamp")
    meta = meta.with_columns(
        (
            (pl.col("NOTE_DATETIME").cast(pl.Datetime("us")) - pl.col(anchor_raw).cast(pl.Datetime("us")))
            .dt.total_microseconds()
            .cast(pl.Float64)
            / MICROSECONDS_PER_DAY
        ).alias("days_from_index")
    ).sort("EMBEDDING_INDEX")
    if meta["EMBEDDING_INDEX"].n_unique() != meta.height:
        raise ValueError("Duplicate EMBEDDING_INDEX values in the note metadata")
    return meta.select(["DFCI_MRN", "EMBEDDING_INDEX", "days_from_index"]), {
        "dropped_missing_timestamp": int(n_dropped)
    }


def load_zst_npy_readonly(path: str) -> np.ndarray:
    """Decompress a zstd-compressed ``.npy`` once and view it without a second copy."""
    with open(path, "rb") as handle:
        raw = zstd.decompress(handle.read())
    if raw[:6] != b"\x93NUMPY":
        raise ValueError(f"{path}: not a .npy payload")
    major = raw[6]
    if major == 1:
        (hlen,) = struct.unpack("<H", raw[8:10])
        start = 10
    else:
        (hlen,) = struct.unpack("<I", raw[8:12])
        start = 12
    header = ast.literal_eval(raw[start : start + hlen].decode("latin1"))
    if header.get("fortran_order"):
        raise ValueError(f"{path}: Fortran-ordered arrays are not supported")
    shape = tuple(header["shape"])
    dtype = np.dtype(header["descr"])
    count = int(np.prod(shape)) if shape else 1
    return np.frombuffer(raw, dtype=dtype, count=count, offset=start + hlen).reshape(shape)


def write_cohort_embeddings(src: np.ndarray, source_rows: np.ndarray, out_path: str, chunk_rows: int) -> None:
    """Chunked float32 copy of the selected rows into ``out_path`` (atomic via os.replace)."""
    if chunk_rows < 1:
        raise ValueError("chunk_rows must be positive")
    n = len(source_rows)
    if n == 0:
        raise ValueError("No notes selected for the cohort; nothing to write")
    if int(source_rows.max()) >= src.shape[0] or int(source_rows.min()) < 0:
        raise ValueError(
            f"EMBEDDING_INDEX out of range for the embedding array ({src.shape[0]:,} rows)"
        )
    tmp = out_path + ".partial"
    out = np.lib.format.open_memmap(tmp, mode="w+", dtype=np.float32, shape=(n, src.shape[1]))
    for start in range(0, n, chunk_rows):
        block = src[source_rows[start : start + chunk_rows]].astype(np.float32)
        if not np.isfinite(block).all():
            del out
            os.remove(tmp)
            raise ValueError("Non-finite values in the selected note embeddings")
        out[start : start + len(block)] = block
    out.flush()
    del out
    os.replace(tmp, out_path)


def write_tables(
    out_dir: str,
    cohort: pl.DataFrame,
    split_idx: dict[str, np.ndarray],
    notes: pl.DataFrame,
    static_cols: list[str],
    static: bool,
) -> None:
    pid = pl.col("DFCI_MRN").cast(pl.Int64).cast(pl.String).alias("patient_id")
    notes.select(
        pl.col("DFCI_MRN").cast(pl.Int64).cast(pl.String).alias("patient_id"),
        pl.col("EMBEDDING_INDEX").cast(pl.Int64).cast(pl.String).alias("note_id"),
        pl.col("days_from_index").cast(pl.Float64),
        pl.int_range(pl.len(), dtype=pl.Int64).alias("embedding_row"),
        pl.lit(EMBEDDING_MODEL_ID).alias("embedding_model_id"),
    ).write_csv(os.path.join(out_dir, "notes.csv"))
    cohort.select(
        pid, pl.col(TT_COL).cast(pl.Float64).alias("time"), pl.col(EVENT).cast(pl.Int64).alias("event")
    ).write_csv(os.path.join(out_dir, "outcomes.csv"))
    labels = np.empty(cohort.height, dtype=object)
    for name, idx in split_idx.items():
        labels[idx] = name
    cohort.select(pid).with_columns(pl.Series("split", labels.tolist(), dtype=pl.String)).write_csv(
        os.path.join(out_dir, "splits.csv")
    )
    if static:
        cohort.select(pid, *[pl.col(c).cast(pl.Float64) for c in static_cols]).write_csv(
            os.path.join(out_dir, "patients.csv")
        )


# --------------------------------------------------------------------------- signature / manifest


def _file_sig(path: str | None) -> dict | None:
    if path is None or not os.path.exists(path):
        return None
    st = os.stat(path)
    return {"path": os.path.abspath(path), "size": st.st_size, "mtime_ns": st.st_mtime_ns}


def preparation_signature(args, anchor: str) -> dict:
    """Everything that changes the data semantics (chunk size deliberately excluded)."""
    sources = {
        "notes_metadata": _file_sig(notes_meta_path()),
        "notes_array": _file_sig(notes_array_path()),
        "surv_df": _file_sig(surv_path(anchor)),
        "cancer_type_df": _file_sig(_feature_file("cancer_type_df.csv.gz", anchor)),
        "ipcw_reference": _file_sig(ipcw_reference_path(anchor)),
    }
    if args.cohort == "text":
        sources["embedding_prediction_df"] = _file_sig(os.path.join(SURV_PATH, embedding_file(SCHEME, anchor)))
    sig = {
        "schema_version": SCHEMA_VERSION,
        "anchor": anchor,
        "cohort": args.cohort,
        "static": bool(args.static),
        "val_frac": args.val_frac,
        "val_seed": args.seed,
        "test_frac": COX_TEST_FRAC,
        "test_seed": COX_TEST_SEED,
        "max_patients": args.max_patients,
        "allow_split_mismatch": bool(args.allow_split_mismatch),
        "timestamp_policy": TIMESTAMP_POLICY,
        "note_types": NOTE_TYPES,
        "embedding_model_id": EMBEDDING_MODEL_ID,
        "sources": sources,
    }
    return json.loads(json.dumps(sig))


def expected_files(static: bool) -> list[str]:
    return sorted(BASE_FILES + (["patients.csv"] if static else []))


def _diff(old, new, prefix=""):
    diffs = []
    if isinstance(old, dict) and isinstance(new, dict):
        for key in sorted(set(old) | set(new)):
            diffs += _diff(old.get(key), new.get(key), f"{prefix}{key}.")
    elif old != new:
        diffs.append(f"{prefix[:-1]}: {old!r} -> {new!r}")
    return diffs


def validate_existing_inputs(inputs_dir: str, signature: dict) -> list[str]:
    """Return the reasons the existing bundle cannot be reused (empty list = reuse)."""
    if not os.path.isdir(inputs_dir):
        return ["inputs directory does not exist"]
    manifest_path = os.path.join(inputs_dir, MANIFEST_NAME)
    if not os.path.exists(manifest_path):
        return [f"{MANIFEST_NAME} missing"]
    try:
        with open(manifest_path) as handle:
            manifest = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        return [f"{MANIFEST_NAME} unreadable: {exc}"]
    problems = _diff(manifest.get("signature"), signature, "signature.")
    want = expected_files(signature["static"])
    have = sorted(f for f in os.listdir(inputs_dir))
    if have != want:
        problems.append(f"file set {have} != expected {want}")
    return problems


def write_manifest(
    out_dir: str,
    signature: dict,
    cohort: pl.DataFrame,
    split_idx: dict[str, np.ndarray],
    notes: pl.DataFrame,
    note_counts: dict,
    static_cols: list[str],
    split_check: str,
) -> dict:
    n_notes = notes.height
    days = notes["days_from_index"]
    per_split = {}
    event = cohort[EVENT].cast(pl.Int64).to_numpy()
    for name, idx in split_idx.items():
        per_split[name] = {"patients": int(len(idx)), "events": int(event[idx].sum())}
    patients_with_notes = notes["DFCI_MRN"].n_unique()
    manifest = {
        "signature": signature,
        "expected_files": expected_files(signature["static"]),
        "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "counts": {
            "patients": cohort.height,
            "events": int(event.sum()),
            "patients_with_notes": int(patients_with_notes),
            "patients_without_notes": int(cohort.height - patients_with_notes),
            "notes": n_notes,
            "notes_exactly_at_anchor": int((days == 0).sum()),
            "notes_within_first_day_after_anchor": int(((days > 0) & (days < 1)).sum()),
            "notes_after_anchor": int((days > 0).sum()),
            "notes_before_anchor": int((days < 0).sum()),
            **note_counts,
            "per_split": per_split,
        },
        "static": {"enabled": signature["static"], "columns": static_cols if signature["static"] else []},
        "timestamp_policy": {
            "policy": TIMESTAMP_POLICY,
            "derivation": "(NOTE_DATETIME - anchor date) in microseconds / 86_400_000_000, float64",
            "inclusive_cutoff": "a note is available at landmark L iff days_from_index <= L",
            "caveat": "anchor dates are date-only; same-day notes with any time of day fall at a "
            "positive fractional offset and are excluded at landmark 0",
        },
        "split_check": split_check,
        "cohort_rule": (
            "Cox text-model cohort replay (embedding prediction df + cancer types, tt_death>0, finite features)"
            if signature["cohort"] == "text"
            else "all death_met_surv_df patients with valid tt_death"
        ),
        "embedding_model_id": EMBEDDING_MODEL_ID,
    }
    with open(os.path.join(out_dir, MANIFEST_NAME), "w") as handle:
        json.dump(manifest, handle, indent=2)
        handle.write("\n")
    return manifest


# --------------------------------------------------------------------------- orchestration


def _swap_in(staging: str, target: str) -> None:
    """Replace ``target`` with ``staging``, restoring the old bundle if the swap fails."""
    backup = None
    if os.path.exists(target):
        backup = f"{target}.backup-{os.getpid()}"
        os.rename(target, backup)
    try:
        os.rename(staging, target)
    except OSError:
        if backup is not None:
            os.rename(backup, target)
        raise
    if backup is not None:
        shutil.rmtree(backup, ignore_errors=True)


def _parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--anchor", default=DEFAULT_ANCHOR, choices=sorted(ANCHORS))
    p.add_argument("--cohort", default="text", choices=["text", "all"])
    p.add_argument("--static", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--val-frac", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=1234, help="seed for the train/val carve")
    p.add_argument("--inputs-dir", default=None)
    p.add_argument("--max-patients", type=int, default=None, help="smoke subset (skips the Cox split check)")
    p.add_argument("--chunk-rows", type=int, default=65536)
    p.add_argument("--allow-split-mismatch", action="store_true")
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args(argv)
    if args.chunk_rows < 1:
        p.error("--chunk-rows must be positive")
    if not 0.0 < args.val_frac < 1.0:
        p.error("--val-frac must be in (0, 1)")
    if args.max_patients is not None and args.max_patients < 20:
        p.error("--max-patients must be at least 20")
    return args


def _validate_bundle(out_dir: str, static: bool, cohort: str) -> None:
    from pipelines.note_transformer.note_survival_transformer.dataset import load_data

    load_data(
        os.path.join(out_dir, "notes.npy"),
        os.path.join(out_dir, "notes.csv"),
        outcomes_path=os.path.join(out_dir, "outcomes.csv"),
        splits_path=os.path.join(out_dir, "splits.csv"),
        patients_path=os.path.join(out_dir, "patients.csv") if static else None,
        embedding_model_id=EMBEDDING_MODEL_ID,
    )


def main(argv=None) -> int:
    args = _parse_args(argv)
    anchor = args.anchor
    inputs_dir = os.path.abspath(args.inputs_dir or default_inputs_dir(anchor, args.cohort, args.max_patients))
    signature = preparation_signature(args, anchor)

    if os.path.exists(inputs_dir) and not args.overwrite:
        problems = validate_existing_inputs(inputs_dir, signature)
        if not problems:
            print(f"[skip] Inputs already prepared and up to date: {inputs_dir}")
            return 0
        raise SystemExit(
            f"[error] {inputs_dir} exists but cannot be reused:\n  - "
            + "\n  - ".join(problems)
            + "\nPass --overwrite to rebuild it, or use a different --inputs-dir."
        )

    t0 = time.time()
    staging = f"{inputs_dir}.staging-{os.getpid()}"
    if os.path.exists(staging):
        shutil.rmtree(staging)
    os.makedirs(staging)
    try:
        cohort, static_cols = (
            build_cox_text_cohort(anchor) if args.cohort == "text" else build_all_cohort(anchor)
        )
        print(f"[prepare] cohort={args.cohort} anchor={anchor}: {cohort.height:,} patients")
        smoke = args.max_patients is not None
        if smoke:
            cohort = subsample_cohort(cohort, args.max_patients, args.seed)
            print(f"[prepare] smoke subset: {cohort.height:,} patients")
        split_idx = split_patients(cohort, args.val_frac, args.seed)
        if smoke:
            split_check = "skipped_smoke"
        elif args.cohort == "all":
            split_check = "not_applicable_all_cohort"
        else:
            split_check = assert_split_matches_cox(cohort, split_idx, anchor, args.allow_split_mismatch)

        notes, note_counts = load_note_metadata(anchor, cohort["DFCI_MRN"].cast(pl.Int64).to_list())
        print(f"[prepare] {notes.height:,} notes for {notes['DFCI_MRN'].n_unique():,} patients")
        src = load_zst_npy_readonly(notes_array_path())
        write_cohort_embeddings(
            src, notes["EMBEDDING_INDEX"].to_numpy(), os.path.join(staging, "notes.npy"), args.chunk_rows
        )
        del src
        write_tables(staging, cohort, split_idx, notes, static_cols, args.static)
        write_manifest(staging, signature, cohort, split_idx, notes, note_counts, static_cols, split_check)
        _validate_bundle(staging, args.static, args.cohort)
        _swap_in(staging, inputs_dir)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    print(f"[done] Prepared {inputs_dir} in {time.time() - t0:.0f}s (split_check={split_check})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
