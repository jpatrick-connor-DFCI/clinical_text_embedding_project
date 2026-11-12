"""Input preparation for the note-survival transformer: contracts, split replay, timestamps, skips."""

import gzip
import io
import json
import os
import subprocess
import sys
from datetime import datetime, timedelta

import numpy as np
import polars as pl
import pytest
import zstandard as zstd
from sklearn.model_selection import train_test_split

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import schemes  # noqa: E402
from pipelines.note_transformer import prepare_os_inputs as prep  # noqa: E402
from pipelines.note_transformer.note_survival_transformer.dataset import StaticTransform, load_data  # noqa: E402
from pipelines.training import slurm_array_utils as su  # noqa: E402

US_PER_DAY = 86_400_000_000
LANDMARKS = (0, 90, 180, 365)
N_PATIENTS = 160
MRNS = list(range(5000, 5000 + N_PATIENTS))
NO_CANCER_TYPE_MRN = MRNS[7]
BOUNDARY_TREATMENT_MRN = MRNS[0]  # notes placed around landmarks relative to FIRST_TREATMENT_START_DT
BOUNDARY_SEQUENCING_MRN = MRNS[2]  # ... relative to SEQUENCING_DT
NO_NOTES_MRN = MRNS[1]
SEQ_OFFSET_DAYS = 10


def boundary_offsets_us():
    offsets = [-12 * 3600 * 10**6, 0, 12 * 3600 * 10**6]
    for landmark in LANDMARKS[1:]:
        base = landmark * US_PER_DAY
        offsets += [base - 3600 * 10**6, base, base + 1, base + 12 * 3600 * 10**6]
    return offsets


def _gz_csv(path, frame):
    with gzip.open(path, "wt") as handle:
        handle.write(frame.write_csv())


def build_store(root, drop_raw_anchor=False):
    rng = np.random.default_rng(1234)
    n = N_PATIENTS
    event = np.zeros(n, dtype=int)
    event[: int(n * 0.55)] = 1
    rng.shuffle(event)
    surv = pl.DataFrame({
        "DFCI_MRN": MRNS,
        "GENDER": rng.integers(0, 2, n).astype(float),
        "AGE_AT_TREATMENTSTART": rng.normal(60, 10, n),
        "AGE_AT_SEQUENCING": rng.normal(61, 10, n),
        "death": event,
        "tt_death": rng.exponential(500, n) + 5,
    })
    emb = surv.with_columns(
        pl.Series("EMBEDDING_0", rng.normal(size=n)), pl.Series("EMBEDDING_1", rng.normal(size=n))
    )
    cancer = pl.DataFrame({
        "DFCI_MRN": MRNS,
        "CANCER_TYPE_LUNG": rng.integers(0, 2, n),
        "CANCER_TYPE_BREAST": rng.integers(0, 2, n),
    }).filter(pl.col("DFCI_MRN") != NO_CANCER_TYPE_MRN)

    surv_dir = root / "time-to-event_analysis"
    feat_dir = root / "clinical_and_genomic_features"
    notes_dir = root / "batched_datasets" / "processed_datasets"
    for d in (surv_dir, feat_dir, notes_dir):
        d.mkdir(parents=True)
    for suffix in ("", "__sequencing"):
        surv.write_parquet(surv_dir / f"death_met_surv_df{suffix}.parquet")
        emb.write_parquet(surv_dir / f"death_met_embedding_prediction_df{suffix}.parquet")
        _gz_csv(feat_dir / f"cancer_type_df{suffix}.csv.gz", cancer)

    rows = []
    for i, mrn in enumerate(MRNS):
        treat = datetime(2018, 1, 1) + timedelta(days=i)
        seq = treat + timedelta(days=SEQ_OFFSET_DAYS)
        if mrn == NO_NOTES_MRN:
            continue
        if mrn in (BOUNDARY_TREATMENT_MRN, BOUNDARY_SEQUENCING_MRN):
            anchor = treat if mrn == BOUNDARY_TREATMENT_MRN else seq
            stamps = [(anchor + timedelta(microseconds=int(us)), "Clinician") for us in boundary_offsets_us()]
        else:
            k = int(rng.integers(0, 7))
            types = rng.choice(["Clinician", "Imaging", "Pathology", "Telephone"], k)
            stamps = [(treat + timedelta(microseconds=int(rng.uniform(-30, 400) * US_PER_DAY)), str(t))
                      for t in types]
        for stamp, note_type in stamps:
            rows.append({"DFCI_MRN": mrn, "NOTE_TYPE": note_type, "NOTE_DATETIME": stamp,
                         "FIRST_TREATMENT_START_DT": treat, "SEQUENCING_DT": seq})
    meta = pl.DataFrame(rows, schema_overrides={
        "NOTE_DATETIME": pl.Datetime("us"), "FIRST_TREATMENT_START_DT": pl.Datetime("us"),
        "SEQUENCING_DT": pl.Datetime("us")})
    total = meta.height
    vectors = rng.normal(size=(total, 4)).astype(np.float16)
    meta = meta.with_columns(pl.int_range(pl.len()).alias("EMBEDDING_INDEX"))
    meta = meta.with_columns(  # legacy integer relative-day columns, deliberately unused
        (pl.col("NOTE_DATETIME") - pl.col("FIRST_TREATMENT_START_DT")).dt.total_days().alias("NOTE_TIME_REL_TREATMENT")
    ).sample(fraction=1.0, shuffle=True, seed=3)  # EMBEDDING_INDEX stays tied to the array row
    if drop_raw_anchor:
        meta = meta.drop("FIRST_TREATMENT_START_DT")
    meta.write_parquet(notes_dir / "full_clinical_notes_embeddings_metadata.parquet")
    buffer = io.BytesIO()
    np.save(buffer, vectors)
    (notes_dir / "full_clinical_notes_embeddings_as_array.npy.zst").write_bytes(zstd.compress(buffer.getvalue()))
    return {"meta": meta, "vectors": vectors, "surv": str(surv_dir) + "/", "feat": str(feat_dir) + "/",
            "notes": str(notes_dir) + "/"}


@pytest.fixture
def store(tmp_path, monkeypatch):
    info = build_store(tmp_path / "data")
    monkeypatch.setattr(prep, "SURV_PATH", info["surv"])
    monkeypatch.setattr(prep, "NOTES_PATH", info["notes"])
    monkeypatch.setattr(schemes, "SURV_PATH", info["surv"])
    monkeypatch.setattr(su, "FEATURE_PATH", info["feat"])
    info["tmp"] = tmp_path
    info["inputs"] = str(tmp_path / "inputs")
    return info


def run(store, *args):
    return prep.main(["--inputs-dir", store["inputs"], *args])


def load_bundle(inputs, static=True):
    return load_data(
        os.path.join(inputs, "notes.npy"), os.path.join(inputs, "notes.csv"),
        outcomes_path=os.path.join(inputs, "outcomes.csv"), splits_path=os.path.join(inputs, "splits.csv"),
        patients_path=os.path.join(inputs, "patients.csv") if static else None,
        embedding_model_id=prep.EMBEDDING_MODEL_ID)


def manifest(inputs):
    with open(os.path.join(inputs, "prepare_manifest.json")) as handle:
        return json.load(handle)


def cox_replay(anchor="treatment"):
    """The Cox cohort + split, replayed independently with the Cox functions."""
    df, _, _, _ = su.build_full_prediction_df("death_met", anchor)
    df = su.filter_event_rows(df, "death")
    idx_trainval, idx_test = train_test_split(
        np.arange(df.height), test_size=0.2, stratify=df["death"].to_numpy(), random_state=1234)
    mrns = df["DFCI_MRN"].to_numpy()
    return df, set(mrns[idx_trainval].tolist()), set(mrns[idx_test].tolist())


def write_reference(anchor, ids, path_dir=None):
    out_dir = schemes.full_cohort_event_dir("death_met", "death", anchor)
    os.makedirs(out_dir, exist_ok=True)
    ref = pl.DataFrame({"eval_data": ["test", "cv"], "fold": [None, 0],
                        "reference_ids": [json.dumps(sorted(ids)), json.dumps(sorted(ids)[:5])],
                        "reference_events": ["[]", "[]"], "reference_times": ["[]", "[]"]})
    _gz_csv(os.path.join(out_dir, "text_ipcw_reference.csv.gz"), ref)


# --------------------------------------------------------------------------- contract + split


def test_prepare_outputs_load_and_match_sources(store):
    assert run(store) == 0
    inputs = store["inputs"]
    assert sorted(os.listdir(inputs)) == sorted(
        ["notes.npy", "notes.csv", "outcomes.csv", "splits.csv", "patients.csv", "prepare_manifest.json"])
    data = load_bundle(inputs)
    outcomes = pl.read_csv(os.path.join(inputs, "outcomes.csv"))
    assert outcomes.columns == ["patient_id", "time", "event"]
    assert outcomes.height == N_PATIENTS - 1  # one MRN lacks a cancer type -> not in the Cox text cohort
    assert str(NO_CANCER_TYPE_MRN) not in set(map(str, outcomes["patient_id"].to_list()))
    patients = pl.read_csv(os.path.join(inputs, "patients.csv"))
    assert patients.columns == ["patient_id", "GENDER", "AGE_AT_TREATMENTSTART", "CANCER_TYPE_LUNG", "CANCER_TYPE_BREAST"]
    # notes.npy rows are the float32 source rows named by note_id (= EMBEDDING_INDEX)
    notes = pl.read_csv(os.path.join(inputs, "notes.csv"), schema_overrides={"patient_id": pl.String, "note_id": pl.String})
    assert notes["embedding_row"].to_list() == list(range(notes.height))
    assert (notes["embedding_model_id"] == prep.EMBEDDING_MODEL_ID).all()
    array = np.load(os.path.join(inputs, "notes.npy"), mmap_mode="r")
    assert array.dtype == np.float32
    for row, note_id in zip(notes["embedding_row"].to_list()[:50], notes["note_id"].to_list()[:50]):
        np.testing.assert_array_equal(array[row], store["vectors"][int(note_id)].astype(np.float32))
    # only Clinician/Imaging/Pathology notes of cohort patients
    meta = store["meta"].filter(pl.col("NOTE_TYPE").is_in(prep.NOTE_TYPES) & (pl.col("DFCI_MRN") != NO_CANCER_TYPE_MRN))
    assert notes.height == meta.height == manifest(inputs)["counts"]["notes"]
    assert data.input_dim == 4


def test_split_reproduces_cox_test_set_and_partitions_outcomes(store):
    run(store)
    df, expected_trainval, expected_test = cox_replay()
    splits = pl.read_csv(os.path.join(store["inputs"], "splits.csv"), schema_overrides={"patient_id": pl.String})
    by = {s: {int(p) for p in splits.filter(pl.col("split") == s)["patient_id"].to_list()} for s in ("train", "val", "test")}
    assert by["test"] == expected_test
    assert by["train"] | by["val"] == expected_trainval
    assert not (by["train"] & by["val"]) and not (by["train"] | by["val"]) & by["test"]
    assert len(by["val"]) == pytest.approx(0.2 * len(expected_trainval), abs=1.5)
    outcomes = pl.read_csv(os.path.join(store["inputs"], "outcomes.csv"), schema_overrides={"patient_id": pl.String})
    assert set(outcomes["patient_id"]) == set(splits["patient_id"]) == {str(m) for m in df["DFCI_MRN"].to_list()}
    # stratified by event: the split event rates track the cohort's
    rate = outcomes["event"].mean()
    for name in ("train", "val", "test"):
        part = outcomes.filter(pl.col("patient_id").is_in([str(m) for m in by[name]]))
        assert part["event"].mean() == pytest.approx(rate, abs=0.08)


def test_split_check_statuses(store):
    run(store)
    assert manifest(store["inputs"])["split_check"] == "reference_absent"
    _, trainval, _ = cox_replay()
    write_reference("treatment", trainval)
    run(store, "--overwrite")
    assert manifest(store["inputs"])["split_check"] == "matched_cox_ipcw_reference"


def test_mismatching_ipcw_reference_fails_the_text_cohort(store):
    _, trainval, _ = cox_replay()
    write_reference("treatment", set(sorted(trainval)[:-3]) | {999_999})
    with pytest.raises(ValueError, match="does not match the Cox IPCW reference"):
        run(store)
    assert not os.path.exists(store["inputs"])
    assert [p for p in os.listdir(store["tmp"]) if "staging" in p] == []
    run(store, "--allow-split-mismatch")
    assert manifest(store["inputs"])["split_check"] == "mismatch_allowed"


def test_smoke_and_all_cohort_statuses(store):
    run(store, "--max-patients", "40")
    m = manifest(store["inputs"])
    assert m["split_check"] == "skipped_smoke" and m["counts"]["patients"] == 40
    all_dir = str(store["tmp"] / "inputs_all")
    assert prep.main(["--inputs-dir", all_dir, "--cohort", "all"]) == 0
    m = manifest(all_dir)
    assert m["split_check"] == "not_applicable_all_cohort"
    assert m["counts"]["patients"] == N_PATIENTS  # includes the patient without a cancer type
    patients = pl.read_csv(os.path.join(all_dir, "patients.csv"))
    assert patients.filter(pl.col("patient_id") == NO_CANCER_TYPE_MRN)["CANCER_TYPE_LUNG"].null_count() == 1
    load_bundle(all_dir)  # missing static values are accepted (median impute + missing flags)


def test_patients_without_notes_are_kept_in_outcomes_and_splits(store):
    run(store)
    data = load_bundle(store["inputs"])
    assert str(NO_NOTES_MRN) in data.patient_ids
    vectors, days = data.history(str(NO_NOTES_MRN), 365.0, 128)
    assert len(days) == 0
    assert manifest(store["inputs"])["counts"]["patients_without_notes"] >= 1


# --------------------------------------------------------------------------- timestamps


def expected_days(meta_row, anchor_col):
    delta = meta_row["NOTE_DATETIME"] - meta_row[anchor_col]
    return (delta // timedelta(microseconds=1)) / US_PER_DAY


@pytest.mark.parametrize("anchor,anchor_col,mrn", [
    ("treatment", "FIRST_TREATMENT_START_DT", BOUNDARY_TREATMENT_MRN),
    ("sequencing", "SEQUENCING_DT", BOUNDARY_SEQUENCING_MRN),
])
def test_fractional_days_and_inclusive_landmark_cutoffs(store, anchor, anchor_col, mrn):
    run(store, "--anchor", anchor)
    data = load_bundle(store["inputs"])
    notes = pl.read_csv(os.path.join(store["inputs"], "notes.csv"),
                        schema_overrides={"patient_id": pl.String, "note_id": pl.String})
    meta = {r["EMBEDDING_INDEX"]: r for r in store["meta"].to_dicts()}
    for note_id, days in zip(notes["note_id"].to_list(), notes["days_from_index"].to_list()):
        assert days == pytest.approx(expected_days(meta[int(note_id)], anchor_col), abs=1e-12)
    offsets = sorted(boundary_offsets_us())
    for landmark in LANDMARKS:
        _, days = data.history(str(mrn), float(landmark), 10_000)
        expected = sum(1 for us in offsets if us <= landmark * US_PER_DAY)
        assert len(days) == expected, f"landmark {landmark}"
    # anchor +12h is NOT available at baseline; exactly at the anchor is
    _, baseline = data.history(str(mrn), 0.0, 10_000)
    assert sorted(baseline.tolist()) == [-0.5, 0.0]
    # one microsecond after a landmark is excluded and 1us before/at is included with float64 precision
    _, d90 = data.history(str(mrn), 90.0, 10_000)
    assert 90.0 in d90.tolist() and not any(90.0 < x < 91.0 for x in d90.tolist())
    assert any(abs(x - (90.0 + 1 / US_PER_DAY)) < 1e-13 for x in notes.filter(
        pl.col("patient_id") == str(mrn))["days_from_index"].to_list())


def test_missing_raw_timestamp_column_is_an_error_not_a_fallback(tmp_path, monkeypatch):
    info = build_store(tmp_path / "data", drop_raw_anchor=True)
    monkeypatch.setattr(prep, "SURV_PATH", info["surv"])
    monkeypatch.setattr(prep, "NOTES_PATH", info["notes"])
    monkeypatch.setattr(schemes, "SURV_PATH", info["surv"])
    monkeypatch.setattr(su, "FEATURE_PATH", info["feat"])
    inputs = str(tmp_path / "inputs")
    with pytest.raises(ValueError, match="FIRST_TREATMENT_START_DT"):
        prep.main(["--inputs-dir", inputs])
    assert not os.path.exists(inputs)


# --------------------------------------------------------------------------- skip / overwrite


def test_identical_rerun_skips_but_changed_preparation_cannot(store, capsys, monkeypatch):
    run(store)
    capsys.readouterr()
    run(store, "--chunk-rows", "7")  # chunk size does not change the data
    assert "[skip]" in capsys.readouterr().out
    for changed in (["--no-static"], ["--val-frac", "0.3"], ["--seed", "5"], ["--max-patients", "100"],
                    ["--allow-split-mismatch"]):
        with pytest.raises(SystemExit, match="cannot be reused"):
            run(store, *changed)
    monkeypatch.setattr(prep, "TIMESTAMP_POLICY", "another_policy")
    with pytest.raises(SystemExit, match="timestamp_policy"):
        run(store)
    monkeypatch.undo()
    # a changed source file invalidates the bundle
    path = store["notes"] + "full_clinical_notes_embeddings_metadata.parquet"
    st = os.stat(path)
    os.utime(path, ns=(st.st_atime_ns, st.st_mtime_ns + 10**9))
    monkeypatch.setattr(prep, "SURV_PATH", store["surv"])
    monkeypatch.setattr(prep, "NOTES_PATH", store["notes"])
    monkeypatch.setattr(schemes, "SURV_PATH", store["surv"])
    monkeypatch.setattr(su, "FEATURE_PATH", store["feat"])
    with pytest.raises(SystemExit, match="notes_metadata"):
        run(store)


def test_incomplete_bundle_is_not_skipped(store):
    run(store)
    os.remove(os.path.join(store["inputs"], "splits.csv"))
    with pytest.raises(SystemExit, match="file set"):
        run(store)
    run(store, "--overwrite")
    assert os.path.exists(os.path.join(store["inputs"], "splits.csv"))


def test_overwrite_with_no_static_removes_patients_csv(store):
    run(store)
    assert os.path.exists(os.path.join(store["inputs"], "patients.csv"))
    run(store, "--no-static", "--overwrite")
    assert not os.path.exists(os.path.join(store["inputs"], "patients.csv"))
    m = manifest(store["inputs"])
    assert m["static"] == {"enabled": False, "columns": []}
    data = load_bundle(store["inputs"], static=False)
    assert StaticTransform.fit(data.patients, data.split_ids("train")).output_dim == 0
    assert [p for p in os.listdir(store["tmp"]) if "backup" in p or "staging" in p] == []


def test_failed_overwrite_retains_the_prior_complete_bundle(store, monkeypatch):
    run(store)
    before = open(os.path.join(store["inputs"], "prepare_manifest.json")).read()
    listing = sorted(os.listdir(store["inputs"]))

    def boom(*a, **k):
        raise RuntimeError("disk full")

    monkeypatch.setattr(prep, "write_cohort_embeddings", boom)
    with pytest.raises(RuntimeError, match="disk full"):
        run(store, "--overwrite", "--val-frac", "0.3")
    assert sorted(os.listdir(store["inputs"])) == listing
    assert open(os.path.join(store["inputs"], "prepare_manifest.json")).read() == before
    assert [p for p in os.listdir(store["tmp"]) if "staging" in p] == []
    load_bundle(store["inputs"])


def test_overwrite_replaces_bundle_atomically(store):
    run(store)
    old = manifest(store["inputs"])["signature"]["val_frac"]
    run(store, "--overwrite", "--val-frac", "0.3")
    assert old == 0.2 and manifest(store["inputs"])["signature"]["val_frac"] == 0.3
    assert [p for p in os.listdir(store["tmp"]) if "backup" in p or "staging" in p] == []


@pytest.mark.parametrize("chunk_rows", [0, -1])
def test_invalid_chunk_rows_rejects_overwrite_without_changing_bundle(store, capsys, chunk_rows):
    run(store)
    inputs = store["tmp"] / "inputs"
    before = {p.name: (p.read_bytes(), p.stat().st_mtime_ns) for p in inputs.iterdir()}

    with pytest.raises(SystemExit) as exc:
        run(store, "--overwrite", "--chunk-rows", str(chunk_rows))

    assert exc.value.code == 2
    assert "--chunk-rows must be positive" in capsys.readouterr().err
    assert {p.name: (p.read_bytes(), p.stat().st_mtime_ns) for p in inputs.iterdir()} == before
    assert [p for p in os.listdir(store["tmp"]) if "backup" in p or "staging" in p] == []


@pytest.mark.parametrize("chunk_rows", [0, -1])
def test_embedding_writer_rejects_invalid_chunk_rows_before_touching_output(tmp_path, chunk_rows):
    output = tmp_path / "notes.npy"
    np.save(output, np.array([[9.0, 8.0]], dtype=np.float32))
    before = output.read_bytes(), output.stat().st_mtime_ns
    source = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float16)

    with pytest.raises(ValueError, match="chunk_rows must be positive"):
        prep.write_cohort_embeddings(source, np.array([1, 0]), str(output), chunk_rows)

    assert (output.read_bytes(), output.stat().st_mtime_ns) == before
    assert sorted(p.name for p in tmp_path.iterdir()) == ["notes.npy"]


# --------------------------------------------------------------------------- CLI


def test_cli_runs_as_a_module_with_ctep_data_path(tmp_path):
    info = build_store(tmp_path / "data")
    env = {**os.environ, "CTEP_DATA_PATH": str(tmp_path / "data") + "/"}
    inputs = tmp_path / "cli_inputs"
    cmd = [sys.executable, "-m", "pipelines.note_transformer.prepare_os_inputs", "--inputs-dir", str(inputs)]
    done = subprocess.run(cmd, cwd=REPO_ROOT, env=env, capture_output=True, text=True)
    assert done.returncode == 0, done.stderr[-2000:]
    assert "[done]" in done.stdout
    again = subprocess.run(cmd, cwd=REPO_ROOT, env=env, capture_output=True, text=True)
    assert "[skip]" in again.stdout
    assert info["vectors"].shape[1] == np.load(inputs / "notes.npy", mmap_mode="r").shape[1]
