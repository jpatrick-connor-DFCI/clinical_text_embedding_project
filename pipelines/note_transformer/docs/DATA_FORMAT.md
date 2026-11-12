# Input data contract

`notes.csv`:

| Column | Type | Meaning |
|---|---|---|
| patient_id | string | Patient key; preserved as a string, including leading zeros |
| note_id | string | Unique within patient; deterministic same-time ordering |
| days_from_index | finite floating point | Actual information-availability time in days |
| embedding_row | integer | Row of the matching vector in notes.npy |

`notes.npy` is a non-pickled float32 NumPy matrix `[N_notes,D]` with `D > 0` and
finite values. Metadata must cover its rows exactly once: `embedding_row` is a
permutation of `0..N_notes-1`. The array is memory-mapped; full histories are not
globally truncated. All vectors must come from the same embedding encoder/version
and normalization convention. Specify `--embedding_model_id encoder-name/version`
at training and inference to enforce that provenance. Same dimension alone does
not imply compatible embedding spaces.

**Availability must be real, not a retrospective service date.** Each vector
must encode only content actually recorded/available by its supplied timestamp.
Do not embed a whole future patient history and assign the resulting vector to an
earlier note. The package can enforce timestamps and shape/provenance checks; it
cannot inspect an embedding to prove that its text was available at that time.

`outcomes.csv`:

| Column | Type | Meaning |
|---|---|---|
| patient_id | string | Exactly one row per patient |
| time | finite floating point | Observed FIRST event or censoring day from index |
| event | 0/1 or boolean | Primary event occurred at `time` |
| death_event | optional 0/1 or boolean | Competing death occurred at `time` |

`event` and `death_event` are mutually exclusive. For a competing death, set
`event=0`, `death_event=1` and `time=death day`. Death then terminates primary-event
follow-up even though there is no death head. Death after a primary event is not
a second outcome row. A censored patient has both flags zero and a finite last
observation day. Do not use separate cause-specific follow-up columns or infinite
time for censoring. Additional outcome columns, duplicate outcome rows, recurrent
event schemas and delayed-entry/start-stop schemas are rejected.

`splits.csv` contains `patient_id,split`, with split exactly `train`, `val` or
`test`. It must cover the outcome population exactly once, with at least one
patient in each split. Splits are fixed before any landmark expansion. Notes
cannot refer to patients absent from outcomes. Patients with no notes still belong
to outcomes/splits and remain part of the risk set.

Optional `patients.csv` contains unique `patient_id` rows and numeric static
columns. All static values must be available at every modeled landmark. Values
are median-imputed/standardized using training patients only, and missingness
indicators are appended. Entirely missing training columns use zero with scale
one. Missing patient rows are treated as missing features. At inference, provide
the same column names; saved statistics are reused. Categorical covariates require
an upstream fixed encoding. Do not treat future measurements as baseline features.

CSV IDs must be nonempty and have no leading/trailing whitespace. Outcome times
remain float64 for risk membership and cutoffs. Raw model timestamps may be
float64; projected time features and vectors are float32.

## Outcome-free patients

Without outcomes, patient IDs come from notes and optional patients.csv. Include
`--patient_ids_path patient_ids.csv` to retain patients with no notes. The caller
must establish event-free/observed status at each requested landmark. All model
inputs use available information only; outcomes are never passed to forward().
