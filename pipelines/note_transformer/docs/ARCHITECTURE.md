# Architecture and landmark time

Choose the clinical **index/reference date** for each patient: admission, diagnosis,
treatment, or another appropriate occasion. It defines day zero for that patient.
A **landmark** `t0` is the time of a prediction relative to that index. For example,
`t0=7` predicts using information available through seven days after index.
In the original VTE study, treatment initiation defined day zero and example
landmarks were -7, 3, 7, 30 and 90 days. This package makes no such assumption. Its default is one landmark at zero; configure your own
schedule. Negative and fractional day values are supported.

Outcomes and notes must use the same origin and units. At retrospective landmark
evaluation, a patient is eligible only when their first-event/censoring time
`T > t0`. Notes at exactly `t0` are included. An event at `t0` is not eligible.
Using same-day records requires an appropriate availability-time definition.

## Compact encoder

```text
notes.csv + notes.npy
  → deterministic chronological ordering (same-time tie-break: note_id)
  → retain availability day <= t0
  → retain the most recent max_notes notes
  → linear note projection + continuous relative-time/landmark/gap features
  → learned landmark-conditioning prefix
  → projection + LayerNorm + dropout
  → small causal TransformerEncoder
  → last valid token pooling (landmark prefix when no notes)
  → optional static-covariate MLP + simple concatenation
  → one unconstrained primary-event log-risk score
```

The documented research design used two layers/two heads; the audited selected
VTE configuration used two layers/one head. Both are configurable. The example
here uses hidden width 64, two layers/two heads and 128 notes plus one landmark
token. It operates on fixed note vectors, not a language model or raw text.
There is no genetics, MoE, EHR token/type/value stack, cancer-specific processing,
Khorana, medication/procedure embedding, or competing-death prediction head.

```python
from note_survival_transformer import Config, NoteSurvivalTransformer

model = NoteSurvivalTransformer(Config(input_dim=768))
log_risk = model(
    x_notes=x_notes,               # float32 [B,L,D]
    days=days,                     # floating [B,L], days from index
    landmark=landmark,             # floating [B] or scalar
    attention_mask=attention_mask, # bool [B,L], True = valid note
    static_covariates=None,        # optional float32 [B,P]
)                                 # floating [B]
```

The forward API requires chronological order and also filters/caps histories to
protect direct callers. The loader handles sorting and same-time ordering.
Padding can be arbitrary and is ignored. Internally, PyTorch's key-padding mask
is the inverse: `True` means padding. Causal masks disallow attending to later
positions. Last-valid pooling assumes the internally compacted right-padded
sequence. The always-valid prefix prevents all-masked/empty-history problems.

For retained note day `d_k`, time features are:

```text
relative_time = tanh((d_k - t0) / time_scale_days)
landmark_time = tanh(t0 / landmark_scale_days)
gap = log1p(min(max(d_k - d_(k-1), 0), gap_cap_days)) / log1p(gap_cap_days)
```

Gaps are computed **after filtering and history capping**. The first retained
note has gap zero. The prefix carries the landmark feature in both time channels
and has zero gap. Default scales are 180/180/1096 days, preserving the audited
formulation; these scales are configurable and may need adjustment for a different
time horizon. The model uses time channels and causal order, without a separate
positional embedding. Precomputed vectors remain frozen; the note projection,
transformer, optional static encoder and risk head are trained.
