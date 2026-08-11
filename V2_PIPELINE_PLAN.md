# v2 analysis decisions

The executable pipeline order and path conventions are documented in
[`REFACTOR_PLAN.md`](REFACTOR_PLAN.md).

Survival analyses support treatment and sequencing time-zero anchors. Because
`CoxnetSurvivalAnalysis` does not represent delayed entry, each arm is rebuilt
at its selected anchor: patients not alive and observable at that anchor are
excluded, note windows are filtered relative to that anchor, and endpoint times
are recomputed from source dates.

Metastatic-burden features are currently validated only for the treatment
anchor. Biomarker cohort 2 uses observed line-specific `MED_START_DT` landmarks
for cases and matched controls; it must never substitute first-treatment dates
for later lines.
