# Note Survival Transformer

A compact causal transformer for first-event, landmark-conditioned survival risk
ranking from precomputed note embeddings. Training uses event-centered controls
from the full eligible training cohort and a stable sampled Cox objective.

See the repository README and docs/ for installation, data schemas, mathematics,
search/resume, validation and collaborator onboarding. Installed CLI commands are
`note-survival-train`, `note-survival-search`, and `note-survival-predict`.
Run `python -m note_survival_transformer.synthetic_example --help` for a self-contained
synthetic example. Outputs are log-risk rankings, not calibrated probabilities.
No real clinical-note dataset or CUDA execution has been validated.
