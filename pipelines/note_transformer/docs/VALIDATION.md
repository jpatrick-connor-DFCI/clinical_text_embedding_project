# Independent-repository validation

## Previous verification and current rerun

The earlier portable implementation reported **25 passing CPU tests**, synthetic
training and search/resume, outcome-free inference and equivalent installed-wheel
predictions. That report was inspected as provenance; its institutional paths and
raw output artifacts were not copied into this repository.

The following checks were independently rerun from this separate Git repository
before the initial commit. All passed. All 15 transferred Python files are
byte-identical to the previously validated source; only documentation, packaging
metadata, ignore rules and lightweight example configurations were organized here.

| Check | Current result |
|---|---|
| Python syntax compilation | 15 files, PASS |
| Complete existing unit suite | 25 tests, 0 failures, 4.451 seconds, PASS |
| Full synthetic CPU demo | Two three-epoch search trials, 0 failed trials; single training run and final test evaluation, PASS |
| Search resume | Successful trials skipped; same selection, PASS |
| Interrupted training/search recovery | Existing unit tests replay interrupted runs and verify checkpoint equality, PASS |
| Four-file CLI walkthrough | Training, two-trial search, resume, explicit test evaluation, PASS |
| New-cohort outcome-free CLI inference | 24 synthetic patients, 72 patient-landmark rows including empty histories, PASS |
| Wheel build and isolated installation | PASS |
| Installed-wheel checkpoint reload and inference outside repository | 144 predictions equivalent to source predictions; no original VTE modules loaded, PASS |
| Documentation links, JSON configuration and CLI help | PASS |

The unit suite covers actual-event-time global risk membership, uniform sampling,
case inclusion, earlier censoring/competing-event exclusion, Breslow ties, strict
random landmark eligibility, history filtering/capping and padding invariance,
empty histories, hand-calculated Cox losses, full-Breslow equivalence, stable
gradients/toy learning, undefined C-index handling, split isolation, group-mean
microbatch gradients, checkpoint serialization, resume and validation-only model
selection. These are implementation checks, not clinical validation.

## Verified environment

| Component | Version |
|---|---|
| Python | 3.9.20 |
| PyTorch | 2.5.0+cu124 |
| NumPy | 1.24.4 |
| pandas | 1.4.4 |
| lifelines | 0.27.8 |
| Executed device | CPU |

The PyTorch build includes CUDA support, but **CUDA execution was not tested**.
No real clinical-note dataset has been validated. No claims of clinical utility,
calibration or generalization are made from synthetic C-index values.

## Commands and outputs

Run from the standalone root with the verified Python environment active:

```bash
python -m unittest discover -s tests -p 'test_note_survival.py' -v
python -m note_survival_transformer.synthetic_example \
  --output_dir results/validation_demo --run_demo
```

Syntax compilation used `py_compile` with compiled outputs redirected into a
temporary directory, covering the package, tests and three CLI scripts. A
reproducible equivalent is:

```bash
python - <<'PY'
from pathlib import Path
import py_compile, tempfile
files = list(Path('note_survival_transformer').glob('*.py'))
files += list(Path('tests').glob('*.py')) + list(Path('.').glob('*.py'))
with tempfile.TemporaryDirectory() as temporary:
    for i, path in enumerate(files):
        py_compile.compile(str(path), cfile=str(Path(temporary) / f'{i}.pyc'), doraise=True)
print(f'{len(files)} syntax checks passed')
PY
```

The [quickstart](COLLABORATOR_QUICKSTART.md) was executed with its exact flags:
48-patient generation, `synthetic_cpu_config.json`, two trials from
`synthetic_search_space.json`, the identical search command plus `--resume`,
explicit final test evaluation, and a separately generated 24-patient cohort
with distinct synthetic IDs for outcome-free inference.

The wheel was built with `--no-deps --no-build-isolation` and installed with
`--no-deps --no-index` into a temporary target. The inference process ran outside
the repository with that target on `PYTHONPATH`. Relocatable verification commands
are below; temporary paths are intentionally not retained in versioned reports:

```bash
export NST_REPO_ROOT="$PWD"
export NST_VERIFY_DIR="$(mktemp -d)"
python -m pip wheel . --no-deps --no-build-isolation --wheel-dir "$NST_VERIFY_DIR/wheels"
python -m pip install --no-deps --no-index --target "$NST_VERIFY_DIR/installed" \
  "$NST_VERIFY_DIR/wheels/note_survival_transformer-0.1.0-py3-none-any.whl"
cd "$NST_VERIFY_DIR"
PYTHONPATH="$NST_VERIFY_DIR/installed" python - <<'PY'
from pathlib import Path
import os, sys, pandas as pd
import note_survival_transformer as package
from note_survival_transformer.dataset import load_data
from note_survival_transformer.predict import inference
target = Path(os.environ['NST_VERIFY_DIR']) / 'installed'
assert Path(package.__file__).is_relative_to(target)
p = Path(os.environ['NST_REPO_ROOT']) / 'results/validation_demo'
data = load_data(p/'data/notes.npy', p/'data/notes.csv',
                 patients_path=p/'data/patients.csv', embedding_model_id='synthetic-v1')
actual = inference(p/'search/best_model.pt', data, [-2., .5, 2.])
expected = pd.read_csv(p/'outcome_free_predictions.csv', dtype={'patient_id': str})
pd.testing.assert_frame_equal(actual, expected, check_exact=False, rtol=1e-14, atol=1e-14)
assert not any(name in sys.modules for name in ('utils_training','utils_data','models','transformer'))
print(f'{len(actual)} installed-wheel predictions matched')
PY
cd "$NST_REPO_ROOT"
```

The generated demo locations are:

- `results/validation_demo/search/best_model.pt`
- `results/validation_demo/single_run/best_model.pt`
- `results/validation_demo/test/test_predictions.csv`
- `results/validation_demo/outcome_free_predictions.csv`
- `results/validation_demo/demo_results.json`
- `results/new_patient_predictions.csv` for the CLI walkthrough's second cohort.

These data/checkpoints/results, wheel/build artifacts and raw execution logs are
ignored and **not committed**. A sanitized, path-free result summary is retained
in [validation_summary.json](validation_summary.json).

## Reproducibility and checkpoint boundaries

Seeds cover Python, NumPy, PyTorch and independently seeded epoch samplers.
Deterministic PyTorch algorithms are enabled; CPU threads are configurable.
Each epoch samples controls and eligible anchors from a new deterministic stream.
Splits remain fixed across trials. Fingerprints bind inputs, encoder provenance,
configuration, static preprocessing and search settings.

Atomic checkpoints contain architecture/dimensions, model weights, static
transforms, encoder provenance, fingerprints, best validation metrics and epoch
history. Resume checkpoints additionally contain optimizer, scheduler and RNG
state. Interrupted epochs are replayed from the previous completed boundary.
Exactly one writer should use each output directory. Checkpoints are trusted
serialized artifacts; load only files you trust.

Resume tests verify equivalence within this environment. Exact replay across
different libraries, platforms or devices is not promised. Risk scores remain
relative hazard rankings, without fitted absolute survival/CIF calibration.

## Privacy and repository isolation

Only audited portable source, configs, tests and sanitized documentation are
committed. Checks cover internal paths/usernames, data-access URLs, credential
patterns, prohibited data/model formats, ignore behavior and the exact Git file
list. Original source files, Git HEAD, configuration/remotes and worktree/index
state are checked against a pre-task baseline and remain unchanged.

No public/open-source license is assigned. Institutional ownership and permitted
code-sharing terms remain for the owner to review. No collaborators are invited
automatically. In this preparation session, GitHub CLI authentication was absent;
private creation/push commands are provided in [SHARING.md](SHARING.md).
