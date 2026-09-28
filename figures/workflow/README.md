# Manuscript workflow schematic

- `clinical_text_workflow.pdf`: vector artwork with embedded TrueType fonts.
- `clinical_text_workflow.svg`: editable vector artwork; labels remain editable text.
- `clinical_text_workflow.png`: 600-dpi raster artwork.
- `clinical_text_workflow_preview.png`: lightweight preview.
- `caption.md`: standalone manuscript legend.

The canvas is 180 × 166.5 mm. The figure is a conceptual schematic, generated without patient data or estimated results. It covers the main manuscript analyses only. A figure number is deliberately left unassigned in the caption.

Regenerate from the repository root with:

```sh
python3 figures/render_workflow.py
```

The renderer requires matplotlib and numpy, and checks that all text stays within the canvas. SVG text uses DejaVu Sans; install that font when editing on another machine or use the PDF for submission. The default output directory is adjacent to the renderer under `figures/workflow/`. Existing manuscript figure renderers are independent of this standalone schematic.

## Method provenance

The schematic and caption were checked against these repository sources:

| Element | Source |
| --- | --- |
| Clinical ModernBERT; mean over token representations | `pipelines/preprocessing/generate_clinical_embeddings.py` |
| Baseline notes strictly before the anchor; note-type pooling; note-era covariates | `survival/preprocessing.py` |
| Baseline exponential decay λ = 0.01; complete note-type requirement | `pipelines/preprocessing/generate_embedding_prediction_datasets.py` |
| 768-dimensional note-type blocks | `semantic_search/README.md` |
| Default first-treatment anchor | `anchors.py` |
| Endpoint scheme definitions | `schemes.py` |
| Text/base Cox models and covariates | `pipelines/training/run_full_cohort_event.py`, `pipelines/training/slurm_array_utils.py` |
| Nested held-out text risk scores | `pipelines/training/run_full_cohort_risk_scores.py` |
| Main and supplemental analysis families | `figures/figure_captions.md`, `README.md` |
| Fixed baseline mortality model, repeated pooling, trajectory decay λ = 0.1 | `pipelines/trajectories/generate_mortality_trajectories.py` |
| Comparison-modality colors | `shared/palette.json` |

The study manuscript itself was not present in this checkout; wording follows the project methods and current figure legends.
