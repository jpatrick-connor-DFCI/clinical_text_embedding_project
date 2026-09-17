# Manuscript figure review

Reviewed the compiled PNGs and vector PDFs in `Desktop/manuscript_figures` for
Figures 1–4. The prepared CSV inputs are on the cluster, so revised scientific
plots must be regenerated there. No curves, observations, or patient counts were
digitized from images, and the downloaded originals were not overwritten.

| Observed issue | Rendering change |
| --- | --- |
| The compiled figures were 18–20 inches wide. At a 180-mm submission width, the smallest text was approximately 3.15–3.19 pt. | Main figures now render directly at 180 mm, with approximately 6–8 pt typography and 9 pt panel letters. PNGs are 600 dpi; Cairo PDFs preserve vector graphics and embedded fonts when Cairo is available. |
| Figure 3 lacked panel labels b and c. | Each label is drawn into its panel before assembly, including nested plots and panels spanning columns. |
| White percentages on pastel pie slices had low contrast; the legend consumed much of the pie's width. | Dark percentage labels and a two-column legend below the pie; shorter headings and a separate cohort subtitle. |
| Panel headings, inline summaries, and notes were disproportionately long. | Shorter headings; statistical/methodological explanations move to full figure captions. |
| Figure 2 mortality KM panels were labeled “event-free survival.” | Axes now read “Overall survival.” Risk tables are computed from the fitted survival objects. |
| Figure 3a displayed 406 endpoints, while 3b/c displayed 493. | Captions distinguish complete-ranking availability from the joint-model endpoint set. These sample sets are not silently forced to match. The legacy mean ± SEM fallback is removed because it was mislabeled as IQR and could reintroduce excluded endpoints. |
| Figure 3c silently omitted coefficients for display and used small footer text. | Its legend explains the 1.5-IQR display trimming; modality-specific plotted sample sizes appear on the x axis. Jitter positions are reproducible. |
| Figure 4b used a truncated vertical scale while 4d used 0–1. | Both now use 0–1 survival scales, common landmark time conventions, and risk tables. Long hazard-ratio summaries are carried into the data-derived caption, with the Cox-model sample size. |
| The figure legends did not fully explain uncertainty, filtering, or denominators. | Versioned captions define the confidence intervals, IQRs, FDR family, landmark convention, cohort differences, and the existing two-sided endpoint filter. |

The endpoint outlier filter and joint-model estimation choices are retained.
The captions disclose their behavior rather than treating this formatting review
as a new statistical analysis. Data-derived hazard ratios, sample sizes, and
landmark months are appended during the cluster render.

## Outputs after rerendering

- `png/figureN/figureN_cindex.png` and `pdf/figureN/figureN_cindex.pdf`: figure artwork.
- `png/figureN/figureN_cindex_captioned.png` and corresponding PDF: review copies
  with the complete legend below the artwork.
- `captions/figureN.md` and `.txt`: standalone submission legends.

The captioned review copies are taller than the submission artwork. Supply the
clean artwork and separate legends if the target journal requests separate files.
The journal-specific dimensions can be adjusted from the 180-mm working width.

Working reference: [Nature initial-submission guidance](https://www.nature.com/nature/for-authors/initial-submission)
recommends final-size preparation, editable vector artwork, and self-contained
legends. The final cluster outputs still require visual inspection before submission.

## Verification and cluster rerun

The nine targeted Python regression checks passed. All 19 R source/test files
and the figure notebook's R chunks passed static syntax parsing; `git diff --check`
also passed. R execution was not available locally (the installed R binary is
incompatible with this Mac's architecture), and the prepared figure CSVs are on
the cluster. The revised figures have therefore not yet been rendered or visually
verified. Synthetic R checks cover nested/spanning panel labels, caption outputs,
and ordinary/delayed-entry risk-table counts.

From the repository root on the cluster, run:

```sh
Rscript tests/test_figure_composition.R
Rscript tests/test_publication_risk_tables.R
bash R/render_all_figures.sh plot_figure_1.R plot_figure_2.R plot_figure_3.R plot_figure_4.R
```

Upload the new shared R helpers and `figures/manuscript_captions.json` along with
the modified plotting scripts. The existing prepared CSV inputs can be reused.
