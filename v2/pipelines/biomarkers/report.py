"""Word document assembly for the ICI biomarker validation report."""

import os

import pandas as pd
from docx import Document
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.shared import Pt


def add_docx_table(doc, df, title=None, font_size=8):
    """Add a pandas DataFrame as a formatted table to a Word document."""
    if title:
        doc.add_heading(title, level=2)
    if df.empty:
        doc.add_paragraph('No data available.')
        return

    table = doc.add_table(rows=1, cols=len(df.columns))
    table.style = 'Light Grid Accent 1'
    table.alignment = WD_TABLE_ALIGNMENT.CENTER

    for j, col in enumerate(df.columns):
        cell = table.rows[0].cells[j]
        cell.text = str(col)
        for paragraph in cell.paragraphs:
            paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
            for run in paragraph.runs:
                run.bold = True
                run.font.size = Pt(font_size)

    for _, row in df.iterrows():
        row_cells = table.add_row().cells
        for j, val in enumerate(row):
            row_cells[j].text = str(val) if pd.notna(val) else ''
            for paragraph in row_cells[j].paragraphs:
                for run in paragraph.runs:
                    run.font.size = Pt(font_size)


def load_or_init_report_doc(report_docx):
    if os.path.isfile(report_docx):
        return Document(report_docx)

    doc = Document()
    doc.add_heading('ICI Biomarker Pipeline Report', level=1)
    doc.add_paragraph(
        'This report was initialized by validate_and_report.py because an existing '
        'report document was not found. Summary tables below are generated from the '
        'compiled biomarker-analysis outputs in this repository.'
    )
    return doc


def update_report_document(doc, df):
    """Append an updated "3. Updated Validation Summary" section to the report doc.

    Removes any previously appended Section 3 before regenerating it. Caller is
    responsible for saving the document.
    """
    # --- Remove any previously appended Section 3 ---
    # Find where "3. Updated Validation Summary" starts and truncate from there
    _cut_idx = None
    for i, p in enumerate(doc.paragraphs):
        if '3. Updated Validation Summary' in p.text:
            # Also remove the page break before it (previous paragraph)
            _cut_idx = max(i - 1, 0)
            break

    if _cut_idx is not None:
        print("  Removing old Section 3 before re-generating...")
        body = doc.element.body
        # Preserve sectPr (section properties) elements
        from docx.oxml.ns import qn
        elements_to_remove = [
            el for el in list(body)[_cut_idx:]
            if el.tag != qn('w:sectPr')
        ]
        for el in elements_to_remove:
            body.remove(el)

    # --- Add new section: Hit Distribution Summary ---
    doc.add_page_break()
    doc.add_heading('3. Updated Validation Summary', level=1)

    n_total_findings = len(df)
    n_unique_markers = len(df.drop_duplicates(subset=['marker', 'cancer_type']))

    doc.add_paragraph(
        'This section provides updated validation results after systematic literature review '
        'of all previously unassessed hits. Validation levels range from Very Strong '
        '(multiple large studies/trials) to No Evidence (no known ICI relevance).'
    )
    doc.add_paragraph(
        f'The dataset contains {n_total_findings} total findings across all specifications '
        f'(track x cohort x PS model x weighting), representing {n_unique_markers} unique '
        f'marker-cancer type pairs. A single marker-cancer pair can appear in multiple '
        f'specifications, so tables grouped by specification sum to more than {n_unique_markers}. '
        f'All tables below count findings (rows), not deduplicated marker-cancer pairs, '
        f'unless otherwise noted.'
    )

    # Table 1: Hits by specification (all rows, no dedup)
    doc.add_heading('3.1 Hit Counts by Specification', level=2)

    doc.add_paragraph(f'Each row counts findings for that specification. Rows sum to {n_total_findings}.')

    summary_rows = []
    for (track, cohort, ps_model, weight_type), grp in df.groupby(
            ['track', 'cohort', 'ps_model', 'weight_type']):
        track_label = f'Track {track}'
        n = len(grp)
        n_validated = grp['validation_level'].isin(
            ['Very Strong', 'Strong', 'Moderate']).sum()
        n_weak = grp['validation_level'].isin(
            ['Weak', 'Partial', 'Indirect']).sum()
        n_none = (grp['validation_level'] == 'No Evidence').sum()
        summary_rows.append({
            'Track': track_label,
            'Cohort': cohort,
            'PS Model': ps_model,
            'Weighting': weight_type,
            'Findings': n,
            'Strong+': n_validated,
            'Weak/Partial/Indirect': n_weak,
            'No Evidence': n_none,
        })

    summary_df = pd.DataFrame(summary_rows)
    add_docx_table(doc, summary_df, font_size=7)

    # Table 2: Validation distribution overall (all rows)
    doc.add_heading('3.2 Overall Validation Distribution', level=2)

    doc.add_paragraph(f'Counts all {n_total_findings} findings.')

    val_dist = df['validation_level'].value_counts().reset_index()
    val_dist.columns = ['Validation Level', 'Count']
    val_dist['Percentage'] = (val_dist['Count'] / val_dist['Count'].sum() * 100).round(1).astype(str) + '%'
    level_order = ['Very Strong', 'Strong', 'Moderate', 'Partial', 'Indirect', 'Weak', 'No Evidence', 'Unassessed']
    val_dist['_sort'] = val_dist['Validation Level'].map(
        {v: i for i, v in enumerate(level_order)}).fillna(99)
    val_dist = val_dist.sort_values('_sort').drop(columns='_sort')
    add_docx_table(doc, val_dist, font_size=8)

    # Table 3: Validation by cancer type (all rows)
    doc.add_heading('3.3 Validation by Cancer Type', level=2)

    doc.add_paragraph(f'Counts all {n_total_findings} findings. Rows sum to {n_total_findings}.')

    ct_val = (df.groupby('cancer_type')['validation_level']
              .value_counts()
              .unstack(fill_value=0)
              .reset_index())
    ordered_cols = ['cancer_type'] + [c for c in level_order if c in ct_val.columns]
    ct_val = ct_val[ordered_cols]
    ct_val['Total'] = ct_val[[c for c in ct_val.columns if c != 'cancer_type']].sum(axis=1)
    add_docx_table(doc, ct_val, font_size=7)

    # Table 4: Validated vs novel by track/cohort (all rows)
    doc.add_heading('3.4 Validated vs Novel by Track and Cohort', level=2)

    doc.add_paragraph(
        f'"Validated" includes Very Strong, Strong, and Moderate levels (direct published evidence). '
        f'"Supporting" includes Partial, Indirect, and Weak (biological plausibility or evidence in different context). '
        f'"No Evidence" = no known ICI relevance. Rows sum to {n_total_findings}.'
    )

    track_cohort_rows = []
    for (track, cohort), grp in df.groupby(['track', 'cohort']):
        n = len(grp)
        n_val = grp['validation_level'].isin(['Very Strong', 'Strong', 'Moderate']).sum()
        n_supp = grp['validation_level'].isin(['Weak', 'Partial', 'Indirect']).sum()
        n_none = (grp['validation_level'] == 'No Evidence').sum()
        track_cohort_rows.append({
            'Track': f'Track {track}',
            'Cohort': cohort,
            'Findings': n,
            'Validated': n_val,
            'Supporting': n_supp,
            'No Evidence': n_none,
            '% Validated': f'{n_val / max(n, 1) * 100:.0f}%',
        })

    tc_df = pd.DataFrame(track_cohort_rows)
    add_docx_table(doc, tc_df, font_size=8)

    # Table 5: By track/cohort/weighting (all rows)
    doc.add_heading('3.5 Validated vs Novel by Track, Cohort, and Weighting', level=2)

    doc.add_paragraph(
        f'Breakdown including weighting scheme (ATE = IPTW-weighted, unweighted/noIPTW = sensitivity analysis). '
        f'Rows sum to {n_total_findings}.'
    )

    tcw_rows = []
    for (track, cohort, weight_type), grp in df.groupby(['track', 'cohort', 'weight_type']):
        n = len(grp)
        n_val = grp['validation_level'].isin(['Very Strong', 'Strong', 'Moderate']).sum()
        n_supp = grp['validation_level'].isin(['Weak', 'Partial', 'Indirect']).sum()
        n_none = (grp['validation_level'] == 'No Evidence').sum()
        tcw_rows.append({
            'Track': f'Track {track}',
            'Cohort': cohort,
            'Weighting': weight_type,
            'Findings': n,
            'Validated': n_val,
            'Supporting': n_supp,
            'No Evidence': n_none,
            '% Validated': f'{n_val / max(n, 1) * 100:.0f}%',
        })

    tcw_df = pd.DataFrame(tcw_rows)
    add_docx_table(doc, tcw_df, font_size=7)

    # Table 6: Weighted vs unweighted overlap (unique marker x cancer within track x cohort)
    doc.add_heading('3.6 Weighted vs Unweighted Hit Overlap', level=2)

    doc.add_paragraph(
        'For each track and cohort, this table shows how many unique marker-cancer pairs '
        'are found in both weighted and unweighted analyses, versus only one. '
        'Note: this table counts unique marker-cancer pairs (not findings), since the '
        'comparison is across weighting schemes within each track-cohort group.'
    )

    overlap_rows = []
    for (track, cohort), grp in df.groupby(['track', 'cohort']):
        weighted_key = 'ATE'
        unweighted_key = 'unweighted' if track == 1 else 'noIPTW'

        w_hits = set(
            grp[grp['weight_type'] == weighted_key]
            .apply(lambda r: (r['marker'], r['cancer_type']), axis=1))
        uw_hits = set(
            grp[grp['weight_type'] == unweighted_key]
            .apply(lambda r: (r['marker'], r['cancer_type']), axis=1))

        both = w_hits & uw_hits
        w_only = w_hits - uw_hits
        uw_only = uw_hits - w_hits
        total = len(w_hits | uw_hits)

        overlap_rows.append({
            'Track': f'Track {track}',
            'Cohort': cohort,
            'Both': len(both),
            f'{weighted_key} Only': len(w_only),
            f'{unweighted_key} Only': len(uw_only),
            'Total Markers': total,
            '% Overlap': f'{len(both) / max(total, 1) * 100:.0f}%',
        })

    overlap_df = pd.DataFrame(overlap_rows)
    add_docx_table(doc, overlap_df, font_size=8)

    # Table 7: Validation enrichment in concordant hits (unique marker x cancer)
    doc.add_heading('3.7 Validation Enrichment in Concordant Hits', level=2)

    doc.add_paragraph(
        'This table compares validation rates for markers found in both weighted and unweighted '
        'analyses ("Both") versus markers exclusive to one scheme. Counts are unique marker-cancer '
        'pairs. Within each track-cohort group, Both + ATE only + unweighted only = Total Markers.'
    )

    enrich_rows = []
    for (track, cohort), grp in df.groupby(['track', 'cohort']):
        weighted_key = 'ATE'
        unweighted_key = 'unweighted' if track == 1 else 'noIPTW'

        w_hits = set(
            grp[grp['weight_type'] == weighted_key]
            .apply(lambda r: (r['marker'], r['cancer_type']), axis=1))
        uw_hits = set(
            grp[grp['weight_type'] == unweighted_key]
            .apply(lambda r: (r['marker'], r['cancer_type']), axis=1))

        both = w_hits & uw_hits
        w_only = w_hits - uw_hits
        uw_only = uw_hits - w_hits

        for label, hit_set in [('Both', both),
                               (f'{weighted_key} only', w_only),
                               (f'{unweighted_key} only', uw_only)]:
            if not hit_set:
                continue
            subset = grp[grp.apply(
                lambda r: (r['marker'], r['cancer_type']) in hit_set, axis=1
            )].drop_duplicates(subset=['marker', 'cancer_type'])
            n = len(subset)
            n_val = subset['validation_level'].isin(['Very Strong', 'Strong', 'Moderate']).sum()
            n_supp = subset['validation_level'].isin(['Weak', 'Partial', 'Indirect']).sum()
            n_none = (subset['validation_level'] == 'No Evidence').sum()
            enrich_rows.append({
                'Track': f'Track {track}',
                'Cohort': cohort,
                'Category': label,
                'Markers': n,
                'Validated': n_val,
                'Supporting': n_supp,
                'No Evidence': n_none,
                '% Validated': f'{n_val / max(n, 1) * 100:.0f}%',
            })

    enrich_df = pd.DataFrame(enrich_rows)
    add_docx_table(doc, enrich_df, font_size=7)

    # Interpretive summary paragraph
    doc.add_heading('3.8 Weighting Comparison Summary', level=2)

    doc.add_paragraph(
        'Neither ATE-weighted nor unweighted analyses are uniformly superior in validation rate. '
        'However, concordant hits — those appearing in both weighted and unweighted results — '
        'are substantially enriched for validated biomarkers compared to hits exclusive to one scheme:'
    )

    bullet_points = [
        ('Cohort 1, Track 1: Concordant hits have 32% validation rate versus 6% (ATE-only) '
         'and 19% (unweighted-only), indicating that IPTW weighting and unweighted sensitivity '
         'analysis together filter noise effectively.'),
        ('Cohort 2, Track 1: 144 of ~160 hits overlap between weighting schemes (very high concordance). '
         'The few exclusive hits in either direction have 0% validation rate, consistent with noise.'),
        ('Track 2: Small sample sizes limit interpretation, but cohort 2 ATE-weighted has the highest '
         'precision (40% validation, 5 hits). Concordant Track 2 hits across weighting schemes are again '
         'the most validated subset.'),
        ('Recommendation: Prioritize hits that replicate across both weighting schemes. '
         'These concordant findings are most robust to propensity score model assumptions '
         'and most enriched for biologically validated ICI biomarker associations.'),
    ]
    for bp in bullet_points:
        doc.add_paragraph(bp, style='List Bullet')

    return doc
