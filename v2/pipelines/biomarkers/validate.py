"""Literature-based validation logic for ICI biomarker hits.

Looks up track1/track2 findings against the curated validation reference
and assigns a validation level + supporting notes to each hit.
"""

import os

import pandas as pd

from data.validation_reference import load_validation_reference

from pipelines.biomarkers.biomarker_common import get_mutation_type

# ============================================================
# Known ICI biomarker validation reference
# ============================================================
# Keys: (gene, mutation_type) for pan-cancer evidence,
#        or (gene, mutation_type, cancer_type) for cancer-specific.
# Values: (validation_level, validation_notes)
#
# Validation levels follow the existing schema:
#   Very Strong — multiple large studies/trials, established clinical biomarker
#   Strong      — large study with direct ICI-specific evidence, prospective data
#   Moderate    — published studies with direct ICI or prognostic evidence
#   Weak        — limited, conflicting, or small-study evidence
#   Partial     — evidence in a different cancer type or alteration context
#   Indirect    — biological plausibility via known ICI-relevant pathway
#   None        — no known connection to ICI biology or prognosis
#
# Loaded from data/validation_reference.csv (see data/validation_reference.py).
VALIDATION_REF = load_validation_reference()


def _parse_gene_and_alteration(marker_name):
    """Extract gene name and alteration type from marker column name."""
    mutation_tags = ('_SNV', '_SV', '_FUSION', '_DEL', '_AMP')
    marker_upper = marker_name.upper()
    for tag in mutation_tags:
        if marker_upper.endswith(tag):
            gene = marker_name[:marker_upper.rfind(tag)]
            return gene, tag
    return marker_name, '_OTHER'


def validate_hit(marker, cancer_type, mutation_type):
    """Look up a hit in the validation reference.

    Searches in order: (gene, mut, cancer), (gene, mut), pan_cancer variants.
    """
    gene, alt_type = _parse_gene_and_alteration(marker)
    if alt_type == '_OTHER':
        alt_type = mutation_type

    # 1. Cancer-type-specific match
    key_specific = (gene, alt_type, cancer_type)
    if key_specific in VALIDATION_REF:
        return VALIDATION_REF[key_specific]

    # 2. Generic (gene, alt_type) match
    key_generic = (gene, alt_type)
    if key_generic in VALIDATION_REF:
        return VALIDATION_REF[key_generic]

    # 3. Pan-cancer explicit match
    key_pan = (gene, alt_type, 'pan_cancer')
    if key_pan in VALIDATION_REF:
        return VALIDATION_REF[key_pan]

    return None


def bootstrap_findings_df(compiled_dir):
    """Rebuild all_findings_with_validation.csv from compiled Track 1/2 outputs."""
    track_specs = [
        (1, 'track1_all_significant_hits.csv'),
        (2, 'track2_all_significant_hits.csv'),
    ]
    frames = []
    missing = []

    for track, filename in track_specs:
        path = os.path.join(compiled_dir, filename)
        if not os.path.isfile(path):
            missing.append(path)
            continue

        df = pd.read_csv(path)
        required = {'marker', 'cancer_type', 'cohort', 'ps_model', 'weight_type'}
        missing_cols = sorted(required - set(df.columns))
        if missing_cols:
            raise ValueError(f"{filename} is missing required columns: {missing_cols}")

        cur = df.copy()
        cur['track'] = track
        cur['mutation_type'] = cur['marker'].map(get_mutation_type)
        cur['validation_level'] = 'Unassessed'
        cur['validation_notes'] = ''
        frames.append(cur)

    if missing:
        raise FileNotFoundError(
            "Could not bootstrap all_findings_with_validation.csv because compiled hit files are missing:\n"
            + "\n".join(missing)
        )

    combined = pd.concat(frames, ignore_index=True, sort=False)
    return combined


def load_or_bootstrap_findings_df(compiled_dir, findings_csv):
    os.makedirs(compiled_dir, exist_ok=True)

    if os.path.isfile(findings_csv):
        print(f"Loading {findings_csv}")
        return pd.read_csv(findings_csv)

    print(f"{findings_csv} not found; rebuilding from compiled Track 1/2 outputs in {compiled_dir}")
    df = bootstrap_findings_df(compiled_dir)
    df.to_csv(findings_csv, index=False)
    print(f"  Bootstrapped findings CSV with {len(df)} rows -> {findings_csv}")
    return df


def validate_findings(df, findings_csv):
    """Assign validation levels to unassessed hits, save the updated CSV, and print a summary.

    Returns the updated DataFrame.
    """
    # Fix any NaN validation_level from previous runs that wrote 'None' (parsed as NaN by pandas)
    nan_mask = df['validation_level'].isna() & df['validation_notes'].notna()
    if nan_mask.sum() > 0:
        df.loc[nan_mask, 'validation_level'] = 'No Evidence'
        print(f"  Fixed {nan_mask.sum()} rows with NaN validation_level -> 'No Evidence'")

    print(f"  {len(df)} total hits, {(df['validation_level'] == 'Unassessed').sum()} unassessed")

    # --- Validate unassessed hits ---
    updated = 0
    still_unassessed = 0
    for idx, row in df.iterrows():
        if row['validation_level'] != 'Unassessed':
            continue

        result = validate_hit(row['marker'], row['cancer_type'], row['mutation_type'])
        if result is not None:
            df.at[idx, 'validation_level'] = result[0]
            df.at[idx, 'validation_notes'] = result[1]
            updated += 1
        else:
            df.at[idx, 'validation_level'] = 'No Evidence'
            df.at[idx, 'validation_notes'] = (
                'No published evidence linking this gene-alteration to ICI response '
                'or immune modulation in this cancer context')
            still_unassessed += 1

    print(f"  Updated {updated} hits with literature validation")
    print(f"  {still_unassessed} hits assigned 'No Evidence' (no known ICI relevance)")

    # Save updated CSV
    df.to_csv(findings_csv, index=False)
    print(f"  Saved updated CSV to {findings_csv}")

    # --- Validation summary ---
    print(f"\nValidation level distribution:")
    print(df['validation_level'].value_counts().to_string())

    return df
