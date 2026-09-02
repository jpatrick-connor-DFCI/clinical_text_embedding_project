"""Literature-based validation logic for ICI biomarker hits.

Looks up interaction findings against the curated validation reference
and assigns a validation level + supporting notes to each hit.
"""

import os

import polars as pl

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
    """Rebuild all_findings_with_validation.csv from the compiled interaction output."""
    # `track` is retained as a constant 2 so downstream schemas are unchanged.
    track_specs = [
        (2, 'track2_all_significant_hits.csv'),
    ]
    frames = []
    missing = []

    for track, filename in track_specs:
        path = os.path.join(compiled_dir, filename)
        if not os.path.isfile(path):
            missing.append(path)
            continue

        df = pl.read_csv(path)
        required = {'marker', 'cancer_type', 'cohort', 'ps_model', 'weight_type'}
        missing_cols = sorted(required - set(df.columns))
        if missing_cols:
            raise ValueError(f"{filename} is missing required columns: {missing_cols}")

        cur = df.with_columns(
            pl.lit(track).alias('track'),
            pl.col('marker').map_elements(get_mutation_type, return_dtype=pl.Utf8).alias('mutation_type'),
            pl.lit('Unassessed').alias('validation_level'),
            pl.lit('').alias('validation_notes'),
        )
        frames.append(cur)

    if missing:
        raise FileNotFoundError(
            "Could not bootstrap all_findings_with_validation.csv because compiled hit files are missing:\n"
            + "\n".join(missing)
        )

    combined = pl.concat(frames, how='diagonal_relaxed')
    return combined


def load_or_bootstrap_findings_df(compiled_dir, findings_csv):
    os.makedirs(compiled_dir, exist_ok=True)

    if os.path.isfile(findings_csv):
        print(f"Loading {findings_csv}")
        return pl.read_csv(findings_csv)

    print(f"{findings_csv} not found; rebuilding from compiled interaction output in {compiled_dir}")
    df = bootstrap_findings_df(compiled_dir)
    df.write_csv(findings_csv)
    print(f"  Bootstrapped findings CSV with {len(df)} rows -> {findings_csv}")
    return df


def validate_findings(df, findings_csv):
    """Assign validation levels to unassessed hits, save the updated CSV, and print a summary.

    Returns the updated DataFrame.
    """
    # Normalize legacy pandas outputs: blank/null values and literal None/NaN
    # strings should all mean a completed review with no supporting evidence.
    level = pl.col('validation_level').cast(pl.String, strict=False)
    missing_level = (
        (level.is_null() | level.str.strip_chars().str.to_lowercase().is_in(['', 'none', 'nan']))
        & pl.col('validation_notes').is_not_null()
    )
    n_nan = int(df.select(missing_level.sum()).item())
    if n_nan > 0:
        df = df.with_columns(
            pl.when(missing_level).then(pl.lit('No Evidence')).otherwise(pl.col('validation_level')).alias('validation_level')
        )
        print(f"  Fixed {n_nan} rows with NaN validation_level -> 'No Evidence'")

    print(f"  {len(df)} total hits, {(df['validation_level'] == 'Unassessed').sum()} unassessed")

    # --- Validate unassessed hits ---
    updated = 0
    still_unassessed = 0
    rows = df.to_dicts()
    for row in rows:
        if row['validation_level'] != 'Unassessed':
            continue

        result = validate_hit(row['marker'], row['cancer_type'], row['mutation_type'])
        if result is not None:
            row['validation_level'] = result[0]
            row['validation_notes'] = result[1]
            updated += 1
        else:
            row['validation_level'] = 'No Evidence'
            row['validation_notes'] = (
                'No published evidence linking this gene-alteration to ICI response '
                'or immune modulation in this cancer context')
            still_unassessed += 1

    df = pl.DataFrame(rows, schema=df.schema)

    print(f"  Updated {updated} hits with literature validation")
    print(f"  {still_unassessed} hits assigned 'No Evidence' (no known ICI relevance)")

    # Save updated CSV
    df.write_csv(findings_csv)
    print(f"  Saved updated CSV to {findings_csv}")

    # --- Validation summary ---
    print(f"\nValidation level distribution:")
    print(df['validation_level'].value_counts())

    return df
