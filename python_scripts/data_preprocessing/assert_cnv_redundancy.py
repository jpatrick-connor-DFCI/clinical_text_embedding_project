"""Assert that CNV columns are fully redundant with AMP/DEL columns in somatic data.

For every gene and every sample, if GENE_CNV == 1 then GENE_AMP == 1 or GENE_DEL == 1.
If this holds universally, CNV columns can be safely dropped.
"""

import os
import pandas as pd

CLINICAL_FEATURE_PATH = '/data/gusev/USERS/jpconnor/data/clinical_text_embedding_project/clinical_and_genomic_features/'

somatic_df = pd.read_csv(os.path.join(CLINICAL_FEATURE_PATH, 'complete_somatic_data_df.csv'))

# Extract gene stems that have a _CNV column
cnv_cols = [c for c in somatic_df.columns if c.endswith('_CNV')]
gene_stems = [c.replace('_CNV', '') for c in cnv_cols]

print(f"Found {len(cnv_cols)} CNV columns across {len(somatic_df)} samples.\n")

violations = []

for gene in gene_stems:
    cnv_col = f'{gene}_CNV'
    amp_col = f'{gene}_AMP'
    del_col = f'{gene}_DEL'

    has_cnv = somatic_df[cnv_col] == 1

    # Build AMP | DEL mask (columns may not exist for every gene)
    has_amp_or_del = pd.Series(False, index=somatic_df.index)
    if amp_col in somatic_df.columns:
        has_amp_or_del |= (somatic_df[amp_col] == 1)
    if del_col in somatic_df.columns:
        has_amp_or_del |= (somatic_df[del_col] == 1)

    # Violation: CNV == 1 but neither AMP nor DEL == 1
    mask = has_cnv & ~has_amp_or_del
    n_violations = mask.sum()

    if n_violations > 0:
        violations.append({
            'gene': gene,
            'n_cnv': has_cnv.sum(),
            'n_violations': n_violations,
            'has_amp_col': amp_col in somatic_df.columns,
            'has_del_col': del_col in somatic_df.columns,
        })

if not violations:
    print("PASSED: Every sample with CNV==1 also has AMP==1 or DEL==1.")
    print("CNV columns are fully redundant and can be safely dropped.")
else:
    viol_df = pd.DataFrame(violations).sort_values('n_violations', ascending=False)
    print(f"FAILED: {len(violations)} gene(s) have CNV==1 without a corresponding AMP/DEL==1:\n")
    print(viol_df.to_string(index=False))

# Also check the reverse: does AMP|DEL always imply CNV?
reverse_violations = []
for gene in gene_stems:
    cnv_col = f'{gene}_CNV'
    amp_col = f'{gene}_AMP'
    del_col = f'{gene}_DEL'

    has_cnv = somatic_df[cnv_col] == 1

    has_amp_or_del = pd.Series(False, index=somatic_df.index)
    if amp_col in somatic_df.columns:
        has_amp_or_del |= (somatic_df[amp_col] == 1)
    if del_col in somatic_df.columns:
        has_amp_or_del |= (somatic_df[del_col] == 1)

    mask = has_amp_or_del & ~has_cnv
    n_rev = mask.sum()
    if n_rev > 0:
        reverse_violations.append({'gene': gene, 'n_amp_or_del_without_cnv': n_rev})

print()
if not reverse_violations:
    print("REVERSE CHECK PASSED: AMP|DEL always implies CNV (perfect 1:1 correspondence).")
else:
    rev_df = pd.DataFrame(reverse_violations).sort_values('n_amp_or_del_without_cnv', ascending=False)
    print(f"REVERSE CHECK: {len(reverse_violations)} gene(s) have AMP|DEL==1 without CNV==1:")
    print(f"(This is expected if CNV is a subset indicator, not a superset.)\n")
    print(rev_df.head(20).to_string(index=False))
    print(f"\n... showing top 20 of {len(reverse_violations)}" if len(reverse_violations) > 20 else "")
