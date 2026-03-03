"""Build a comprehensive ICD-10-CM to phecode mapping from the Dana-Farber dataset.

Uses the phetk package (pip install phetk) which bundles the official
Phecode v1.2 and PhecodeX v1.0 ICD-10-CM mapping tables.
Uses v1.2 as the primary mapping and fills gaps with PhecodeX.

Outputs:
    CODE_PATH/icd10_to_phecode_mapping.csv   – the lookup table used downstream
    CODE_PATH/icd10_unmapped_codes.csv        – ICD-10 codes with no phecode match

Usage:
    python generate_icd10_to_phecode_mapping.py
"""

import os
import re
from typing import Optional

import pandas as pd
from phetk._utils import get_phecode_mapping_table

# ── paths ────────────────────────────────────────────────────────────────────
DATA_PATH = "/data/gusev/USERS/jpconnor/data/clinical_text_embedding_project/"
CODE_PATH = os.path.join(DATA_PATH, "code_data/")
SURV_PATH = os.path.join(DATA_PATH, "time-to-event_analysis/")


# ── helpers ──────────────────────────────────────────────────────────────────
def _normalize_icd10(code: str) -> Optional[str]:
    """Strip to uppercase alphanumeric, remove dots."""
    if pd.isna(code):
        return None
    code = str(code).strip().upper()
    code = re.sub(r"[^A-Z0-9]", "", code)
    return code if code else None


def _normalize_phecode(code) -> Optional[str]:
    if pd.isna(code):
        return None
    code = str(code).strip()
    code = re.sub(r"[^0-9.]", "", code)
    if not code:
        return None
    if code.count(".") > 1:
        left, right = code.split(".", 1)
        code = f"{left}.{right.replace('.', '')}"
    return code


def _load_v12_mapping() -> pd.DataFrame:
    """Load Phecode v1.2 ICD-10-CM mapping from phetk bundled data."""
    print("Loading Phecode v1.2 ICD-10-CM mapping from phetk …")
    df = get_phecode_mapping_table(
        phecode_version="1.2",
        icd_version="US",
        phecode_map_file_path=None,
        keep_all_columns=True,
    )
    print(f"  v1.2 mapping: {len(df)} rows, columns = {list(df.columns)}")
    return df


def _load_phecodex_mapping() -> pd.DataFrame:
    """Load PhecodeX v1.0 ICD-10-CM mapping from phetk bundled data."""
    print("Loading PhecodeX v1.0 ICD-10-CM mapping from phetk …")
    df = get_phecode_mapping_table(
        phecode_version="X",
        icd_version="US",
        phecode_map_file_path=None,
        keep_all_columns=True,
    )
    print(f"  PhecodeX mapping: {len(df)} rows, columns = {list(df.columns)}")
    return df


def _find_column(df: pd.DataFrame, candidates: list[str]) -> str:
    """Find the first matching column name (case-insensitive)."""
    col_map = {col.strip().lower(): col for col in df.columns}
    for c in candidates:
        if c.lower() in col_map:
            return col_map[c.lower()]
    raise ValueError(f"None of {candidates} found in {list(df.columns)}")


# ── main ─────────────────────────────────────────────────────────────────────
def main() -> None:
    os.makedirs(CODE_PATH, exist_ok=True)

    # 1. Load all unique ICD-10 codes from the Dana-Farber dataset
    icd_file = os.path.join(SURV_PATH, "timestamped_icd_info.csv")
    print(f"Loading ICD data from {icd_file} …")
    icd_df = pd.read_csv(icd_file, usecols=["DIAGNOSIS_ICD10_CD"])
    raw_codes = icd_df["DIAGNOSIS_ICD10_CD"].dropna().unique()
    print(f"  {len(raw_codes)} unique raw ICD-10 codes in dataset")

    norm_to_raw: dict[str, str] = {}
    for raw in raw_codes:
        norm = _normalize_icd10(raw)
        if norm and norm not in norm_to_raw:
            norm_to_raw[norm] = str(raw).strip()
    unique_norms = set(norm_to_raw.keys())
    print(f"  {len(unique_norms)} unique normalized ICD-10 codes")

    # 2. Load mapping tables from phetk
    v12_df = _load_v12_mapping()
    phecodeX_df = _load_phecodex_mapping()

    # 3. Normalize the v1.2 mapping
    # v1.2 columns: phecode_unrolled, ICD, flag, exclude_range, ...
    v12_icd_col = _find_column(v12_df, ["ICD", "icd10cm", "icd"])
    v12_phe_col = _find_column(v12_df, ["phecode_unrolled", "phecode"])
    v12_map = v12_df[[v12_icd_col, v12_phe_col]].copy()
    v12_map.columns = ["icd10_raw", "phecode_raw"]
    v12_map["icd10_norm"] = v12_map["icd10_raw"].map(_normalize_icd10)
    v12_map["phecode"] = v12_map["phecode_raw"].map(_normalize_phecode)
    v12_map = v12_map.dropna(subset=["icd10_norm", "phecode"]).drop_duplicates(
        subset=["icd10_norm", "phecode"]
    )
    v12_map["source"] = "phecode_v1.2"
    print(f"  v1.2 normalized: {len(v12_map)} unique (ICD, phecode) pairs")

    # 4. Normalize the PhecodeX mapping
    # PhecodeX columns: phecode, ICD, flag, code_val
    px_icd_col = _find_column(phecodeX_df, ["ICD", "icd"])
    px_phe_col = _find_column(phecodeX_df, ["phecode"])
    px_map = phecodeX_df[[px_icd_col, px_phe_col]].copy()
    px_map.columns = ["icd10_raw", "phecode_raw"]
    px_map["icd10_norm"] = px_map["icd10_raw"].map(_normalize_icd10)
    px_map["phecode"] = px_map["phecode_raw"].map(_normalize_phecode)
    px_map = px_map.dropna(subset=["icd10_norm", "phecode"]).drop_duplicates(
        subset=["icd10_norm", "phecode"]
    )
    px_map["source"] = "phecodeX_v1.0"
    print(f"  PhecodeX normalized: {len(px_map)} unique (ICD, phecode) pairs")

    # 5. Merge: v1.2 primary, PhecodeX fills gaps
    v12_covered = set(v12_map["icd10_norm"])
    px_gap_fill = px_map.loc[~px_map["icd10_norm"].isin(v12_covered)].copy()
    combined = pd.concat([v12_map, px_gap_fill], ignore_index=True)
    print(
        f"  Combined: {len(combined)} (ICD, phecode) pairs "
        f"({len(v12_map)} from v1.2 + {len(px_gap_fill)} gap-filled from PhecodeX)"
    )

    # 6. Parent-code fallback for unmapped codes
    # If a specific code like E1165 isn't mapped, try its parent E116, then E11
    combined_norms = set(combined["icd10_norm"])
    unmapped = unique_norms - combined_norms
    print(f"  {len(unmapped)} codes still unmapped after direct matching — trying parent codes …")

    parent_rows = []
    newly_mapped = 0
    for norm_code in sorted(unmapped):
        for prefix_len in range(len(norm_code) - 1, 2, -1):
            parent = norm_code[:prefix_len]
            parent_matches = combined.loc[combined["icd10_norm"] == parent, "phecode"].unique()
            if len(parent_matches) > 0:
                for phe in parent_matches:
                    parent_rows.append(
                        {
                            "icd10_raw": norm_to_raw.get(norm_code, norm_code),
                            "phecode_raw": phe,
                            "icd10_norm": norm_code,
                            "phecode": phe,
                            "source": "parent_code_match",
                        }
                    )
                newly_mapped += 1
                break

    if parent_rows:
        parent_df = pd.DataFrame(parent_rows)
        combined = pd.concat([combined, parent_df], ignore_index=True)
    print(f"  Parent-code matching recovered {newly_mapped} additional codes")

    # 7. Filter to only codes in the dataset
    final = combined.loc[combined["icd10_norm"].isin(unique_norms)].copy()
    final = final.drop_duplicates(subset=["icd10_norm", "phecode"])
    mapped_norms = set(final["icd10_norm"])
    still_unmapped = unique_norms - mapped_norms

    print(f"\n=== Summary ===")
    print(f"  Total unique ICD-10 codes in dataset:  {len(unique_norms)}")
    print(f"  Mapped to at least one phecode:        {len(mapped_norms)} ({100*len(mapped_norms)/len(unique_norms):.1f}%)")
    print(f"  Unmapped:                              {len(still_unmapped)} ({100*len(still_unmapped)/len(unique_norms):.1f}%)")
    print(f"  Total (ICD-10, phecode) pairs:         {len(final)}")
    print(f"  Unique phecodes:                       {final['phecode'].nunique()}")

    # 8. Write outputs
    out_cols = ["icd10_norm", "phecode", "source"]
    output = final[out_cols].copy()
    output.columns = ["icd10_code", "phecode", "source"]
    output = output.sort_values(["icd10_code", "phecode"]).reset_index(drop=True)

    out_path = os.path.join(CODE_PATH, "icd10_to_phecode_mapping.csv")
    output.to_csv(out_path, index=False)
    print(f"\nWrote mapping: {out_path}  ({len(output)} rows)")

    if still_unmapped:
        unmapped_df = pd.DataFrame(
            {
                "icd10_code": sorted(still_unmapped),
                "raw_example": [norm_to_raw.get(c, c) for c in sorted(still_unmapped)],
            }
        )
        unmapped_path = os.path.join(CODE_PATH, "icd10_unmapped_codes.csv")
        unmapped_df.to_csv(unmapped_path, index=False)
        print(f"Wrote unmapped codes: {unmapped_path}  ({len(unmapped_df)} rows)")


if __name__ == "__main__":
    main()
