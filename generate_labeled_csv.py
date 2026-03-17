"""
Generate all_findings_with_validation.csv — labeled CSV with track, validation level, notes,
and (for Track 1) a consistency score.
"""

import re
import pandas as pd
import numpy as np

# ── paths ──────────────────────────────────────────────────────────────────────
DATA_DIR = "/Users/connorpa/Documents/BIG PhD/Gusev Lab/Projects/Clinical Embeddings/compiled_results"
OUT_PATH  = f"{DATA_DIR}/all_findings_with_validation.csv"

# ── helper: strip mutation-type suffix ────────────────────────────────────────
SUFFIXES = re.compile(r"(_SNV|_DEL|_AMP|_SV|_FUSION)$", re.IGNORECASE)

def extract_gene(marker: str) -> str:
    return SUFFIXES.sub("", str(marker))

# ── validation lookup tables ───────────────────────────────────────────────────

TRACK2_LOOKUP: dict[tuple[str, str], tuple[str, str]] = {
    ("CSF1R", "pan_cancer"): ("Very Strong", "CSF1R controls TAM immunosuppression; CSF1R inhibitor+ICI synergy extensively validated; loss-of-function SNV → less immunosuppressive TME"),
    ("MET",   "pan_cancer"): ("Strong",      "2024 pan-cancer study directly validates MET mutation as independent favorable ICI biomarker via enhanced immune infiltration"),
    ("ESR1",  "pan_cancer"): ("Strong",      "ESR1/ER expression drives immunosuppressive TME; 2025 study shows ESR1 expression predicts lower ICI response (AUC=0.88)"),
    ("PTPRD", "BRAIN"):      ("Strong",      "Multiple studies show PTPRD/PTPRT mutations associate with better ICI outcomes via JAK-STAT/antigen presentation; pan-cancer validated"),
    ("SDHD",  "SOFT_TISSUE"):("Moderate",    "SDH-deficient GISTs/sarcomas documented as ICI non-responsive (nivolumab → no PRs); consistent with ICI harm direction"),
    ("MAP2K1","BRAIN"):      ("Partial",     "MAP2K1/2 mutations predict anti-CTLA-4 benefit in melanoma; MAPK-ICI connection established; brain-tumor-specific evidence limited"),
    ("BRAF",  "pan_cancer"): ("Partial",     "BRAF fusions associated with ICI challenges in case reports; pan-cancer BRAF SV literature not systematized"),
    ("ZNRF3", "BREAST"):     ("Indirect",    "ZNRF3 deletion activates Wnt/beta-catenin; Wnt pathway activation is an established mechanism of immune exclusion and ICI resistance"),
    ("NF1",   "SOFT_TISSUE"):("Partial",     "NF1 mutations associated with higher TMB and ICI benefit in NSCLC/melanoma; harm direction in STS is inconsistent with broader literature"),
    ("ATM",   "BRAIN"):      ("Partial",     "ATM-ICI connection via DDR/TMB established, but harm direction in brain tumors contradicts expected DDR benefit; possibly confounded"),
    ("HNF1A", "LUNG"):       ("Weak",        "HNF1A-immune cell infiltration correlation reported; no direct clinical evidence for ICI benefit in lung cancer"),
    ("ATM",   "SOFT_TISSUE"):("Weak",        "ATM loss generally predicts ICI benefit via DDR; harm direction in STS not supported and biologically counterintuitive"),
    ("PIK3R1","BRAIN"):      ("Weak",        "PI3K pathway has indirect immunological relevance; PIK3R1-specific ICI biomarker evidence in brain tumors absent"),
    ("CBL",   "SOFT_TISSUE"):("None",        "No established literature linking CBL deletion to ICI response in any cancer type"),
    ("RUNX1", "pan_cancer"): ("None",        "RUNX1 SVs primarily studied in hematologic malignancies; no solid tumor ICI biomarker evidence"),
    ("NEIL2", "BREAST"):     ("None",        "No clinical evidence for NEIL2 deletion as ICI predictive biomarker in breast cancer"),
    ("NKX3-1","BREAST"):     ("None",        "NKX3-1 primarily studied in prostate cancer; no breast cancer ICI biomarker role established"),
    ("PTK2B", "BREAST"):     ("None",        "No established literature linking PTK2B deletion to ICI response in breast cancer"),
    ("PML",   "BRAIN"):      ("None",        "PML deletion has no established role as ICI predictive biomarker in brain tumors"),
    ("GBA",   "pan_cancer"): ("None",        "GBA mutations not established as ICI biomarkers; single-specification finding likely spurious"),
}

TRACK1_LOOKUP: dict[tuple[str, str], tuple[str, str]] = {
    ("STK11",  "LUNG"):        ("Very Strong", "STK11/LKB1 loss is one of the most validated ICI resistance biomarkers in NSCLC; multiple clinical trials confirm immune-cold phenotype"),
    ("STK11",  "pan_cancer"):  ("Very Strong", "Pan-cancer validated ICI resistance marker; STK11 loss drives neutrophil-rich immune-excluded TME"),
    ("KEAP1",  "LUNG"):        ("Strong",      "KEAP1 mutations drive NRF2 hyperactivation and T-cell exclusion; validated ICI resistance biomarker in NSCLC (Cell Reports 2023, JTO 2022)"),
    ("KEAP1",  "pan_cancer"):  ("Strong",      "Pan-cancer validated ICI resistance marker via NRF2/immune evasion axis"),
    ("CD274",  "pan_cancer"):  ("Strong",      "CD274 is the PD-L1 gene itself; deletion reduces drug target expression, directly relevant to ICI mechanism"),
    ("PDCD1LG2","pan_cancer"): ("Strong",      "PDCD1LG2 is the PD-L2 gene; deletion alters the PD-1 ligand landscape, directly mechanistically relevant"),
    ("PBRM1",  "pan_cancer"):  ("Strong",      "Directly validated ICI response biomarker in RCC (JAMA Oncology 2019); pan-cancer evidence from Frontiers Genetics 2022"),
    ("ARID1A", "pan_cancer"):  ("Strong",      "Pan-cancer analysis shows ARID1A-altered patients have prolonged OS under ICI (28 vs 18 months); elevated TMB, TILs, PD-L1"),
    ("MSH2",   "BREAST"):      ("Very Strong", "MSH2 mutations cause dMMR/MSI-H — the most validated pan-cancer ICI biomarker category; FDA-approved pembrolizumab indication"),
    ("BRCA2",  "BREAST"):      ("Moderate",    "BRCA2-mutant tumors show enriched adaptive/innate immunity signatures and better checkpoint blockade response (Nature Cancer 2021)"),
    ("RAD51C", "BREAST"):      ("Moderate",    "HRD gene; DNA repair deficiency and ICI response via TMB is established; RAD51C-specific breast cancer ICI evidence moderate"),
    ("PALB2",  "BREAST"):      ("Moderate",    "HRD gene similar to BRCA2; DDR deficiency-ICI connection established"),
    ("KMT2D",  "pan_cancer"):  ("Moderate",    "KMT2 family mutations associated with better ICI outcomes across 4 ICI cohorts and 10 cancer types (PMC 2024)"),
    ("MET",    "pan_cancer"):  ("Moderate",    "2024 pan-cancer study validates MET mutation as favorable ICI biomarker; MET exon 14 in NSCLC shows enriched immune landscape"),
    ("MET",    "LUNG"):        ("Moderate",    "MET exon 14 skipping in NSCLC associated with enriched immune landscape (JCO PO 2025)"),
    ("SMARCA4","pan_cancer"):  ("Moderate",    "SMARCA4 mutations studied as ICI biomarkers in NSCLC; results mixed but biomarker relevance established"),
    ("SMARCA4","LUNG"):        ("Moderate",    "SMARCA4 mutations in NSCLC studied as ICI biomarkers; prognostic significance published (PMC 2024)"),
    ("VHL",    "pan_cancer"):  ("Moderate",    "VHL mutations drive HIF pathway; complex relationship with ICI response in RCC; pan-cancer moderate evidence"),
    ("SUZ12",  "SOFT_TISSUE"): ("Moderate",    "SUZ12 is PRC2 component; EZH2/PRC2 inhibition epigenetically enhances tumor immunogenicity in sarcoma models"),
    ("CDK12",  "SOFT_TISSUE"): ("Moderate",    "CDK12 mutations produce high neoantigen loads via tandem duplications and improved ICI response in prostate cancer; STS mechanism plausible"),
    ("BRD4",   "pan_cancer"):  ("Indirect",    "BET/BRD4 inhibitors modulate PD-L1 expression and anti-tumor immunity; BRD4 deletion in ICI context is biologically coherent"),
    ("BRD3",   "BREAST"):      ("Indirect",    "BRD3 is BRD4 family member; BET inhibitors modulate PD-L1 and anti-tumor immunity; breast ICI-specific evidence limited"),
    ("NRG1",   "BREAST"):      ("Indirect",    "NRG1 activates HER3/HER2 and affects CAF-mediated TME; tumor microenvironment role documented but ICI-specific breast evidence absent"),
    ("NF1",    "SOFT_TISSUE"): ("Partial",     "NF1 mutations associated with higher TMB and ICI benefit in NSCLC/melanoma; STS-specific evidence limited"),
    ("ATM",    "BRAIN"):       ("Partial",     "ATM-ICI connection via DDR/TMB established pan-cancer; brain-tumor-specific ICI evidence limited"),
    ("YAP1",   "SOFT_TISSUE"): ("Indirect",    "YAP/TAZ Hippo pathway promotes immune exclusion; YAP1 deletion could theoretically enhance anti-tumor immunity"),
    ("MAP2K1", "BREAST"):      ("Partial",     "MAP2K1/2 mutations predict ICI benefit in melanoma; breast-specific evidence absent"),
    ("CDKN2A", "pan_cancer"):  ("Weak",        "CDKN2A is common pan-cancer deletion; limited direct ICI-prognostic evidence"),
    ("CDKN2B", "pan_cancer"):  ("Weak",        "CDKN2B is common deletion co-occurring with CDKN2A; limited direct ICI-prognostic evidence"),
    ("CDKN2A", "BRAIN"):       ("Weak",        "CDKN2A deletion common in GBM; limited ICI-specific prognostic evidence in brain tumors"),
    ("ERBB2",  "SOFT_TISSUE"): ("Weak",        "HER2 deletion in STS is unusual; limited ICI-specific evidence in soft tissue sarcoma"),
    ("RARA",   "SOFT_TISSUE"): ("Weak",        "RARA loss could affect differentiation/immune signaling; no direct ICI evidence in STS"),
    ("ATM",    "SOFT_TISSUE"): ("Weak",        "ATM loss generally predicts ICI benefit via DDR; harm direction in STS not well supported"),
    ("ERG",    "LUNG"):        ("Weak",        "ERG primarily studied in prostate cancer; not established as ICI biomarker in lung cancer"),
    ("NEIL2",  "BREAST"):      ("None",        "No direct literature support for NEIL2 as ICI prognostic biomarker in breast cancer"),
    ("NEIL1",  "BREAST"):      ("None",        "No direct literature support for NEIL1 as ICI prognostic biomarker in breast cancer"),
    ("NKX3-1", "BREAST"):      ("None",        "NKX3-1 primarily studied in prostate cancer; no breast cancer ICI biomarker role established"),
    ("PTK2B",  "BREAST"):      ("None",        "No established literature linking PTK2B deletion to ICI prognosis in breast cancer"),
    ("TRIM37", "BREAST"):      ("None",        "TRIM37 not established as ICI biomarker"),
    ("FANCI",  "BREAST"):      ("None",        "FANCI Fanconi anemia gene; DDR relevance but no direct ICI-specific breast cancer evidence"),
    ("FAH",    "BREAST"):      ("None",        "FAH (fumarylacetoacetase) not established as ICI biomarker"),
    ("SMC3",   "BREAST"):      ("None",        "SMC3 cohesin component; not established as ICI biomarker in breast cancer"),
}

UNASSESSED = ("Unassessed", "Not assessed in literature review")


def lookup_validation(gene: str, cancer_type: str, lookup: dict) -> tuple[str, str]:
    return lookup.get((gene, cancer_type), UNASSESSED)


# ── load tracks ────────────────────────────────────────────────────────────────
print("Loading Track 1 …")
t1 = pd.read_csv(f"{DATA_DIR}/track1_all_significant_hits.csv")
print(f"  {len(t1):,} rows")

print("Loading Track 2 …")
t2 = pd.read_csv(f"{DATA_DIR}/track2_all_significant_hits.csv")
print(f"  {len(t2):,} rows")

# ── annotate Track 1 ───────────────────────────────────────────────────────────
t1["track"] = 1
t1["gene"]  = t1["marker"].apply(extract_gene)

# consistency: count unique (cohort, weight_type) combos per marker × cancer_type
combo_counts = (
    t1.groupby(["marker", "cancer_type"])[["cohort", "weight_type"]]
    .apply(lambda df: df.drop_duplicates().shape[0])
    .reset_index(name="consistency")
)
t1 = t1.merge(combo_counts, on=["marker", "cancer_type"], how="left")

val = t1.apply(
    lambda r: pd.Series(lookup_validation(r["gene"], r["cancer_type"], TRACK1_LOOKUP),
                        index=["validation_level", "validation_notes"]),
    axis=1,
)
t1 = pd.concat([t1, val], axis=1)

# ── annotate Track 2 ───────────────────────────────────────────────────────────
t2["track"]       = 2
t2["gene"]        = t2["marker"].apply(extract_gene)
t2["consistency"] = np.nan  # not applicable for Track 2

val2 = t2.apply(
    lambda r: pd.Series(lookup_validation(r["gene"], r["cancer_type"], TRACK2_LOOKUP),
                        index=["validation_level", "validation_notes"]),
    axis=1,
)
t2 = pd.concat([t2, val2], axis=1)

# ── combine and save ───────────────────────────────────────────────────────────
combined = pd.concat([t1, t2], ignore_index=True, sort=False)
combined.to_csv(OUT_PATH, index=False)
print(f"\nSaved {len(combined):,} rows to:\n  {OUT_PATH}")

# quick summary
print("\nValidation level distribution:")
print(combined.groupby(["track", "validation_level"]).size().to_string())
