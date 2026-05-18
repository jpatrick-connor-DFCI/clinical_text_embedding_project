# Build a comprehensive ICD-10-CM to phecode mapping from the Dana-Farber dataset.
#
# Extracts the official Phecode v1.2 and PhecodeX v1.0 ICD-10-CM mapping tables
# from the R 'Phecode' package (github.com/vcastro/Phecode), then maps every
# unique ICD-10 code in the cohort to phecodes.
# Uses v1.2 as the primary mapping and fills gaps with PhecodeX.
#
# Prerequisites:
#   devtools::install_github("vcastro/Phecode")
#
# Outputs:
#   CODE_PATH/icd10_to_phecode_mapping.csv
#   CODE_PATH/icd10_unmapped_codes.csv
#
# Usage:
#   Rscript generate_icd10_to_phecode_mapping.R

library(Phecode)
library(data.table)

# ── paths ────────────────────────────────────────────────────────────────────
DATA_PATH <- "/data/gusev/USERS/jpconnor/data/clinical_text_embedding_project/"
CODE_PATH <- file.path(DATA_PATH, "code_data")
SURV_PATH <- file.path(DATA_PATH, "time-to-event_analysis")
dir.create(CODE_PATH, showWarnings = FALSE, recursive = TRUE)

# ── helpers ──────────────────────────────────────────────────────────────────
normalize_icd10 <- function(codes) {
  codes <- toupper(trimws(as.character(codes)))
  codes <- gsub("[^A-Z0-9]", "", codes)
  codes[codes == ""] <- NA_character_
  codes
}

normalize_phecode <- function(codes) {
  codes <- trimws(as.character(codes))
  codes <- gsub("[^0-9.]", "", codes)
  codes[codes == ""] <- NA_character_
  codes
}

# ── 1. Load unique ICD-10 codes from dataset ────────────────────────────────
cat("Loading ICD data …\n")
icd_file <- file.path(SURV_PATH, "timestamped_icd_info.csv.gz")
icd_dt <- fread(icd_file, select = "DIAGNOSIS_ICD10_CD")
raw_codes <- unique(na.omit(icd_dt$DIAGNOSIS_ICD10_CD))
cat(sprintf("  %d unique raw ICD-10 codes in dataset\n", length(raw_codes)))

norm_codes <- normalize_icd10(raw_codes)
# lookup: normalized → first raw example
norm_to_raw <- setNames(raw_codes, norm_codes)
norm_to_raw <- norm_to_raw[!is.na(names(norm_to_raw))]
norm_to_raw <- norm_to_raw[!duplicated(names(norm_to_raw))]
unique_norms <- names(norm_to_raw)
cat(sprintf("  %d unique normalized ICD-10 codes\n", length(unique_norms)))

# ── 2. Load mapping tables from R package ────────────────────────────────────
cat("Loading Phecode v1.2 mapping …\n")
v12 <- as.data.table(Phecode_map)
cat(sprintf("  v1.2: %d rows, columns = %s\n", nrow(v12), paste(names(v12), collapse = ", ")))

cat("Loading PhecodeX mapping …\n")
px <- as.data.table(PhecodeX_map)
cat(sprintf("  PhecodeX: %d rows, columns = %s\n", nrow(px), paste(names(px), collapse = ", ")))

# ── 3. Normalize v1.2 mapping ───────────────────────────────────────────────
# Identify ICD and phecode columns (handle varying names)
find_col <- function(dt, candidates) {
  cols_lower <- tolower(names(dt))
  for (c in tolower(candidates)) {
    idx <- which(cols_lower == c)
    if (length(idx) > 0) return(names(dt)[idx[1]])
  }
  stop(sprintf("None of [%s] found in [%s]",
               paste(candidates, collapse = ", "),
               paste(names(dt), collapse = ", ")))
}

v12_icd_col <- find_col(v12, c("code", "icd", "icd10cm", "icd10"))
v12_phe_col <- find_col(v12, c("phecode", "phecode_unrolled"))

v12_map <- data.table(
  icd10_norm = normalize_icd10(v12[[v12_icd_col]]),
  phecode    = normalize_phecode(v12[[v12_phe_col]]),
  source     = "phecode_v1.2"
)
# Filter to ICD-10 rows if vocabulary column exists
if ("vocabulary_id" %in% names(v12)) {
  vocab_upper <- toupper(trimws(v12$vocabulary_id))
  keep <- vocab_upper %in% c("ICD10CM", "ICD10")
  v12_map <- v12_map[keep]
}
v12_map <- v12_map[!is.na(icd10_norm) & !is.na(phecode)]
v12_map <- unique(v12_map, by = c("icd10_norm", "phecode"))
cat(sprintf("  v1.2 normalized: %d unique (ICD, phecode) pairs\n", nrow(v12_map)))

# ── 4. Normalize PhecodeX mapping ───────────────────────────────────────────
px_icd_col <- find_col(px, c("code", "icd", "icd10cm", "icd10"))
px_phe_col <- find_col(px, c("phecode", "phecode_unrolled"))

px_map <- data.table(
  icd10_norm = normalize_icd10(px[[px_icd_col]]),
  phecode    = normalize_phecode(px[[px_phe_col]]),
  source     = "phecodeX_v1.0"
)
if ("vocabulary_id" %in% names(px)) {
  vocab_upper <- toupper(trimws(px$vocabulary_id))
  keep <- vocab_upper %in% c("ICD10CM", "ICD10")
  px_map <- px_map[keep]
}
px_map <- px_map[!is.na(icd10_norm) & !is.na(phecode)]
px_map <- unique(px_map, by = c("icd10_norm", "phecode"))
cat(sprintf("  PhecodeX normalized: %d unique (ICD, phecode) pairs\n", nrow(px_map)))

# ── 5. Merge: v1.2 primary, PhecodeX fills gaps ─────────────────────────────
v12_covered <- unique(v12_map$icd10_norm)
px_gap <- px_map[!icd10_norm %in% v12_covered]
combined <- rbind(v12_map, px_gap)
cat(sprintf("  Combined: %d pairs (%d from v1.2 + %d gap-filled from PhecodeX)\n",
            nrow(combined), nrow(v12_map), nrow(px_gap)))

# ── 6. Parent-code fallback ─────────────────────────────────────────────────
combined_norms <- unique(combined$icd10_norm)
unmapped <- setdiff(unique_norms, combined_norms)
cat(sprintf("  %d codes still unmapped — trying parent codes …\n", length(unmapped)))

parent_rows <- list()
newly_mapped <- 0L
for (code in sort(unmapped)) {
  matched <- FALSE
  # Try progressively shorter prefixes: e.g. E1165 → E116 → E11
  for (plen in (nchar(code) - 1):3) {
    parent <- substr(code, 1, plen)
    hits <- combined[icd10_norm == parent, unique(phecode)]
    if (length(hits) > 0) {
      parent_rows[[length(parent_rows) + 1]] <- data.table(
        icd10_norm = code,
        phecode    = hits,
        source     = "parent_code_match"
      )
      newly_mapped <- newly_mapped + 1L
      matched <- TRUE
      break
    }
  }
}
if (length(parent_rows) > 0) {
  combined <- rbind(combined, rbindlist(parent_rows))
}
cat(sprintf("  Parent-code matching recovered %d additional codes\n", newly_mapped))

# ── 7. Filter to dataset codes ──────────────────────────────────────────────
final <- combined[icd10_norm %in% unique_norms]
final <- unique(final, by = c("icd10_norm", "phecode"))
mapped_norms <- unique(final$icd10_norm)
still_unmapped <- setdiff(unique_norms, mapped_norms)

cat("\n=== Summary ===\n")
cat(sprintf("  Total unique ICD-10 codes in dataset:  %d\n", length(unique_norms)))
cat(sprintf("  Mapped to at least one phecode:        %d (%.1f%%)\n",
            length(mapped_norms), 100 * length(mapped_norms) / length(unique_norms)))
cat(sprintf("  Unmapped:                              %d (%.1f%%)\n",
            length(still_unmapped), 100 * length(still_unmapped) / length(unique_norms)))
cat(sprintf("  Total (ICD-10, phecode) pairs:         %d\n", nrow(final)))
cat(sprintf("  Unique phecodes:                       %d\n", length(unique(final$phecode))))

# ── 8. Write outputs ────────────────────────────────────────────────────────
output <- final[, .(icd10_code = icd10_norm, phecode, source)]
output <- output[order(icd10_code, phecode)]

out_path <- file.path(CODE_PATH, "icd10_to_phecode_mapping.csv")
fwrite(output, out_path)
cat(sprintf("\nWrote mapping: %s  (%d rows)\n", out_path, nrow(output)))

if (length(still_unmapped) > 0) {
  unmapped_dt <- data.table(
    icd10_code  = sort(still_unmapped),
    raw_example = norm_to_raw[sort(still_unmapped)]
  )
  unmapped_path <- file.path(CODE_PATH, "icd10_unmapped_codes.csv")
  fwrite(unmapped_dt, unmapped_path)
  cat(sprintf("Wrote unmapped codes: %s  (%d rows)\n", unmapped_path, nrow(unmapped_dt)))
}
