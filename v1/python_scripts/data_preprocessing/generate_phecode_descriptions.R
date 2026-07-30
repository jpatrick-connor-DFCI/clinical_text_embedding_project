# Build a phecode -> human-readable description lookup from the R 'Phecode' package.
#
# Companion to generate_icd10_to_phecode_mapping.R: pulls the dedicated Phecode
# v1.2 and PhecodeX v1.0 definition tables and extracts each phecode's description,
# deduplicated to one row per phecode. v1.2 is primary; PhecodeX fills in phecodes
# v1.2 doesn't have. (The *_map tables do not contain description columns.)
#
# Prerequisites:
#   devtools::install_github("vcastro/Phecode")
#
# Output:
#   CODE_PATH/phecode_descriptions.csv   (phecode, description, source)
#
# Usage:
#   Rscript generate_phecode_descriptions.R

library(Phecode)
library(data.table)

# ── paths ────────────────────────────────────────────────────────────────────
DATA_PATH <- "/data/gusev/USERS/jpconnor/data/clinical_text_embedding_project/"
CODE_PATH <- file.path(DATA_PATH, "code_data")
dir.create(CODE_PATH, showWarnings = FALSE, recursive = TRUE)

# ── helpers ──────────────────────────────────────────────────────────────────
# Mirrors generate_icd10_to_phecode_mapping.R::normalize_phecode so lookups agree
# with the phecode identifiers already written by _map_events_to_columns() there
# (trailing ".0"/"0" trimmed, e.g. "250.20" -> "250.2").
normalize_phecode <- function(codes) {
  codes <- trimws(as.character(codes))
  codes <- gsub("[^0-9.]", "", codes)
  codes[codes == ""] <- NA_character_
  # CSV type inference turns zero-padded phecodes such as "059" into "59".
  # Canonicalize them here so description keys match generated endpoint names.
  codes <- sub("^0+(?=[0-9])", "", codes, perl = TRUE)
  has_dot <- grepl("\\.", codes)
  codes[has_dot] <- sub("0+$", "", codes[has_dot])
  codes[has_dot] <- sub("\\.$", "", codes[has_dot])
  codes
}

# Some phecode description strings in the source tables carry invalid UTF-8
# bytes (observed on phecode 995's description), which crashes data.table's
# trimws()/sub() with "input string N is invalid UTF-8". Force-repair the
# encoding (replacing any unrepresentable bytes) before any string op touches it.
clean_utf8 <- function(x) {
  enc2utf8(iconv(as.character(x), from = "UTF-8", to = "UTF-8", sub = "byte"))
}

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

# ── 1. Load definition tables from R package ─────────────────────────────────
cat("Loading Phecode v1.2 definitions …\n")
v12 <- as.data.table(Phecode_definitions)
cat(sprintf("  v1.2: %d rows, columns = %s\n", nrow(v12), paste(names(v12), collapse = ", ")))

cat("Loading PhecodeX definitions …\n")
px <- as.data.table(PhecodeX_definitions)
cat(sprintf("  PhecodeX: %d rows, columns = %s\n", nrow(px), paste(names(px), collapse = ", ")))

# ── 2. Extract (phecode, description) pairs from each table ────────────────
v12_phe_col <- find_col(v12, c("phecode", "phecode_unrolled"))
v12_desc_col <- find_col(v12, c("phecode_desc", "phecode_string", "phenotype", "description", "phecode_description"))
v12_desc <- data.table(
  phecode     = normalize_phecode(v12[[v12_phe_col]]),
  description = trimws(clean_utf8(v12[[v12_desc_col]])),
  source      = "phecode_v1.2"
)

px_phe_col <- find_col(px, c("phecode", "phecode_unrolled"))
px_desc_col <- find_col(px, c("phecode_string", "phecode_desc", "phenotype", "description", "phecode_description"))
px_desc <- data.table(
  phecode     = normalize_phecode(px[[px_phe_col]]),
  description = trimws(clean_utf8(px[[px_desc_col]])),
  source      = "phecodeX_v1.0"
)

v12_desc <- v12_desc[!is.na(phecode) & nzchar(description)]
px_desc  <- px_desc[!is.na(phecode) & nzchar(description)]
v12_desc <- unique(v12_desc, by = "phecode")
px_desc  <- unique(px_desc, by = "phecode")

# ── 3. Merge: v1.2 primary, PhecodeX fills gaps ─────────────────────────────
px_gap <- px_desc[!phecode %in% v12_desc$phecode]
combined <- rbind(v12_desc, px_gap)
combined <- combined[order(as.numeric(phecode))]
cat(sprintf("  Combined: %d unique phecodes (%d from v1.2 + %d gap-filled from PhecodeX)\n",
            nrow(combined), nrow(v12_desc), nrow(px_gap)))

# ── 4. Write output ──────────────────────────────────────────────────────────
out_path <- file.path(CODE_PATH, "phecode_descriptions.csv")
fwrite(combined, out_path)
cat(sprintf("\nWrote phecode descriptions: %s  (%d rows)\n", out_path, nrow(combined)))
