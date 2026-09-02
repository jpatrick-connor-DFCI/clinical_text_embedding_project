# Install R packages required by the manuscript-figure rendering pipeline.
#
# One-time bootstrap; safe to re-run (skips already-installed packages).
#   Rscript R/install_packages.R   (run from the repo root)

CRAN <- "https://cloud.r-project.org"

required <- c(
  # Core ggplot stack
  "ggplot2", "patchwork", "scales",
  # Tidyverse data wrangling + IO
  "dplyr", "tidyr", "readr", "forcats", "tibble", "stringr", "purrr",
  # Survival (KMs for Figs 2D, 2E, 4B, 5D)
  "survival", "ggsurvfit",
  # Significance stars / brackets (Figs 1C, 2B, 3D)
  "ggsignif",
  # Correlation heatmap (Fig 3B)
  "ggcorrplot",
  # Fig 1A schematic + Fig 5A inset
  "cowplot", "viridisLite",
  # Top-Delta labels in Fig 2A
  "ggrepel",
  # shared/palette.json loader (figure_utils.R)
  "jsonlite"
)

installed <- rownames(installed.packages())
missing <- setdiff(required, installed)

if (length(missing) == 0) {
  message("All required R packages are already installed.")
} else {
  message("Installing: ", paste(missing, collapse = ", "))
  install.packages(missing, repos = CRAN)
}

# Verify by loading
invisible(lapply(required, function(pkg) {
  if (!requireNamespace(pkg, quietly = TRUE)) {
    stop(sprintf("Package '%s' failed to install or load", pkg))
  }
}))
message("OK — all packages loadable.")
