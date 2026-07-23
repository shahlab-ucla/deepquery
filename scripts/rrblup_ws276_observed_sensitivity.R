#!/usr/bin/env Rscript

# rrBLUP runner for the predeclared current-WS276 marker-universe sensitivity.
# This lane is exploratory, reuses the published phenotype, and cannot support
# reproduction, replication, or biological claims.

stop_with <- function(message) {
  stop(message, call. = FALSE)
}

args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 7L) {
  stop_with(paste(
    "usage: rrblup_ws276_observed_sensitivity.R GENOTYPE_MATRIX",
    "PHENOTYPE_DIR OUTPUT_DIR P3D CORES KINSHIP_RDS EXPECTED_MATRIX_SHA256"
  ))
}

if (!requireNamespace("rrBLUP", quietly = TRUE)) {
  stop_with("the rrBLUP package is required")
}

genotype_path <- normalizePath(args[[1]], mustWork = TRUE)
phenotype_dir <- normalizePath(args[[2]], mustWork = TRUE)
output_arg <- args[[3]]
p3d_text <- toupper(args[[4]])
if (!(p3d_text %in% c("TRUE", "FALSE"))) {
  stop_with("P3D must be exactly TRUE or FALSE")
}
p3d <- identical(p3d_text, "TRUE")

cores <- suppressWarnings(as.integer(args[[5]]))
if (is.na(cores) || cores < 1L) {
  stop_with("CORES must be a positive integer")
}

kinship_arg <- args[[6]]
expected_matrix_sha256 <- tolower(args[[7]])
if (!grepl("^[0-9a-f]{64}$", expected_matrix_sha256)) {
  stop_with("EXPECTED_MATRIX_SHA256 must be a lowercase SHA-256 digest")
}

sha256_file <- function(path) {
  output <- system2("sha256sum", args = shQuote(path), stdout = TRUE, stderr = TRUE)
  status <- attr(output, "status")
  if ((!is.null(status) && status != 0L) || length(output) != 1L) {
    stop_with(paste("sha256sum failed for", path))
  }
  digest <- tolower(strsplit(output[[1]], "[[:space:]]+")[[1]][[1]])
  if (!grepl("^[0-9a-f]{64}$", digest)) {
    stop_with(paste("sha256sum returned malformed output for", path))
  }
  digest
}

observed_matrix_sha256 <- sha256_file(genotype_path)
if (!identical(observed_matrix_sha256, expected_matrix_sha256)) {
  stop_with("genotype matrix SHA-256 differs from the shell-verified receipt")
}

output_parent <- normalizePath(dirname(output_arg), mustWork = TRUE)
output_dir <- file.path(output_parent, basename(output_arg))
if (file.exists(output_dir)) {
  stop_with(paste("refusing existing OUTPUT_DIR:", output_dir))
}
if (!dir.create(output_dir, recursive = FALSE, showWarnings = FALSE)) {
  stop_with(paste("could not create OUTPUT_DIR:", output_dir))
}

kinship_parent <- normalizePath(dirname(kinship_arg), mustWork = TRUE)
kinship_path <- file.path(kinship_parent, basename(kinship_arg))

genotype <- read.delim(
  genotype_path,
  header = TRUE,
  sep = "\t",
  quote = "",
  comment.char = "",
  check.names = FALSE,
  stringsAsFactors = FALSE,
  na.strings = c("NA")
)

if (ncol(genotype) < 5L ||
    !identical(names(genotype)[1:4], c("CHROM", "POS", "REF", "ALT"))) {
  stop_with("genotype matrix must start with CHROM, POS, REF, ALT")
}
if (nrow(genotype) != 75852L) {
  stop_with(sprintf("genotype matrix has %d rows; expected exactly 75852", nrow(genotype)))
}

sample_names <- names(genotype)[-(1:4)]
if (length(sample_names) != 209L || anyDuplicated(sample_names)) {
  stop_with("genotype matrix must contain exactly 209 unique sample columns")
}
if (anyDuplicated(paste(genotype$CHROM, genotype$POS, sep = ":"))) {
  stop_with("genotype matrix contains duplicate CHROM:POS markers")
}
if (!("MtDNA" %in% as.character(genotype$CHROM))) {
  stop_with("MtDNA is absent; the predeclared sensitivity preserves MtDNA")
}

genotype_values <- data.matrix(genotype[, -(1:4), drop = FALSE])
allowed_values <- is.na(genotype_values) | genotype_values == -1 | genotype_values == 1
if (!all(allowed_values)) {
  stop_with("genotypes must be encoded only as -1, 1, or NA")
}

# Preserve the historical Run_Mappings.R behavior: markers containing any
# heterozygous/unsupported calls (encoded NA) are removed before A.mat/GWAS.
complete_marker <- complete.cases(genotype_values)
analysis_genotype <- genotype[complete_marker, , drop = FALSE]
analysis_values <- genotype_values[complete_marker, , drop = FALSE]
if (nrow(analysis_genotype) != 75831L) {
  stop_with(sprintf(
    "complete-case genotype matrix has %d rows; expected exactly 75831",
    nrow(analysis_genotype)
  ))
}
if (!("MtDNA" %in% as.character(analysis_genotype$CHROM))) {
  stop_with("MtDNA is absent after complete-case removal")
}

marker_ids <- paste(analysis_genotype$CHROM, analysis_genotype$POS, sep = "_")
if (anyDuplicated(marker_ids)) {
  stop_with("analysis marker IDs are not unique")
}

if (file.exists(kinship_path)) {
  kinship_payload <- readRDS(kinship_path)
  required_payload <- c(
    "schema_version", "sample_names", "marker_ids",
    "genotype_matrix_sha256", "kinship"
  )
  if (!is.list(kinship_payload) || !all(required_payload %in% names(kinship_payload))) {
    stop_with("existing KINSHIP_RDS is not a compatible sensitivity payload")
  }
  if (!identical(
      kinship_payload$schema_version,
      "wormctx-ws276-observed-sensitivity-kinship-1.0"
  )) {
    stop_with("existing kinship payload has the wrong schema")
  }
  if (!identical(kinship_payload$genotype_matrix_sha256, observed_matrix_sha256)) {
    stop_with("existing kinship matrix was built from a different genotype matrix")
  }
  if (!identical(kinship_payload$sample_names, sample_names)) {
    stop_with("existing kinship sample order differs from genotype matrix")
  }
  if (!identical(kinship_payload$marker_ids, marker_ids)) {
    stop_with("existing kinship marker set differs from genotype matrix")
  }
  kinship <- kinship_payload$kinship
  kinship_action <- "reused"
} else {
  kinship <- rrBLUP::A.mat(t(analysis_values), n.core = cores)
  kinship_payload <- list(
    schema_version = "wormctx-ws276-observed-sensitivity-kinship-1.0",
    sample_names = sample_names,
    marker_ids = marker_ids,
    genotype_matrix_sha256 = observed_matrix_sha256,
    kinship = kinship
  )
  saveRDS(kinship_payload, kinship_path, version = 3)
  kinship_action <- "created"
}

if (!identical(dim(kinship), c(209L, 209L))) {
  stop_with("kinship matrix is not 209 by 209")
}
if (!identical(rownames(kinship), sample_names) ||
    !identical(colnames(kinship), sample_names)) {
  stop_with("kinship sample names/order differ from genotype matrix")
}

gwas_genotype <- data.frame(
  marker = marker_ids,
  CHROM = analysis_genotype$CHROM,
  POS = analysis_genotype$POS,
  analysis_genotype[, -(1:4), drop = FALSE],
  check.names = FALSE,
  stringsAsFactors = FALSE
)

traits <- list(
  mean_EXT = "mean.EXT",
  mean_TOF = "mean.TOF",
  mean_norm_EXT = "mean.norm.EXT",
  norm_n = "norm.n"
)

metadata <- data.frame(
  key = c(
    "classification", "strict_reproduction", "biological_claims_permitted",
    "P3D", "cores", "rrBLUP_version", "input_samples", "input_markers",
    "complete_markers", "MtDNA_preserved", "genotype_matrix_sha256",
    "kinship_action", "kinship_path"
  ),
  value = c(
    "exploratory_current_ws276_isotype_marker_universe_sensitivity",
    "false", "false", p3d_text, as.character(cores),
    as.character(utils::packageVersion("rrBLUP")),
    as.character(length(sample_names)), as.character(nrow(genotype)),
    as.character(nrow(analysis_genotype)), "true", observed_matrix_sha256,
    kinship_action, kinship_path
  ),
  stringsAsFactors = FALSE
)

set.seed(1L)
for (slug in names(traits)) {
  trait_name <- traits[[slug]]
  phenotype_path <- file.path(phenotype_dir, paste0(slug, ".tsv"))
  if (!file.exists(phenotype_path)) {
    stop_with(paste("missing phenotype file:", phenotype_path))
  }
  phenotype <- read.delim(
    phenotype_path,
    header = TRUE,
    sep = "\t",
    quote = "",
    comment.char = "",
    check.names = FALSE,
    stringsAsFactors = FALSE,
    na.strings = c("NA")
  )
  if (!identical(names(phenotype), c("strain", trait_name))) {
    stop_with(sprintf("%s must have header strain<TAB>%s", phenotype_path, trait_name))
  }
  if (nrow(phenotype) != 209L || anyDuplicated(phenotype$strain) || anyNA(phenotype)) {
    stop_with(paste("phenotype file must contain 209 unique, complete rows:", phenotype_path))
  }
  if (!identical(sort(as.character(phenotype$strain)), sort(sample_names))) {
    stop_with(paste("phenotype and genotype sample sets differ:", phenotype_path))
  }
  phenotype[[trait_name]] <- suppressWarnings(as.numeric(phenotype[[trait_name]]))
  if (anyNA(phenotype[[trait_name]]) || any(!is.finite(phenotype[[trait_name]]))) {
    stop_with(paste("phenotype values must be finite numeric values:", phenotype_path))
  }

  raw_mapping <- rrBLUP::GWAS(
    pheno = phenotype,
    geno = gwas_genotype,
    K = kinship,
    min.MAF = 0.05,
    n.core = cores,
    P3D = p3d,
    plot = FALSE
  )
  if (!is.data.frame(raw_mapping) || ncol(raw_mapping) != 4L ||
      nrow(raw_mapping) == 0L) {
    stop_with(paste("rrBLUP returned a malformed mapping for", trait_name))
  }
  names(raw_mapping) <- c("marker", "CHROM", "POS", "log10p")
  if (anyDuplicated(raw_mapping$marker) || anyNA(raw_mapping$log10p)) {
    stop_with(paste("rrBLUP returned duplicate markers or NA statistics for", trait_name))
  }
  output_path <- file.path(output_dir, paste0(slug, "_raw_mapping.tsv"))
  write.table(
    raw_mapping,
    file = output_path,
    sep = "\t",
    quote = FALSE,
    row.names = FALSE,
    col.names = TRUE,
    na = "NA"
  )
  metadata <- rbind(
    metadata,
    data.frame(
      key = paste0(slug, "_tested_markers"),
      value = as.character(nrow(raw_mapping)),
      stringsAsFactors = FALSE
    )
  )
}

write.table(
  metadata,
  file = file.path(output_dir, "run_metadata.tsv"),
  sep = "\t",
  quote = FALSE,
  row.names = FALSE,
  col.names = TRUE
)
sink(file.path(output_dir, "R_sessionInfo.txt"))
print(sessionInfo())
sink()
