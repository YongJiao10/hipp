#!/usr/bin/env Rscript

# Fit the exploratory Native-space JJ_TMS L/R hippocampal surface beta models.
# This fits healthy and patient groups separately, with separate branches for
# individual template matching and HCP fixed atlas.
# This is not a numbered formal-workflow entry point.

args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 1L || length(args) > 2L) {
  stop("usage: jj_tms_sptms_native_surface_model.R <run_dir> [output_dir]")
}

run_dir <- normalizePath(args[[1]], mustWork = TRUE)
output_root <- if (length(args) == 2L) {
  args[[2]]
} else {
  file.path(run_dir, "models")
}
dir.create(output_root, recursive = TRUE, showWarnings = FALSE)

if (!requireNamespace("nlme", quietly = TRUE)) {
  stop("required R package nlme is unavailable")
}

subject_csv <- file.path(run_dir, "subject_network_beta.csv")
site_csv <- file.path(run_dir, "site_level_beta.csv")
if (!file.exists(subject_csv) || !file.exists(site_csv)) {
  stop("run directory must contain subject_network_beta.csv and site_level_beta.csv")
}

network_levels <- c("Control", "Default", "VentralAttention")
cluster_levels <- c("1", "2", "3")
hipp_levels <- c("L", "R")
target_levels <- c("L", "R")
group_levels <- c("healthy", "patient")
parcellation_levels <- c("template_matching", "hcp")

read_beta_table <- function(path, source_name) {
  data <- read.csv(path, stringsAsFactors = FALSE, check.names = FALSE)
  required <- c(
    "subject", "group", "site_network", "target_hemi", "hippocampal_hemi",
    "cluster", "parcellation", "beta_mean"
  )
  missing <- setdiff(required, names(data))
  if (length(missing) > 0L) {
    stop(paste(source_name, "is missing columns:", paste(missing, collapse = ",")))
  }
  if ("bilateral" %in% tolower(unique(as.character(data$hippocampal_hemi)))) {
    stop(paste(source_name, "contains forbidden bilateral rows"))
  }
  data$beta_mean <- suppressWarnings(as.numeric(data$beta_mean))
  data$cluster <- as.character(data$cluster)
  data$hippocampal_hemi <- as.character(data$hippocampal_hemi)
  data$target_hemi <- as.character(data$target_hemi)
  data$site_network <- as.character(data$site_network)
  data$group <- as.character(data$group)
  data$parcellation <- as.character(data$parcellation)
  data$source_table <- source_name
  data <- data[is.finite(data$beta_mean), , drop = FALSE]
  data <- data[
    data$group %in% group_levels &
      data$site_network %in% network_levels &
      data$cluster %in% cluster_levels &
      data$hippocampal_hemi %in% hipp_levels &
      data$target_hemi %in% target_levels &
      data$parcellation %in% parcellation_levels,
    , drop = FALSE
  ]
  if (nrow(data) == 0L) {
    stop(paste(source_name, "has no finite rows after the JJ_TMS analysis filters"))
  }
  data
}

safe_sd <- function(values) {
  if (length(values) < 2L) return(NA_real_)
  sd(values)
}

cell_descriptives <- function(data) {
  groups <- split(
    data,
    interaction(
      data$site_network, data$cluster, data$hippocampal_hemi,
      drop = TRUE, sep = "\r"
    )
  )
  rows <- lapply(groups, function(cell) {
    n <- nrow(cell)
    mean_value <- mean(cell$beta_mean)
    sd_value <- safe_sd(cell$beta_mean)
    se <- if (n > 1L) sd_value / sqrt(n) else NA_real_
    critical <- if (n > 1L) qt(0.975, df = n - 1L) else NA_real_
    data.frame(
      site_network = cell$site_network[[1]],
      cluster = cell$cluster[[1]],
      hippocampal_hemi = cell$hippocampal_hemi[[1]],
      n_observations = n,
      n_subjects = length(unique(cell$subject)),
      beta_mean = mean_value,
      beta_sd = sd_value,
      beta_ci95_low = mean_value - critical * se,
      beta_ci95_high = mean_value + critical * se,
      stringsAsFactors = FALSE
    )
  })
  do.call(rbind, rows)
}

model_df <- function(data) {
  data$subject <- factor(data$subject)
  data$site_network <- factor(data$site_network, levels = network_levels)
  data$cluster <- factor(data$cluster, levels = cluster_levels)
  data$hippocampal_hemi <- factor(data$hippocampal_hemi, levels = hipp_levels)
  data$target_hemi <- factor(data$target_hemi, levels = target_levels)
  data
}

get_formula <- function(data) {
  if (length(unique(data$target_hemi)) > 1L) {
    beta_mean ~ site_network * cluster * hippocampal_hemi +
      target_hemi * cluster + target_hemi * hippocampal_hemi
  } else {
    beta_mean ~ site_network * cluster * hippocampal_hemi
  }
}

fit_lme <- function(data) {
  if (length(unique(data$site_network)) != 3L) {
    stop("all three fixed networks are required")
  }
  if (length(unique(data$cluster)) != 3L || length(unique(data$hippocampal_hemi)) != 2L) {
    stop("all three clusters and both hippocampal hemispheres are required")
  }
  nlme::lme(
    fixed = get_formula(data),
    random = ~1 | subject / hippocampal_hemi,
    data = model_df(data),
    method = "REML",
    na.action = na.fail,
    control = nlme::lmeControl(opt = "optim")
  )
}

write_fixed_effects <- function(model, output_dir) {
  coefficients <- as.data.frame(summary(model)$tTable)
  coefficients$term <- rownames(coefficients)
  rownames(coefficients) <- NULL
  write.csv(coefficients, file.path(output_dir, "fixed_effect_coefficients.csv"), row.names = FALSE)
  omnibus <- as.data.frame(anova(model, type = "marginal"))
  omnibus$term <- rownames(omnibus)
  rownames(omnibus) <- NULL
  write.csv(omnibus, file.path(output_dir, "fixed_effect_omnibus.csv"), row.names = FALSE)
  main_term <- omnibus[grepl("site_network:cluster:hippocampal_hemi", omnibus$term, fixed = TRUE), , drop = FALSE]
  write.csv(main_term, file.path(output_dir, "main_site_network_cluster_hippocampal_hemi_test.csv"), row.names = FALSE)
}

contrast_vector <- function(model, network_a, network_b, cluster, hippocampal_hemi, formula_obj, has_target_hemi) {
  if (has_target_hemi) {
    grid <- expand.grid(
      site_network = c(network_a, network_b),
      cluster = cluster,
      hippocampal_hemi = hippocampal_hemi,
      target_hemi = target_levels,
      KEEP.OUT.ATTRS = FALSE,
      stringsAsFactors = FALSE
    )
    grid$target_hemi <- factor(grid$target_hemi, levels = target_levels)
  } else {
    grid <- expand.grid(
      site_network = c(network_a, network_b),
      cluster = cluster,
      hippocampal_hemi = hippocampal_hemi,
      KEEP.OUT.ATTRS = FALSE,
      stringsAsFactors = FALSE
    )
  }
  grid$site_network <- factor(grid$site_network, levels = network_levels)
  grid$cluster <- factor(grid$cluster, levels = cluster_levels)
  grid$hippocampal_hemi <- factor(grid$hippocampal_hemi, levels = hipp_levels)
  terms_obj <- delete.response(terms(formula_obj))
  design <- model.matrix(terms_obj, grid)
  fixed_names <- names(nlme::fixef(model))
  design <- design[, fixed_names, drop = FALSE]
  first <- colMeans(design[grid$site_network == network_a, , drop = FALSE])
  second <- colMeans(design[grid$site_network == network_b, , drop = FALSE])
  first - second
}

planned_contrasts <- function(model, data) {
  pairs <- list(
    c("Control", "Default"),
    c("Control", "VentralAttention"),
    c("Default", "VentralAttention")
  )
  formula_obj <- get_formula(data)
  has_target_hemi <- length(unique(data$target_hemi)) > 1L
  beta <- nlme::fixef(model)
  covariance <- vcov(model)
  residual_df <- max(1, model$dims$N - length(beta))
  rows <- list()
  row_index <- 1L
  for (cluster in cluster_levels) {
    for (hemi in hipp_levels) {
      for (pair in pairs) {
        vector <- contrast_vector(model, pair[[1]], pair[[2]], cluster, hemi, formula_obj, has_target_hemi)
        estimate <- as.numeric(sum(vector * beta))
        variance <- as.numeric(vector %*% covariance %*% vector)
        se <- sqrt(max(variance, 0))
        statistic <- if (se > 0) estimate / se else NA_real_
        p_value <- if (is.finite(statistic)) {
          2 * pt(abs(statistic), df = residual_df, lower.tail = FALSE)
        } else {
          NA_real_
        }
        critical <- qt(0.975, df = residual_df)
        rows[[row_index]] <- data.frame(
          contrast = paste(pair[[1]], "-", pair[[2]]),
          network_a = pair[[1]],
          network_b = pair[[2]],
          cluster = cluster,
          hippocampal_hemi = hemi,
          target_hemi_marginalized = has_target_hemi,
          estimate = estimate,
          std_error = se,
          df = residual_df,
          statistic = statistic,
          p_value_raw = p_value,
          p_value_holm = NA_real_,
          ci95_low = estimate - critical * se,
          ci95_high = estimate + critical * se,
          stringsAsFactors = FALSE
        )
        row_index <- row_index + 1L
      }
    }
  }
  result <- do.call(rbind, rows)
  result$p_value_holm <- p.adjust(result$p_value_raw, method = "holm")
  result
}

fit_branch <- function(data, source_name, group_name, parcellation_name, output_dir) {
  branch_dir <- file.path(output_dir, group_name, parcellation_name, source_name)
  dir.create(branch_dir, recursive = TRUE, showWarnings = FALSE)
  branch <- data[
    data$group == group_name & data$parcellation == parcellation_name,
    , drop = FALSE
  ]
  if (nrow(branch) == 0L) {
    stop(paste("no finite rows for", source_name, group_name, parcellation_name))
  }
  model <- fit_lme(branch)
  write.csv(model_df(branch), file.path(branch_dir, "model_input.csv"), row.names = FALSE)
  write.csv(cell_descriptives(branch), file.path(branch_dir, "cell_descriptives.csv"), row.names = FALSE)
  write_fixed_effects(model, branch_dir)
  write.csv(planned_contrasts(model, branch), file.path(branch_dir, "planned_network_pairwise_contrasts_holm.csv"), row.names = FALSE)
  capture.output(summary(model), file = file.path(branch_dir, "model_summary.txt"))
  capture.output(anova(model, type = "marginal"), file = file.path(branch_dir, "model_anova.txt"))
  capture.output(planned_contrasts(model, branch), file.path(branch_dir, "planned_network_pairwise_contrasts_holm.txt"))
  data.frame(
    source = source_name,
    group = group_name,
    parcellation = parcellation_name,
    n_rows = nrow(branch),
    n_subjects = length(unique(branch$subject)),
    n_networks = length(unique(branch$site_network)),
    status = "complete",
    error = NA_character_,
    stringsAsFactors = FALSE
  )
}

subject_data <- read_beta_table(subject_csv, "subject_network_beta.csv")
site_data <- read_beta_table(site_csv, "site_level_beta.csv")

branch_status <- list()
status_index <- 1L
errors <- character()
for (group_name in group_levels) {
  for (parcellation_name in parcellation_levels) {
    for (source_entry in list(
      list(name = "primary_subject_network", data = subject_data),
      list(name = "sensitivity_site_level", data = site_data)
    )) {
      result <- tryCatch(
        fit_branch(source_entry$data, source_entry$name, group_name, parcellation_name, output_root),
        error = function(error) {
          errors <<- c(errors, paste(source_entry$name, group_name, parcellation_name, ":", conditionMessage(error)))
          data.frame(
            source = source_entry$name,
            group = group_name,
            parcellation = parcellation_name,
            n_rows = NA_integer_,
            n_subjects = NA_integer_,
            n_networks = NA_integer_,
            status = "failed",
            error = conditionMessage(error),
            stringsAsFactors = FALSE
          )
        }
      )
      branch_status[[status_index]] <- result
      status_index <- status_index + 1L
    }
  }
}

status_table <- do.call(rbind, branch_status)
write.csv(status_table, file.path(output_root, "branch_status.csv"), row.names = FALSE)
manifest <- list(
  status = if (length(errors) == 0L) "complete" else "failed",
  input_run_dir = run_dir,
  random_formula = "~1 | subject / hippocampal_hemi",
  groups_fitted_separately = group_levels,
  parcellations = parcellation_levels,
  source_tables = c(primary = subject_csv, sensitivity = site_csv),
  main_test = "site_network:cluster:hippocampal_hemi",
  health_patient_interaction = FALSE,
  pairwise_adjustment = "Holm within each group x parcellation x source branch",
  native_space_input = TRUE,
  surface_to_volume_executed = FALSE,
  errors = errors
)
jsonlite_available <- requireNamespace("jsonlite", quietly = TRUE)
if (jsonlite_available) {
  jsonlite::write_json(manifest, file.path(output_root, "model_manifest.json"), pretty = TRUE, auto_unbox = TRUE)
} else {
  writeLines(capture.output(dput(manifest)), con = file.path(output_root, "model_manifest.dput"))
}

if (length(errors) > 0L) {
  stop(paste(errors, collapse = "\n"))
}
