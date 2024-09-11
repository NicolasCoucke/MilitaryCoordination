#load in packages
library(ggplot2)
library(dplyr)
library(httpgd)
library(readr)
library(readxl)
library(tidyverse)
library(openxlsx)
library(lme4)
library(lmerTest)
library(car)
library(emmeans)
library(BayesFactor)
library(patchwork)


file_path <- "C:/Users/nicoucke/OneDrive - UGent/Desktop/Hyperscanning 1/individual_ROI_results.csv"

# Read the Excel file and combine with z scores
data <- read.csv(file_path)

# Define custom colors for groups
custom_colors <- c("civilian" = "orange", "military" = "green")
# Darker color for 'Alone' condition
custom_colors_darker <- c("civilian" = "#f8da2ed5", "military" = "#008000B0")

# Plot the reaction times
data_egalitarian <- subset(data, hierarchy == 'Egalitarian')
data_egalitarian$synchrony <- factor(data_egalitarian$synchrony, levels = c("Alone", "Synchronous", "Complementary"))

y_variables <- c("Theta_MidFrontal", "Alpha_Frontal", "Alpha_Central", "Beta_Central")

# Titles for each subplot
plot_titles <- c("Theta Mid Frontal", "Alpha Frontal", "Alpha Central", "Beta Central")

# Create individual plots for each y variable
plots <- lapply(seq_along(y_variables), function(i) {
  y_var <- y_variables[i]
  plot_title <- plot_titles[i]
  ggplot(data_egalitarian, aes(x = synchrony, y = .data[[y_var]], fill = group)) +
    geom_bar(stat = "summary", fun = "mean", position = position_dodge(width = 0.75), width = 0.7, aes(alpha = synchrony)) +
    geom_errorbar(stat = "summary", fun.data = "mean_se", position = position_dodge(width = 0.75), width = 0.25) +
    geom_point(aes(color = group, alpha = synchrony), position = position_jitterdodge(jitter.width = 0.2), size = 3, stroke = 0.6) +
    ggtitle(plot_title) +
    labs(x = " ", y = 'Power (dB)', fill = "Group", color = "Group") +
    theme_minimal(base_size = 14) +
    scale_x_discrete(limits = c("Alone", "Synchronous", "Complementary")) +
    scale_fill_manual(values = c("civilian" = "#eeb704f3", "military" = "#008000B0")) +
    scale_color_manual(values = c("civilian" = "#eeb704f3", "military" = "#008000B0")) +
    scale_alpha_manual(values = c("Alone" = 0.25, "Synchronous" = 0.5, "Complementary" = 0.5), guide = "none") + # Darken 'Alone' condition
    theme(axis.text.x = element_text(size = 12, angle = 45, hjust = 1),
          axis.text.y = element_text(size = 12),
          axis.title = element_text(size = 14),
          legend.title = element_text(size = 14),
          legend.text = element_text(size = 12))
})

# Combine the plots using patchwork
combined_plot <- plots[[1]] + plots[[2]] + plots[[3]] + plots[[4]] + plot_layout(ncol = 2)

# Print the combined plot
print(combined_plot)

# Fit the repeated measures ANOVA and perform post-hoc tests for each


# Fit the repeated measures ANOVA and perform post-hoc tests for each y variable
results <- lapply(y_variables, function(y_var) {
  formula <- as.formula(paste(y_var, "~ synchrony * group + Error(pair/(synchrony * group))"))
  aov_model <- aov(formula, data = data_egalitarian)
  wald_test <- Anova(aov_model, type = "II")
  emm_int <- emmeans(aov_model, ~ group | synchrony)
  pairwise_comparisons_group <- pairs(emm_int)
  emm_int <- emmeans(aov_model, ~ synchrony | group)
  pairwise_comparisons_synchrony <- pairs(emm_int)
  
  list(
    y_var = y_var,
    wald_test = wald_test,
    pairwise_comparisons_group = summary(pairwise_comparisons_group, adjust = "tukey"),
    pairwise_comparisons_synchrony = summary(pairwise_comparisons_synchrony, adjust = "tukey")
  )
})

# Print the statistical results
for (result in results) {
  cat("\n\nResults for:", result$y_var, "\n")
  print(result$wald_test)
  cat("\nPairwise Comparisons (Group):\n")
  print(result$pairwise_comparisons_group)
  cat("\nPairwise Comparisons (Synchrony):\n")
  print(result$pairwise_comparisons_synchrony)
}

# Fit the mixed-effects model and perform post-hoc tests for each y variable
results <- lapply(y_variables, function(y_var) {
  formula <- as.formula(paste(y_var, "~ synchrony * group + (1|pair)"))
  mixed_model <- lmer(formula, data = data_egalitarian)
  wald_test <- Anova(mixed_model, type = "II")
  emm_int <- emmeans(mixed_model, ~ group | synchrony)
  pairwise_comparisons_group <- pairs(emm_int)
  emm_int <- emmeans(mixed_model, ~ synchrony | group)
  pairwise_comparisons_synchrony <- pairs(emm_int)
  
  list(
    y_var = y_var,
    wald_test = wald_test,
    pairwise_comparisons_group = summary(pairwise_comparisons_group, adjust = "tukey"),
    pairwise_comparisons_synchrony = summary(pairwise_comparisons_synchrony, adjust = "tukey")
  )
})

# Print the statistical results
for (result in results) {
  cat("\n\nResults for:", result$y_var, "\n")
  print(result$wald_test)
  cat("\nPairwise Comparisons (Group):\n")
  print(result$pairwise_comparisons_group)
  cat("\nPairwise Comparisons (Synchrony):\n")
  print(result$pairwise_comparisons_synchrony)
}




# Display the combined plot
print(combined_plot)



data_hierarchical <- subset(data, hierarchy == 'Hierarchical')
data_hierarchical$leader <- ifelse(data_hierarchical$leader == "yes", "Leader", 
                           ifelse(data_hierarchical$leader == "no", "Follower", data_hierarchical$leader))
data_hierarchical_synchronous <- subset(data_hierarchical, synchrony == 'Synchronous')

plots_synchronous <- lapply(seq_along(y_variables), function(i) {
  y_var <- y_variables[i]
  plot_title <- plot_titles[i]
  ggplot(data_hierarchical_synchronous, aes(x = leader, y = .data[[y_var]], fill = group)) +
    geom_bar(stat = "summary", fun = "mean", position = position_dodge(width = 0.75), width = 0.7) +
    geom_errorbar(stat = "summary", fun.data = "mean_se", position = position_dodge(width = 0.75), width = 0.25) +
    geom_point(aes(color = group), position = position_jitterdodge(jitter.width = 0.2), size = 3, stroke = 0.6) +
    ggtitle(plot_title) +
    labs(x = " ", y = 'Power (dB)', fill = "Group", color = "Group") +
    theme_minimal(base_size = 14) +
    scale_fill_manual(values = c("civilian" = "#eeb704f3", "military" = "#008000B0")) +
    scale_color_manual(values = c("civilian" = "#eeb704f3", "military" = "#008000B0")) +
    theme(axis.text.x = element_text(size = 12, angle = 45, hjust = 1),
          axis.text.y = element_text(size = 12),
          axis.title = element_text(size = 14),
          legend.title = element_text(size = 14),
          legend.text = element_text(size = 12))
})

# Combine the plots for both conditions using patchwork
combined_plot_synchronous <- plots_synchronous[[1]] + plots_synchronous[[2]] + plots_synchronous[[3]] + plots_synchronous[[4]] +plot_layout(ncol = 2)

# Display the combined plot
print(combined_plot_synchronous)


# Plot the hierarchical data for 'Complementary' condition
data_hierarchical_complementary <- subset(data_hierarchical, synchrony == 'Complementary')

plots_complementary <- lapply(seq_along(y_variables), function(i) {
  y_var <- y_variables[i]
  plot_title <- plot_titles[i]
  ggplot(data_hierarchical_complementary, aes(x = leader, y = .data[[y_var]], fill = group)) +
    geom_bar(stat = "summary", fun = "mean", position = position_dodge(width = 0.75), width = 0.7) +
    geom_errorbar(stat = "summary", fun.data = "mean_se", position = position_dodge(width = 0.75), width = 0.25) +
    geom_point(aes(color = group), position = position_jitterdodge(jitter.width = 0.2), size = 3, stroke = 0.6) +
    ggtitle(plot_title) +
    labs(x = " ", y = 'Power (dB)', fill = "Group", color = "Group") +
    theme_minimal(base_size = 14) +
    scale_fill_manual(values = c("civilian" = "#eeb704f3", "military" = "#008000B0")) +
    scale_color_manual(values = c("civilian" = "#eeb704f3", "military" = "#008000B0")) +
    theme(axis.text.x = element_text(size = 12, angle = 45, hjust = 1),
          axis.text.y = element_text(size = 12),
          axis.title = element_text(size = 14),
          legend.title = element_text(size = 14),
          legend.text = element_text(size = 12))
})

# Combine the plots for both conditions using patchwork
combined_plot_complementary <- plots_complementary[[1]] + plots_complementary[[2]] + plots_complementary[[3]] + plots_complementary[[4]] +plot_layout(ncol = 2)

# Display the combined plot
print(combined_plot_complementary)




# Combine the synchronous and complementary plots along rows
final_combined_plot <- combined_plot_synchronous | combined_plot_complementary

# Display the final combined plot
print(final_combined_plot)










file_path <- "C:/Users/nicoucke/OneDrive - UGent/Desktop/Hyperscanning 1/homologous_ROI_results.csv"

# Read the Excel file and combine with z scores
data <- read.csv(file_path)

# Plot the reaction times
data_egalitarian <- subset(data, hierarchy == 'Egalitarian')
y_variables <- c("Alpha_RightParietal", "Alpha_Frontal", "Alpha_Central", "Beta_Central")


# Titles for each subplot
plot_titles <- c("Alpha Right Parietal", "Alpha Frontal", "Alpha Central", "Beta Central")

# Create individual plots for each y variable
plots <- lapply(seq_along(y_variables), function(i) {
  y_var <- y_variables[i]
  plot_title <- plot_titles[i]
  ggplot(data_egalitarian, aes(x = synchrony, y = .data[[y_var]], fill = group)) +
   geom_boxplot(position = position_dodge(width = 0.75), width = 0.7) +
    #geom_errorbar(stat = "summary", fun.data = "mean_se", position = position_dodge(width = 0.75), width = 0.25) +
    geom_point(aes(color = group), position = position_jitterdodge(jitter.width = 0.2), size = 3, alpha = 0.4,  stroke = 0.6)   + 
    ggtitle(plot_title) +
    labs(x = " ", y = 'Eveloppe correlation (ppc)', fill = "Group", color = "Group") +
    theme_minimal(base_size = 14) +
    scale_x_discrete(limits = c("Alone", "Synchronous", "Complementary")) +
    scale_fill_manual(values = c("civilian" = "#eeb704f3", "military" = "#008000B0")) +
    scale_color_manual(values = c("civilian" = "#eeb704f3", "military" = "#008000B0")) +
    scale_alpha_manual(values = c("Alone" = 0.25, "Synchronous" = 0.5, "Complementary" = 0.5), guide = "none") + # Darken 'Alone' condition
    theme(axis.text.x = element_text(size = 12, angle = 45, hjust = 1),
          axis.text.y = element_text(size = 12),
          axis.title = element_text(size = 14),
          legend.title = element_text(size = 14),
          legend.text = element_text(size = 12))
})

# Combine the plots using patchwork
combined_plot <- plots[[1]] + plots[[2]] + plots[[3]] + plots[[4]] + plot_layout(ncol = 2)
# Print the combined plot
print(combined_plot)


# Create individual plots for each y variable
plots <- lapply(y_variables, function(y_var) {
  ggplot(data_egalitarian, aes(x = synchrony, y = .data[[y_var]], fill = group)) +
    geom_boxplot(position = position_dodge(width = 0.75), width = 0.7) +
    #geom_errorbar(stat = "summary", fun.data = "mean_se", position = position_dodge(width = 0.75), width = 0.25) +
    geom_point(aes(color = group), position = position_jitterdodge(jitter.width = 0.2), size = 3, alpha = 0.4,  stroke = 0.6)   + 
    labs(x = "Condition", y = y_var,
         fill = "Group", color = "Group") +
    theme_minimal(base_size = 14) +
    scale_fill_manual(values = c("civilian" = "#eeb704f3", "military" = "#008000B0")) +
    scale_color_manual(values = c("civilian" = "#eeb704f3", "military" = "#008000B0")) +
    theme(axis.text.x = element_text(size = 12, angle = 45, hjust = 1),
          axis.text.y = element_text(size = 12),
          axis.title = element_text(size = 14),
          legend.title = element_text(size = 14),
          legend.text = element_text(size = 12))
})

# Combine the plots using patchwork
combined_plot <- plots[[1]] + plots[[2]] + plots[[3]] + plots[[4]] + plot_layout(ncol = 2)
print(combined_plot)



# Plot the reaction times
data_egalitarian <- subset(data, synchrony = 'Synchronous')
y_variables <- c("Alpha_RightParietal", "Alpha_Frontal", "Alpha_Central", "Beta_Central")

# Create individual plots for each y variable
plots <- lapply(y_variables, function(y_var) {
  ggplot(data_egalitarian, aes(x = hierarchy, y = .data[[y_var]], fill = group)) +
    geom_boxplot(position = position_dodge(width = 0.75), width = 0.7) +
    #geom_errorbar(stat = "summary", fun.data = "mean_se", position = position_dodge(width = 0.75), width = 0.25) +
    geom_point(aes(color = group), position = position_jitterdodge(jitter.width = 0.2), size = 3, alpha = 0.4,  stroke = 0.6)   + 
    labs(x = "Condition", y = y_var,
         fill = "Group", color = "Group") +
    theme_minimal(base_size = 14) +
    scale_fill_manual(values = c("civilian" = "orange", "military" = "green")) +
    scale_color_manual(values = c("civilian" = "orange", "military" = "green")) +
    theme(axis.text.x = element_text(size = 12, angle = 45, hjust = 1),
          axis.text.y = element_text(size = 12),
          axis.title = element_text(size = 14),
          legend.title = element_text(size = 14),
          legend.text = element_text(size = 12))
})

# Combine the plots using patchwork
combined_plot <- plots[[1]] + plots[[2]] + plots[[3]] + plots[[4]] + plot_layout(ncol = 2)
print(combined_plot)