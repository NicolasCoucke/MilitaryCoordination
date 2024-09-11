install.packages("readxl")
library("readxl")
library("tidyr")
library("ggplot2")
library("gridExtra")
library("lme4")
library("lmerTest")
#install.packages('MKinfer')
install.packages("sjPlot")
library("sjPlot")
library("stargazer")
library("texreg")
install.packages("effects")
library("effects")
#library(rstatix)
library("loo")
library("ggeffects")
library("dotwhisker")
library("merDeriv")
library(plyr); library(dplyr)
library("car")
install.packages("RColorBrewer")
install.packages('stargazer')
library(RColorBrewer)

file_path <- "C:/Users/nicoucke/OneDrive - UGent/Desktop/Hyperscanning 1/behavioral data/data_dictionary.xlsx"

df <- read_excel(file_path)
df$trial <- as.numeric(df$trial)

df <- df %>%
  filter(!pair %in% c(26,36,39))

df_avg <- df %>%
  group_by(group, pair, condition) %>%  # Group by the specified columns
  summarise_at(c('interactive', 'sync', 'hierarchy', 'Speed', 'Speed_Variability', 'Speed_Variability_LF', 'num_tries', 'Asymmetry','Lag_Variability'), mean, na.rm = TRUE)


# Filter for 'Sync_LF' and 'Sync_FL' conditions
filtered_df <- df_avg %>%
  filter(condition %in% c("Sync_LF", "Sync_FL"))

# Calculate the mean of all variables for each pair and condition
df_avg_sync <- filtered_df %>%
  group_by(group, pair) %>%
  summarise_at(c('interactive', 'sync', 'hierarchy', 'Speed', 'Speed_Variability', 'Speed_Variability_LF', 'num_tries', 'Asymmetry','Lag_Variability'), mean, na.rm = TRUE)

filtered_df <- df_avg %>%
  filter(condition %in% c("Desync_LF", "Desync_FL"))

df_avg_desync <- filtered_df %>%
  group_by(group, pair) %>%
  summarise_at(c('interactive', 'sync', 'hierarchy', 'Speed', 'Speed_Variability', 'Speed_Variability_LF', 'num_tries', 'Asymmetry','Lag_Variability'), mean, na.rm = TRUE)

filtered_df <- df_avg %>%
  filter(condition %in% c('Sync_Egalitarian', 'Desync_Egalitarian', 'Sync_solo'))


# now merge it together again:
df_combined <- bind_rows(df_avg_sync, df_avg_desync, filtered_df)






df_combined$sync <- as.factor(df_combined$sync)
df_combined$hierarchy <- as.factor(df_combined$hierarchy)
df_combined$group <- as.factor(df_combined$group)


df_combined <- df_combined %>%
  dplyr::mutate(hierarchy = dplyr::recode(hierarchy,
                                          '0' = 'Egalitarian',
                                          '1' = 'Hierarchical'))



library(ggplot2)
library(gridExtra)

# Define a function to create the plot for a given dependent variable
create_plot <- function(dep_var) {
  ggplot(df_combined, aes_string(x = "hierarchy", y = dep_var, fill = "group")) +
    geom_boxplot(aes(group = interaction(hierarchy, group)), 
                 position = position_dodge(width = 0.75), width = 0.7) +
    geom_point(aes_string(color = "group", group = "interaction(hierarchy, group)"), 
               position = position_jitterdodge(jitter.width = 0.2, dodge.width = 0.75), 
               size = 3, stroke = 0.6) +
    labs(x = NULL, y = dep_var, fill = "Group", color = "Group") +
    facet_wrap(~ sync, labeller = labeller(sync = c("1" = "Synchrony", "0" = "Complementary"))) +  # Create subplots based on 'sync'
    theme_minimal(base_size = 14) +
    scale_fill_manual(values = c("civilian" = "#eeb704f3", "military" = "#008000B0")) +
    scale_color_manual(values = c("civilian" = "#eeb704f3", "military" = "#008000B0")) +
    theme(axis.text.x = element_text(size = 12, angle = 45, hjust = 1),
          axis.text.y = element_text(size = 12),
          axis.title = element_text(size = 14),
          legend.title = element_text(size = 14),
          legend.text = element_text(size = 12),
          strip.text = element_text(size = 14),  # Size of facet labels
          axis.title.x = element_blank())  # Remove x-axis labels
}

# List of dependent variables you want to plot
dep_vars <- c("num_tries", "num_tries", "Speed", "Speed_Variability", "Asymmetry", "Lag_Variability")

# Open a PDF device
pdf("C:/Users/nicoucke/OneDrive - UGent/Desktop/Hyperscanning 1/paper plots/Figure3.pdf", width = 14, height = 10)  # Adjust the width and height as needed

# Generate the plots
plots <- lapply(dep_vars, create_plot)

# Arrange the plots in a 3x2 grid
grid.arrange(grobs = plots, ncol = 2, nrow = 3)

# Close the PDF device
dev.off()


