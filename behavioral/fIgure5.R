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
library(RColorBrewer)

file_path <- "C:/Users/nicoucke/OneDrive - UGent/Desktop/Hyperscanning 1/trial_based_EEG_data.csv"
df <- read.csv(file_path)
df$trial <- as.numeric(df$trial)
df <- df %>%
  filter(!pair %in% c(26,36,39))
df_EEG <- df
# achieve only one value per pair so that it is possible to do an average trial correlation
df_EEG_participant <- df_EEG %>% group_by(group, pair, participant, condition) %>%  # Group by the specified columns
summarise_at(c('alpha_frontal', 'alpha_parietal_motor', 'alpha_parietal_parietal', 'alpha_motor_motor'), mean, na.rm = TRUE)


# Create 'leader' and 'synchrony' variables and filter rows based on 'LeaderFollower' or 'FollowerLeader'
df_EEG_participant <- df_EEG_participant %>%
  mutate(
    leader = case_when(
      grepl("LeaderFollower", condition) ~ "Leader",     # 'Leader' if 'LeaderFollower' is in the condition
      grepl("FollowerLeader", condition) ~ "Follower"    # 'Follower' if 'FollowerLeader' is in the condition
    ),
    synchrony = case_when(
      grepl("Synchronous", condition) ~ "Synchrony",     # 'Synchrony' if 'Synchronous' is in the condition
      grepl("Complementary", condition) ~ "Complementary" # 'Complementary' if 'Complementary' is in the condition
    )
  ) %>%
  filter(!is.na(leader) & !is.na(synchrony))  

library(ggplot2)
library(gridExtra)

# Define a function to create the plot for a given dependent variable
create_plot <- function(dep_var) {
  ggplot(df_EEG_participant, aes_string(x = "leader", y = dep_var, fill = "group")) +
    geom_boxplot(aes(group = interaction(leader, group)), 
                 position = position_dodge(width = 0.75), width = 0.7) +
    geom_point(aes_string(color = "group", group = "interaction(leader, group)"), 
               position = position_jitterdodge(jitter.width = 0.2, dodge.width = 0.75), 
               size = 3, stroke = 0.6) +
    labs(x = NULL, y = dep_var, fill = "Group", color = "Group") +
    facet_wrap(~ synchrony, labeller = labeller(sync = c("Synchrony" = "Synchrony", "Complementary" = "Complementary"))) +  # Create subplots based on 'sync'
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
dep_vars <- c('alpha_frontal', 'alpha_parietal_motor', 'alpha_parietal_parietal', 'alpha_motor_motor')

# Open a PDF device
pdf("C:/Users/nicoucke/OneDrive - UGent/Desktop/Hyperscanning 1/paper plots/Figure5.pdf", width = 14, height = 10)  # Adjust the width and height as needed

# Generate the plots
plots <- lapply(dep_vars, create_plot)

# Arrange the plots in a 3x2 grid
grid.arrange(grobs = plots, ncol = 2, nrow = 3)

# Close the PDF device
dev.off()
