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
install.packages("corrplot")
library(corrplot)
install.packages("ggpubr")
library(ggpubr)
file_path <- "C:/Users/nicoucke/OneDrive - UGent/Desktop/Hyperscanning 1/behavioral data/data_dictionary.xlsx"

df <- read_excel(file_path)
df$trial <- as.numeric(df$trial)
df <- df %>%
  filter(!pair %in% c(26,36,39))
df_behavioral <- df



# match condition names for the behavioral and EEG data
df_behavioral <- df_behavioral %>%
  mutate(condition = case_when(
    condition == "Sync_Egalitarian" ~ "Synchronous/Egalitarian",
    condition == "Sync_LF" ~ "Synchronous/LeaderFollower",
    condition == "Sync_FL" ~ "Synchronous/FollowerLeader",
    condition == "Desync_Egalitarian" ~ "Complementary/Egalitarian",
    condition == "Desync_LF" ~ "Complementary/LeaderFollower",
    condition == "Desync_FL" ~ "Complementary/FollowerLeader",
    condition == "Sync_Solo" ~ "Individual/Egalitarian",
    TRUE ~ condition  # Keep other values unchanged
  ))


df_avg <- df_behavioral %>%
  group_by(group, pair, trial, condition) %>%  # Group by the specified columns
  summarise_at(c('interactive', 'sync', 'hierarchy', 'Speed', 'Speed_Variability', 'Speed_Variability_LF', 'num_tries', 'Asymmetry','Lag_Variability'), mean, na.rm = TRUE)







# Define your dataframes here
df1 <- df_avg %>% filter(condition %in% c("Synchronous/Egalitarian"))
df2 <- df_avg %>% filter(condition %in% c("Synchronous/Egalitarian")) %>% filter(group %in% c("military"))
df3 <- df_avg %>% filter(condition %in% c("Synchronous/Egalitarian")) %>% filter(group %in% c("civilian"))
df4 <- df_avg %>% filter(condition %in% c("Complementary/Egalitarian"))
df5 <- df_avg %>% filter(condition %in% c("Complementary/Egalitarian")) %>% filter(group %in% c("military"))
df6 <- df_avg %>% filter(condition %in% c("Complementary/Egalitarian")) %>% filter(group %in% c("civilian"))

# List of dataframes
data_list <- list(df1, df2, df3, df4, df5, df6)

# Define x_columns, y_columns, and random_effect
x_columns <- c("num_tries", "Speed", "Speed_Variability", "Asymmetry", "Lag_Variability")
y_columns <- c("num_tries", "Speed", "Speed_Variability", "Asymmetry", "Lag_Variability")
random_effect <- "pair"




# Open a PDF device
pdf("C:/Users/nicoucke/OneDrive - UGent/Desktop/Hyperscanning 1/paper plots/Figure4.pdf", width = 14, height = 10)  # Adjust the width and height as needed

# List to store ggplot objects
plot_list <- list()

# Loop over each dataframe in the list
for (i in seq_along(data_list)) {
  # Get the current dataframe
  data <- data_list[[i]]
  
  # Initialize a dataframe to store effect sizes and p-values
  effect_size_df <- data.frame()
  
  # Loop over each x_column and y_column to fill the lower triangular matrix
  for (j in seq_along(x_columns)) {
    x_column <- x_columns[j]
    
    for (k in 1:length(y_columns)) {
      y_column <- y_columns[k]
      
      # Ensure we only compute for unique pairs and skip comparisons of the same variable
      if (!is.na(x_column) && !is.na(y_column) && x_column != y_column) {
        
        # Select specific columns for the model
        data_xy <- data %>% select(all_of(c(x_column, y_column, random_effect)))
        
        # Convert data to numeric where necessary
        numeric_data <- data.frame(lapply(data_xy, function(x) as.numeric(as.character(x))))
        numeric_data$pair <- data[[random_effect]]  # Include random effect column as a factor
        
        # Fit a mixed model: y_column ~ x_column + (1 | random_effect)
        formula <- as.formula(paste(y_column, "~", x_column, "+ (1 |", random_effect, ")"))
        mixed_model <- lmer(formula, data = numeric_data)
        
        # Extract the fixed effect estimate (effect size) and p-value
        coef_summary <- summary(mixed_model)$coefficients
        effect_size <- coef_summary[2, "t value"]  # Fixed effect estimate
        p_value <- coef_summary[2, "Pr(>|t|)"]  # p-value
   
        # Determine asterisks based on p-value
        significance <- if (p_value < 0.001) {
          "***"
        } else if (p_value < 0.01) {
          "**"
        } else if (p_value < 0.05) {
          "*"
        } else {
          ""
        }

        if(k > j)
        {
          effect_size = 0
          significance = ""
        }
        
        # Store the result in a dataframe
        effect_size_df <- rbind(effect_size_df, data.frame(
          x_column_ = x_column,
          y_column_ = y_column,
          effect_size_ = effect_size,
          significance = significance
        ))
      }
      else {
          effect_size = 0
          significance = ""
          effect_size_df <- rbind(effect_size_df, data.frame(
          x_column_ = x_column,
          y_column_ = y_column,
          effect_size_ = effect_size,
          significance = significance))
      }
    }
  }

  effect_size_df  <- effect_size_df  %>%
  mutate(x_column_ = factor(x_column_, levels = x_columns),
          y_column_ = factor(y_column_, levels = y_columns))

  
  # Create a heatmap-like plot using ggplot2
  p <- ggplot(effect_size_df, aes(x = x_column_, y = y_column_, fill = effect_size_)) +
    geom_tile() +                                # Create heatmap tiles
    scale_fill_gradient2(low = "blue", mid = "white", high = "red", midpoint = 0, 
                         name = "Effect Size") +  # Color scale based on effect size
    labs(title = paste("Plot", i), x = "Predictor (X Column)", y = "Response (Y Column)") +
    theme_minimal() +
    theme(axis.text.x = element_text(angle = 45, hjust = 1)) +
    coord_fixed() +  # Maintain aspect ratio for better visualization
    geom_text(aes(label = significance), color = "black", size = 7, # Adjust size for larger stars
              fontface = "bold",  # Make text bold for better visibility
              stroke = 0.5,  # Add white edge around text
              color = "white")  # Edge color
  
  # Add the plot to the list
  plot_list[[i]] <- p
}

# Arrange all plots on the same page
ggarrange(plotlist = plot_list, ncol = 3, nrow = 2)  # Adjust ncol and nrow based on the layout

# Close the PDF device
dev.off()




