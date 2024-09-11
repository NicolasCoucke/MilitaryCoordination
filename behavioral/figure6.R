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

file_path <- "C:/Users/nicoucke/OneDrive - UGent/Desktop/Hyperscanning 1/trial_based_EEG_data.csv"
df <- read.csv(file_path)
df$trial <- as.numeric(df$trial)
df <- df %>%
  filter(!pair %in% c(26,36,39))
df_EEG <- df


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


# achieve only one value per pair so that it is possible to do an average trial correlation
df_EEG_pair <- df_EEG %>% group_by(group, pair, trial, condition) %>%  # Group by the specified columns
summarise_at(c('alpha_frontal', 'alpha_parietal_motor', 'alpha_parietal_parietal', 'alpha_motor_motor'), mean, na.rm = TRUE)



df_merged <- df_EEG_pair %>%
left_join(df_behavioral, by = c("group", "pair", "trial", "condition"))



df_avg <- df_merged %>%
  group_by(group, pair, condition) %>%  # Group by the specified columns
  summarise_at(c('interactive', 'sync', 'hierarchy', 'Speed', 'Speed_Variability', 'Speed_Variability_LF', 'num_tries', 'Asymmetry','Lag_Variability', 'alpha_frontal', 'alpha_parietal_motor', 'alpha_parietal_parietal', 'alpha_motor_motor'), mean, na.rm = TRUE)


df_avg <- df_merged





df1 <- df_avg %>% filter(condition %in% c("Synchronous/Egalitarian"))

# Initialize a dataframe to store effect sizes
effect_size_df <- data.frame()

# Loop over each y_column (dependent variable)
for (y_column in y_columns) {

# Loop over each x_column (predictor variable)
for (x_column in x_columns) {
    
    data <- df1
    # Select specific columns for the model
    data_xy <- data %>% select(all_of(c(x_column, y_column, random_effect)))
    
    # Convert data to numeric where necessary
    numeric_data <- data.frame(lapply(data_xy, function(x) as.numeric(as.character(x))))
    numeric_data$pair <- data[[random_effect]] # Include random effect column as a factor
    
    # Fit a mixed model: y_column ~ x_column + (1 | random_effect)
    formula <- as.formula(paste(y_column, "~", x_column, "+ (1 |", random_effect, ")"))
    mixed_model <- lmer(formula, data = numeric_data)
    
    # Extract the fixed effect estimate (effect size)
    effect_size <- summary(mixed_model)$coefficients[2, "t value"] # The second row contains the x_column effect
    
    # Store the result in a dataframe
    effect_size_df <- rbind(effect_size_df, data.frame(
    x_column = x_column,
    y_column = y_column,
    effect_size = effect_size
    ))
}
}
  

  create_effect_size_matrix <- function(effect_size_df, title = "Effect Size Matrix") {
  ggplot(effect_size_df, aes(x = x_column, y = y_column, fill = effect_size)) +
    geom_tile() +                                # Create heatmap tiles
    scale_fill_gradient2(low = "blue", mid = "white", high = "red", midpoint = 0, 
                         name = "Effect Size") +  # Color scale based on effect size
    labs(title = title, x = "Predictor (X Column)", y = "Response (Y Column)") +
    theme_minimal() +
    theme(axis.text.x = element_text(angle = 45, hjust = 1)) +
    coord_fixed()  # Maintain aspect ratio for better visualization
}

 
pdf("C:/Users/nicoucke/OneDrive - UGent/Desktop/Hyperscanning 1/paper plots/Figure6.pdf", width = 14, height = 10)  # Adjust the width and height as needed

 # Create a heatmap-like plot using ggplot2
# Function to create a heatmap-like plot using ggplot2

# Call the function with the data frame
create_effect_size_matrix(effect_size_df, title = "Effect Size Matrix Heatmap")
# Close the PDF device
dev.off()


print(effect_size_df)




# Function to fit multiple mixed-effects models and create a correlation matrix plot of effect sizes
create_effect_size_matrix <- function(data, x_columns, y_columns, random_effect, title = "Effect Size Matrix") {
  
  # Initialize a dataframe to store effect sizes
  effect_size_df <- data.frame()
  
  # Loop over each y_column (dependent variable)
  for (y_column in y_columns) {
    
    # Loop over each x_column (predictor variable)
    for (x_column in x_columns) {
      
      data <- df1
      # Select specific columns for the model
      data_xy <- data %>% select(all_of(c(x_column, y_column, random_effect)))
      
      # Convert data to numeric where necessary
      numeric_data <- data.frame(lapply(data_xy, function(x) as.numeric(as.character(x))))
      numeric_data$pair <- data[[random_effect]] # Include random effect column as a factor
      
      # Fit a mixed model: y_column ~ x_column + (1 | random_effect)
      formula <- as.formula(paste(y_column, "~", x_column, "+ (1 |", random_effect, ")"))
      mixed_model <- lmer(formula, data = numeric_data)
      
      # Extract the fixed effect estimate (effect size)
      effect_size <- summary(mixed_model)$coefficients[2, "t value"] # The second row contains the x_column effect
      
      # Store the result in a dataframe
      effect_size_df <- rbind(effect_size_df, data.frame(
        x_column_ = x_column,
        y_column_ = y_column,
        effect_size = effect_size
      ))
    }
  }
  
   ggplot(effect_size_df, aes(x = x_column_, y = y_column_, fill = effect_size)) +
    geom_tile() +                                # Create heatmap tiles
    scale_fill_gradient2(low = "blue", mid = "white", high = "red", midpoint = 0, 
                         name = "Effect Size") +  # Color scale based on effect size
    labs(title = title, x = "Predictor (X Column)", y = "Response (Y Column)") +
    theme_minimal() +
    theme(axis.text.x = element_text(angle = 45, hjust = 1)) +
    coord_fixed()  # Maintain aspect ratio for better visualization                       # Create heatmap tiles
   
}
# Example of how to call the function
# x_columns are the predictor variables, y_columns are the response variables, and 'pair' is the random effect
x_columns <- c("Speed", "Speed_Variability", "Asymmetry", "Lag_Variability")
y_columns <- c("num_tries", "Accuracy")
random_effect <- "pair"


# Define your dataframes here
df1 <- df_avg %>% filter(condition %in% c("Synchronous/Egalitarian"))
df2 <- df_avg %>% filter(condition %in% c("Synchronous/Egalitarian")) %>% filter(group %in% c("military"))
df3 <- df_avg %>% filter(condition %in% c("Synchronous/Egalitarian")) %>% filter(group %in% c("civilian"))
df4 <- df_avg %>% filter(condition %in% c("Complementary/Egalitarian"))
df5 <- df_avg %>% filter(condition %in% c("Complementary/Egalitarian")) %>% filter(group %in% c("military"))
df6 <- df_avg %>% filter(condition %in% c("Complementary/Egalitarian")) %>% filter(group %in% c("civilian"))

# List of dataframes
data_list <- list(df1, df2, df3, df4, df5, df6)

# Open a PDF device
pdf("C:/Users/nicoucke/OneDrive - UGent/Desktop/Hyperscanning 1/paper plots/Figure6.pdf", width = 14, height = 10)  # Adjust the width and height as needed

# Set up the plotting layout
par(mfrow = c(2, 3))  # 2 rows, 3 columns

# Example of how to call the function with different x and y axis columns
# x_columns could be predictors, and y_columns could be target variables
x_columns <- c("num_tries", "Speed", "Speed_Variability","Asymmetry", "Lag_Variability")
y_columns <- c('alpha_frontal', 'alpha_parietal_motor', 'alpha_parietal_parietal', 'alpha_motor_motor')
random_effect <- "pair"


# Create and plot each correlation plot
for (i in seq_along(data_list)) {
    # Assuming 'df' is your dataset
    create_mixed_model_plot(data_list[[i]], x_columns, y_columns, random_effect, title = paste("Plot", i))

}

# Close the PDF device
dev.off()




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
y_columns <- c('alpha_frontal', 'alpha_parietal_motor', 'alpha_parietal_parietal', 'alpha_motor_motor')
random_effect <- "pair"


# Open a PDF device
pdf("C:/Users/nicoucke/OneDrive - UGent/Desktop/Hyperscanning 1/paper plots/Figure6.pdf", width = 14, height = 10)  # Adjust the width and height as needed

# List to store ggplot objects
plot_list <- list()

# Loop over each dataframe in the list
for (i in seq_along(data_list)) {
  # Get the current dataframe
  data <- data_list[[i]]
  
  # Initialize a dataframe to store effect sizes and p-values
  effect_size_df <- data.frame()
  
  # Loop over each y_column (dependent variable)
  for (y_column in y_columns) {
    
    # Loop over each x_column (predictor variable)
    for (x_column in x_columns) {
      
      # Select specific columns for the model
      data_xy <- data %>% select(all_of(c(x_column, y_column, random_effect)))
      
      # Convert data to numeric where necessary
      numeric_data <- data.frame(lapply(data_xy, function(x) as.numeric(as.character(x))))
      numeric_data$pair <- data[[random_effect]] # Include random effect column as a factor
      
      # Fit a mixed model: y_column ~ x_column + (1 | random_effect)
      formula <- as.formula(paste(y_column, "~", x_column, "+ (1 |", random_effect, ")"))
      mixed_model <- lmer(formula, data = numeric_data)
      
      # Extract the fixed effect estimate (effect size) and p-value
      coef_summary <- summary(mixed_model)$coefficients
      effect_size <- coef_summary[2, "t value"]  # Fixed effect estimate
      p_value <- coef_summary[2, "Pr(>|t|)"]  # p-value
      
      # Determine asterisks based on p-value
      if (p_value < 0.001) {
        significance <- "***"
      } else if (p_value < 0.01) {
        significance <- "**"
      } else if (p_value < 0.05) {
        significance <- "*"
      } else {
        significance <- ""
      }
      
      # Store the result in a dataframe
      effect_size_df <- rbind(effect_size_df, data.frame(
        x_column_ = x_column,
        y_column_ = y_column,
        effect_size_ = effect_size,
        significance = significance
      ))
    }
  }
  
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









































