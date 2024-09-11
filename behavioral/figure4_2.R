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
library(stringr)

# now we will load in the questionnaire data:

file_path <- "C:/Users/nicoucke/OneDrive - UGent/Desktop/Hyperscanning 1/Hyperscanning Questionnaire encoding.xlsx"
df <- read_excel(file_path)
data_transposed <- as.data.frame(t(df))

# Set the first row as column names (if necessary)
colnames(data_transposed) <- data_transposed[1, ]
data_transposed <- data_transposed[-1, ]


# Get the column names
col_names <- colnames(data_transposed)

# Initialize an empty vector to store the new column names
new_col_names <- character(length(col_names))

# Initialize a variable to store the preceding string
preceding_string <- ""

# Loop through the column names to modify them
for (i in seq_along(col_names)) {
  # Check if the column name is numeric with a decimal (e.g., 1.0, 2.5, etc.)
  if (grepl("^[0-9]+(\\.[0-9]+)?$", col_names[i])) {
    # Concatenate the preceding string with the number (removing the ".0" if present)
    new_col_names[i] <- paste0(preceding_string, "_", sub("\\.0$", "", col_names[i]))
  } else {
    # Update the preceding string for the next set of columns
    preceding_string <- col_names[i]
    # Use the original string column name
    new_col_names[i] <- col_names[i]
  }
}


# Assign the new column names to the transposed data frame
colnames(data_transposed) <- new_col_names

# Named list for leadership traits and associated MLQ columns
leadership_traits <- list(
  "Idealized influence" = c(1, 8, 15),
  "Inspirational motivation" = c(2, 9, 16),
  "Intellectual Stimulation" = c(3, 10, 17),
  "Individualized consideration" = c(4, 11, 18),
  "Contingent reward" = c(5, 12, 19),
  "Management-by exception" = c(6, 13, 20),
  "Laissez-faire leadership" = c(7, 14, 21)
)


for (col in colnames(data_transposed)) {
  # Check if the column is not numeric and convert it
  if (!is.numeric(data_transposed[[col]])) {
    data_transposed[[col]] <- as.numeric(as.character(data_transposed[[col]]))
  }
}

# Loop through each leadership trait and compute the sum of specified MLQ columns
for (trait in names(leadership_traits)) {
  # Get the MLQ column numbers associated with the current trait
  mlq_columns <- leadership_traits[[trait]]
  
  # Construct the actual column names (e.g., "MLQ_1", "MLQ_8", "MLQ_15")
  column_names <- paste0("MLQ_", mlq_columns)
  
  # Ensure column_names is a character vector and check if these columns exist in the data frame
  if(all(column_names %in% colnames(data_transposed))) {
    # Compute the sum of the specified columns for each row
    data_transposed[[trait]] <- rowSums(data_transposed[, column_names])
  } else {
    # Print an error message if any of the required columns are missing
    missing_cols <- setdiff(column_names, colnames(data_transposed))
    print(paste("Missing columns for trait:", trait, "->", paste(missing_cols, collapse = ", ")))
  }
}



data_transposed$pair <- ((seq_len(nrow(data_transposed)) + 1) %/% 2)
data_transposed$pair_position <- ifelse(seq_len(nrow(data_transposed)) %% 2 == 1, 1, 2)

# Rename columns to remove spaces
data_transposed <- data_transposed %>%
  rename_with(~ gsub(" ", "_", .))

# Rename columns to remove spaces
data_transposed  <- # Rename columns to remove spaces
data_transposed  %>%
  filter(!pair %in% c(26, 32, 36, 39))


# Create 'Sync_Self good leader' column based on 'pair_position'
data_transposed$`Sync_Self good leader` <- ifelse(data_transposed$pair_position == 1, 
                                                  data_transposed$Sync_LF_1, 
                                                  data_transposed$Sync_FL_1)

# Create 'Desync_Self good leader' column based on 'pair_position'
data_transposed$`Desync_Self good leader` <- ifelse(data_transposed$pair_position == 1, 
                                                    data_transposed$Desync_LF_1, 
                                                    data_transposed$Desync_FL_1)


# Create 'Sync_In control' column based on 'pair_position'
data_transposed$`Sync_In control` <- ifelse(data_transposed$pair_position == 1, 
                                            data_transposed$Sync_LF_7, 
                                            data_transposed$Sync_FL_7)

# Create 'Desync_In control' column based on 'pair_position'
data_transposed$`Desync_In control` <- ifelse(data_transposed$pair_position == 1, 
                                              data_transposed$Desync_LF_7, 
                                              data_transposed$Desync_FL_7)

# Initialize the new column 'Sync_Good leader other' with NA to store the calculated values
data_transposed$`Sync_Good leader other` <- NA

# Assign 'Sync_Good leader other' value for participants with 'pair_position' == 1
data_transposed$`Sync_Good leader other`[data_transposed$pair_position == 1] <- data_transposed$Sync_LF_3[data_transposed$pair_position == 2]

# Assign 'Sync_Good leader other' value for participants with 'pair_position' == 2
data_transposed$`Sync_Good leader other`[data_transposed$pair_position == 2] <- data_transposed$Sync_FL_3[data_transposed$pair_position == 1]

# Initialize the new column 'Desync_Good leader other' with NA to store the calculated values
data_transposed$`Desync_Good leader other` <- NA

# Assign 'Desync_Good leader other' value for participants with 'pair_position' == 1
data_transposed$`Desync_Good leader other`[data_transposed$pair_position == 1] <- data_transposed$Desync_LF_3[data_transposed$pair_position == 2]

# Assign 'Desync_Good leader other' value for participants with 'pair_position' == 2
data_transposed$`Desync_Good leader other`[data_transposed$pair_position == 2] <- data_transposed$Desync_FL_3[data_transposed$pair_position == 1]

data_transposed$Familiarity_average <- rowMeans(data_transposed[, c("Familiarity_1", "Familiarity_2", "Familiarity_3", "Familiarity_4")])

data_questionnaire <- data_transposed %>%
  select((ncol(data_transposed) - 15):ncol(data_transposed))

# now just add the behavioral data et voila
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


df_behavioral_avg <- df_behavioral %>%
  group_by(group, pair, condition) %>%  # Group by the specified columns
  summarise_at(c('interactive', 'sync', 'hierarchy', 'who_leader', 'Speed', 'Speed_Variability', 'Speed_Variability_LF', 'num_tries', 'Asymmetry','Lag_Variability'), mean, na.rm = TRUE)

df_behavioral_avg <- df_behavioral_avg %>% rename(participant = who_leader)

df_behavioral_sync <- df_behavioral_avg %>% filter(condition %in% c("Synchronous/LeaderFollower", "Synchronous/FollowerLeader"))
filtered_behavioral_sync <- df_behavioral_sync  %>%
  filter((condition == "Synchronous/LeaderFollower" & participant == 1) |
         (condition == "Synchronous/FollowerLeader" & participant == 2))

df_behavioral_desync <- df_behavioral_avg %>% filter(condition %in% c("Complementary/LeaderFollower", "Complementary/FollowerLeader"))
filtered_behavioral_desync <- df_behavioral_sync  %>%
  filter((condition == "Complementary/LeaderFollower" & participant == 1) |
         (condition == "Complementary/FollowerLeader" & participant == 2))

# now attach the dataframes to the questionnaire data
data_questionnaire <- data_questionnaire %>% rename(participant = pair_position)

df_merged_sync <- data_questionnaire %>%
left_join(df_behavioral_sync, by = c("pair", "participant"))
df_sync <- df_merged_sync %>%
  # Remove columns that start with "Sync_"
  select(-starts_with("Desync_")) %>%
  # Rename columns that start with "Desync_" by removing the "Desync_" prefix
  rename_with(~ str_remove(., "^Sync_"), starts_with("Sync_"))

df_merged_desync <- data_questionnaire %>%
left_join(df_behavioral_desync, by = c("pair", "participant"))
df_desync <- df_merged_desync %>%
  select(-starts_with("Sync_")) %>%
  rename_with(~ str_remove(., "^Desync_"), starts_with("Desync_"))









# Define your dataframes here
df1 <- df_sync 
df2 <- df_sync %>% filter(group %in% c("military"))
df3 <- df_sync %>% filter(group %in% c("civilian"))
df4 <- df_desync 
df5 <- df_desync %>% filter(group %in% c("military"))
df6 <- df_desync %>% filter(group %in% c("civilian"))

# List of dataframes
data_list <- list(df1, df2, df3, df4, df5, df6)




x_columns <- c('Speed', 'Speed_Variability', 'num_tries', 'Asymmetry','Lag_Variability')  # Example y-axis variables
y_columns <- c("Idealized_influence", "Inspirational_motivation","Intellectual_Stimulation","Individualized_consideration",
            "Contingent_reward", "Management-by_exception", "Laissez-faire_leadership", 
            "Self good leader", "In control", "Good leader other", "Familiarity_average")  # Example x-axis variables

# Open a PDF device
pdf("C:/Users/nicoucke/OneDrive - UGent/Desktop/Hyperscanning 1/paper plots/Figure4_2.pdf", width = 14, height = 10)  # Adjust the width and height as needed

# List to store ggplot objects
plot_list <- list()

# Loop over each dataframe in the list
for (i in seq_along(data_list)) {
  # Get the current dataframe
  data <- data_list[[i]]
  
  # Initialize a dataframe to store correlations and p-values
  correlation_df <- data.frame()
  
  # Loop over each y_column (dependent variable)
  for (y_column in y_columns) {
    
    # Loop over each x_column (predictor variable)
    for (x_column in x_columns) {
      
      # Select specific columns for correlation
      data_xy <- data %>% select(all_of(c(x_column, y_column)))
      

         # Convert data to numeric where necessary
      numeric_data <- data_xy %>%
        mutate(across(everything(), as.numeric))  # Ensure all columns are numeric
      
      # Check for any NA values and remove them
      numeric_data <- na.omit(numeric_data)
      
      
      # Perform Pearson correlation
      cor_test <- cor.test(numeric_data[[x_column]], numeric_data[[y_column]], method = "pearson")
      correlation <- cor_test$estimate
      p_value <- cor_test$p.value
      
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
      correlation_df <- rbind(correlation_df, data.frame(
        x_column_ = x_column,
        y_column_ = y_column,
        correlation_ = correlation,
        significance = significance
      ))
    }
  }

   # Convert columns to factors with specified levels
  correlation_df <- correlation_df %>%
    mutate(x_column_ = factor(x_column_, levels = x_columns),
           y_column_ = factor(y_column_, levels = y_columns))
  
  # Create a heatmap-like plot using ggplot2
  p <- ggplot(correlation_df, aes(x = x_column_, y = y_column_, fill = correlation_)) +
    geom_tile() +                                # Create heatmap tiles
    scale_fill_gradient2(low = "blue", mid = "white", high = "red", midpoint = 0, 
                         name = "Correlation") +  # Color scale based on correlation values
    labs(title = paste("Plot", i), x = "Predictor (X Column)", y = "Response (Y Column)") +
    theme_minimal() +
    theme(axis.text.x = element_text(angle = 45, hjust = 1)) +
    coord_fixed() +  # Maintain aspect ratio for better visualization
   # geom_text(aes(label = round(correlation_, 2)), color = "black", size = 4) +  # Show correlation values
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