#=================================================================================#
# Statistical Learning and Data Mining
# Regression Challenge
# by MUHAMMAD ZAIN AMIN
#=================================================================================#


# Libraries
library(caret)
library(FNN)

# Ensures that we get reproducible results when using random functions
set.seed(42)

# Working directory
setwd("E:/Regression Challenge (SLDM)/")

# Read the train and test sets
train_data<- read.csv("train_ch.csv")
test_data<- read.csv("test_ch.csv")
head(train_data)
head(test_data)

# Removing the X columns from the train and test sets
train_data <- train_data[, -1]
train_data
test_data <- test_data[, -1]
test_data

#=================================================================================#
# -------------------------------- DATA ANALYSIS -------------------------------- #
#=================================================================================#

# Check for missing values in train and test sets
sum(is.na(train_data))
sum(is.na(test_data))

# Calculate z-scores for each numeric column in the dataset
z_scores <- apply(train_features, 2, function(x) (x - mean(x)) / sd(x))

# Define a threshold for outlier detection (e.g., z-score greater than 3)
outlier_threshold <- 3

# Identify rows with any column having a z-score greater than the threshold
outlier_rows <- which(rowSums(abs(z_scores) > outlier_threshold) > 0)

# Identify columns with outliers
outlier_columns <- colSums(abs(z_scores) > outlier_threshold) > 0

# Visualize outliers using box plots
boxplot(train_features[, outlier_columns], main = "Box Plot of Outliers",
        ylab = "Values", names = names(outlier_columns[outlier_columns]))

# Identify the rows with outliers
cat("Rows with outliers:", paste(outlier_rows, collapse = ", "), "\n")

# Identify the columns with outliers
cat("Columns with outliers:", paste(names(outlier_columns[outlier_columns]), collapse = ", "), "\n")

# Scatter plot for single predictors
single_predictor_plots <- lapply(names(train_features), function(predictor) {
  ggplot(train_data, aes_string(x = predictor, y = "Y")) +
    geom_point() +
    labs(title = paste("Scatter Plot of", predictor, "vs. Y"))
})

# Save single predictor plots to files
for (i in seq_along(single_predictor_plots)) {
  ggsave(paste0("single_predictor_plot_", names(train_features)[i], ".png"),
         plot = single_predictor_plots[[i]],
         width = 6, height = 4, dpi = 300)
}

# Scatter plots for pairs of predictors
pairwise_plots <- combn(names(train_features), 2, function(pair) {
  ggplot(train_data, aes_string(x = pair[1], y = pair[2])) +
    geom_point() +
    labs(title = paste("Scatter Plot of", pair[1], "vs.", pair[2]))
}, simplify = FALSE)


# Save pairwise plots to files
for (i in seq_along(pairwise_plots)) {
  predictor1 <- names(train_features)[combn(length(train_features), 2)[1, i]]
  predictor2 <- names(train_features)[combn(length(train_features), 2)[2, i]]
  
  ggsave(paste0("pairwise_plot_", predictor1, "_vs_", predictor2, ".png"),
         plot = pairwise_plots[[i]],
         width = 6, height = 4, dpi = 300)
}

# Calculate the correlation matrix
cor_matrix <- cor(train_features)

# Create a correlation plot
corrplot(cor_matrix, method = "color", type = "upper",
         tl.col = "black", tl.srt = 45)

# Print the highly correlated pairs
cat("Highly correlated predictor pairs:\n")
for (i in seq_along(highly_correlated)) {
  pair <- names(train_features)[highly_correlated[i]]
  cat(paste(pair, "\n"))
}

#=================================================================================#
# ------------------------------ Linear Regression ------------------------------ #
#=================================================================================#

#=================================================================================#
# 1. Train the linear regression model with all the predictors
#=================================================================================#

# Separating features from the label in the training data
train_labels <- train_data$Y
train_features <- subset(train_data, select = 1:9)
train_labels
train_features

# Check for Non-Linearity
# Plot the relationship between predictors and the response variable (Y)
plot(train_features$v1, train_labels)
plot(train_features$v2, train_labels)
plot(train_features$v3, train_labels)
plot(train_features$v4, train_labels)
plot(train_features$v5, train_labels)
plot(train_features$v6, train_labels)
plot(train_features$v7, train_labels)
plot(train_features$v8, train_labels)
plot(train_features$v9, train_labels)

# Apply Squared
train_features$v1_squared <-(train_features$v1)^2
train_features$v2_squared <-(train_features$v2)^2
train_features$v3_squared <-(train_features$v3)^2
train_features$v4_squared <-(train_features$v4)^2
train_features$v5_squared <-(train_features$v5)^2
train_features$v6_squared <-(train_features$v6)^2
train_features$v7_squared <-(train_features$v7)^2
train_features$v8_squared <-(train_features$v8)^2
train_features$v9_squared <-(train_features$v9)^2
train_features

# Train the linear regression model
lm_model_transformed <- lm(train_labels ~ ., data = train_features)

# Step 2: Assumption on Residuals

# Check for heteroscedasticity (varying spread of residuals)
plot(predict(lm_model_transformed), residuals(lm_model_transformed))

# Check for normality of residuals
hist(residuals(lm_model_transformed))
qqnorm(residuals(lm_model_transformed))

# Train the linear regression model 
lm_model <- lm(train_labels ~ ., data = train_features, method = "qr")

# Performance evaluation of the model
predicted_values <- predict(lm_model, train_features)

# Residual Sum of Squares (RSS):
rss <- sum((train_labels - predicted_values)^2)
cat("Residual Sum of Squares (RSS)", rss, "\n")

# Mean Squared Error (MSE):
mse <- mean((train_labels - predicted_values)^2)
cat("Mean Squared Error (MAE):", mse, "\n")

# Root Mean Square Error (RMSE)
rmse <- sqrt(mse)
cat("Root Mean Square Error (RMSE):", rmse, "\n")

# Mean Absolute Error (MAE)
mae <- mean(abs(train_labels - predicted_values))
cat("Mean Absolute Error (MAE):", mae, "\n")

# R-squared (coefficient of determination):
r_squared <- cor(train_labels, predicted_values)^2
cat("R-squared :", r_squared, "\n")

# Apply the square on all the predictors
test_data$v1_squared <- (test_data$v1)^2
test_data$v2_squared <- (test_data$v2)^2
test_data$v3_squared <- (test_data$v3)^2
test_data$v4_squared <- (test_data$v4)^2
test_data$v5_squared <- (test_data$v5)^2
test_data$v6_squared <- (test_data$v6)^2
test_data$v7_squared <- (test_data$v7)^2
test_data$v8_squared <- (test_data$v8)^2
test_data$v9_squared <- (test_data$v9)^2
test_data

# Get predictions on the test set
test_predictions <- predict(lm_model, newdata = test_data)

# Save the predictions to a CSV file
write.csv(test_predictions, file = "LR_all_predictors.csv", row.names = FALSE)

#=================================================================================#
# 2. Train the linear regression model with v1,v2,v3,v4,v6,v8,v9 predictors
#=================================================================================#

# Separating features from the label in the training data
train_labels <- train_data$Y
train_features <- subset(train_data, select = c("v1", "v2", "v3", "v4", "v6", "v8", "v9"))
train_labels
train_features

# Check for Non-Linearity
# Plot the relationship between predictors and the response variable (Y)
plot(train_features$v1, train_labels)
plot(train_features$v2, train_labels)
plot(train_features$v3, train_labels)
plot(train_features$v4, train_labels)
plot(train_features$v6, train_labels)
plot(train_features$v8, train_labels)
plot(train_features$v9, train_labels)

# Apply Squared
train_features$v1_squared <-(train_features$v1)^2
train_features$v2_squared <-(train_features$v2)^2
train_features$v3_squared <-(train_features$v3)^2
train_features$v4_squared <-(train_features$v4)^2
train_features$v6_squared <-(train_features$v6)^2
train_features$v8_squared <-(train_features$v8)^2
train_features$v9_squared <-(train_features$v9)^2
train_features

# Train the linear regression model
lm_model_transformed <- lm(train_labels ~ ., data = train_features)

# Step 2: Assumption on Residuals

# Check for heteroscedasticity (varying spread of residuals)
plot(predict(lm_model_transformed), residuals(lm_model_transformed))

# Check for normality of residuals
hist(residuals(lm_model_transformed))
qqnorm(residuals(lm_model_transformed))

# Train the model
lm_model <- lm(train_labels ~ ., data = train_features, method = "qr")

# Performance evaluation of the model
predicted_values <- predict(lm_model, train_features)

# Residual Sum of Squares (RSS):
rss <- sum((train_labels - predicted_values)^2)
cat("Residual Sum of Squares (RSS)", rss, "\n")

# Mean Squared Error (MSE):
mse <- mean((train_labels - predicted_values)^2)
cat("Mean Squared Error (MAE):", mse, "\n")

# Root Mean Square Error (RMSE)
rmse <- sqrt(mse)
cat("Root Mean Square Error (RMSE):", rmse, "\n")

# Mean Absolute Error (MAE)
mae <- mean(abs(train_labels - predicted_values))
cat("Mean Absolute Error (MAE):", mae, "\n")

# R-squared (coefficient of determination):
r_squared <- cor(train_labels, predicted_values)^2
cat("R-squared :", r_squared, "\n")

test_features <- subset(test_data, select = c("v1", "v2", "v3", "v4", "v6", "v8", "v9"))

# Apply squared
test_features$v1_squared <- (test_features$v1)^2
test_features$v2_squared <- (test_features$v2)^2
test_features$v3_squared <- (test_features$v3)^2
test_features$v4_squared <- (test_features$v4)^2
test_features$v6_squared <- (test_features$v6)^2
test_features$v8_squared <- (test_features$v8)^2
test_features$v9_squared <- (test_features$v9)^2
test_features

# Get predictions on the test set
test_predictions <- predict(lm_model, newdata = test_features)

# Save the predictions to a CSV file
write.csv(test_predictions, file = "LR_v1v2v3v4v6v8v9_predictors.csv", row.names = FALSE)

#=================================================================================#
# 3. Train the linear regression model with v1,v2,v3,v4,v6,v8 predictors
#=================================================================================#

# Separating features from the label in the training data
train_labels <- train_data$Y
train_features <- subset(train_data, select = c("v1", "v2", "v3", "v4", "v6", "v8"))
train_labels
train_features

# Check for Non-Linearity
# Plot the relationship between predictors and the response variable (Y)

plot(train_features$v1, train_labels)
plot(train_features$v2, train_labels)
plot(train_features$v3, train_labels)
plot(train_features$v4, train_labels)
plot(train_features$v6, train_labels)
plot(train_features$v8, train_labels)

# Apply squared
train_features$v1_squared <-(train_features$v1)^2
train_features$v2_squared <-(train_features$v2)^2
train_features$v3_squared <-(train_features$v3)^2
train_features$v4_squared <-(train_features$v4)^2
train_features$v6_squared <-(train_features$v6)^2
train_features$v8_squared <-(train_features$v8)^2
train_features

# Train the linear regression model
lm_model_transformed <- lm(train_labels ~ ., data = train_features)

# Step 2: Assumption on Residuals

# Check for heteroscedasticity (varying spread of residuals)
plot(predict(lm_model_transformed), residuals(lm_model_transformed))

# Check for normality of residuals
hist(residuals(lm_model_transformed))
qqnorm(residuals(lm_model_transformed))

# Train the model
lm_model <- lm(train_labels ~ ., data = train_features, method = "qr")

# Performance evaluation of the model
predicted_values <- predict(lm_model, train_features)

# Residual Sum of Squares (RSS):
rss <- sum((train_labels - predicted_values)^2)
cat("Residual Sum of Squares (RSS)", rss, "\n")

# Mean Squared Error (MSE):
mse <- mean((train_labels - predicted_values)^2)
cat("Mean Squared Error (MAE):", mse, "\n")

# Root Mean Square Error (RMSE)
rmse <- sqrt(mse)
cat("Root Mean Square Error (RMSE):", rmse, "\n")

# Mean Absolute Error (MAE)
mae <- mean(abs(train_labels - predicted_values))
cat("Mean Absolute Error (MAE):", mae, "\n")

# R-squared (coefficient of determination):
r_squared <- cor(train_labels, predicted_values)^2
cat("R-squared :", r_squared, "\n")

test_features <- subset(test_data, select = c("v1", "v2", "v3", "v4", "v6", "v8"))

# Apply squared
test_features$v1_squared <- (test_features$v1)^2
test_features$v2_squared <- (test_features$v2)^2
test_features$v3_squared <- (test_features$v3)^2
test_features$v4_squared <- (test_features$v4)^2
test_features$v6_squared <- (test_features$v6)^2
test_features$v8_squared <- (test_features$v8)^2
test_features

# Get predictions on the test set
test_predictions <- predict(lm_model, newdata = test_features)

# Save the predictions to a CSV file
write.csv(test_predictions, file = "LR_v1v2v3v4v6v8_predictors.csv", row.names = FALSE)

#=================================================================================#
# 4. Train the linear regression model with v1,v2,v3,v4 predictors
#=================================================================================#

# Separating features from the label in the training data
train_labels <- train_data$Y
train_features <- subset(train_data, select = c("v1", "v2", "v3", "v4"))
train_labels
train_features

# Check for Non-Linearity
# Plot the relationship between predictors and the response variable (Y)

plot(train_features$v1, train_labels)
plot(train_features$v2, train_labels)
plot(train_features$v3, train_labels)
plot(train_features$v4, train_labels)

# Apply squared
train_features$v1_squared <-(train_features$v1)^2
train_features$v2_squared <-(train_features$v2)^2
train_features$v3_squared <-(train_features$v3)^2
train_features$v4_squared <-(train_features$v4)^2
train_features

# Train the linear regression model
lm_model_transformed <- lm(train_labels ~ ., data = train_features)

# Step 2: Assumption on Residuals

# Check for heteroscedasticity (varying spread of residuals)
plot(predict(lm_model_transformed), residuals(lm_model_transformed))

# Check for normality of residuals
hist(residuals(lm_model_transformed))
qqnorm(residuals(lm_model_transformed))

# Train the model
lm_model <- lm(train_labels ~ ., data = train_features, method = "qr")

# Performance evaluation of the model
predicted_values <- predict(lm_model, train_features)

# Residual Sum of Squares (RSS):
rss <- sum((train_labels - predicted_values)^2)
cat("Residual Sum of Squares (RSS)", rss, "\n")

# Mean Squared Error (MSE):
mse <- mean((train_labels - predicted_values)^2)
cat("Mean Squared Error (MAE):", mse, "\n")

# Root Mean Square Error (RMSE)
rmse <- sqrt(mse)
cat("Root Mean Square Error (RMSE):", rmse, "\n")

# Mean Absolute Error (MAE)
mae <- mean(abs(train_labels - predicted_values))
cat("Mean Absolute Error (MAE):", mae, "\n")

# R-squared (coefficient of determination):
r_squared <- cor(train_labels, predicted_values)^2
cat("R-squared :", r_squared, "\n")

test_features <- subset(test_data, select = c("v1", "v2", "v3", "v4"))

# Apply squared
test_features$v1_squared <- (test_features$v1)^2
test_features$v2_squared <- (test_features$v2)^2
test_features$v3_squared <- (test_features$v3)^2
test_features$v4_squared <- (test_features$v4)^2
test_data

# Get predictions on the test set
test_predictions <- predict(lm_model, newdata = test_features)

# Save the predictions to a CSV file
write.csv(test_predictions, file = "LR_v1v2v3v4_predictors.csv", row.names = FALSE)

#=================================================================================#
# 5. Train the linear regression model with v1,v2,v3 predictors
#=================================================================================#

# Separating features from the label in the training data
train_labels <- train_data$Y
train_features <- subset(train_data, select = c("v1", "v2", "v3"))
train_labels
train_features

# Check for Non-Linearity
# Plot the relationship between predictors and the response variable (Y)

plot(train_features$v1, train_labels)
plot(train_features$v2, train_labels)
plot(train_features$v3, train_labels)

# Apply squared
train_features$v1_squared <-(train_features$v1)^2
train_features$v2_squared <-(train_features$v2)^2
train_features$v3_squared <-(train_features$v3)^2
train_features

# Train the linear regression model
lm_model_transformed <- lm(train_labels ~ ., data = train_features)

# Step 2: Assumption on Residuals

# Check for heteroscedasticity (varying spread of residuals)
plot(predict(lm_model_transformed), residuals(lm_model_transformed))

# Check for normality of residuals
hist(residuals(lm_model_transformed))
qqnorm(residuals(lm_model_transformed))

# Train the model 
lm_model <- lm(train_labels ~ ., data = train_features, method = "qr")

# Performance evaluation of the model
predicted_values <- predict(lm_model, train_features)

# Residual Sum of Squares (RSS):
rss <- sum((train_labels - predicted_values)^2)
cat("Residual Sum of Squares (RSS)", rss, "\n")

# Mean Squared Error (MSE):
mse <- mean((train_labels - predicted_values)^2)
cat("Mean Squared Error (MAE):", mse, "\n")

# Root Mean Square Error (RMSE)
rmse <- sqrt(mse)
cat("Root Mean Square Error (RMSE):", rmse, "\n")

# Mean Absolute Error (MAE)
mae <- mean(abs(train_labels - predicted_values))
cat("Mean Absolute Error (MAE):", mae, "\n")

# R-squared (coefficient of determination):
r_squared <- cor(train_labels, predicted_values)^2
cat("R-squared :", r_squared, "\n")

test_features <- subset(test_data, select = c("v1", "v2", "v3"))

# Apply squared
test_features$v1_squared <- (test_features$v1)^2
test_features$v2_squared <- (test_features$v2)^2
test_features$v3_squared <- (test_features$v3)^2
test_features

# Get predictions on the test set
test_predictions <- predict(lm_model, newdata = test_features)

# Save the predictions to a CSV file
write.csv(test_predictions, file = "LR_v1v2v3_predictors.csv", row.names = FALSE)

#=================================================================================#
# 6. Train the linear regression model with v1,v3 predictors
#=================================================================================#

# Separating features from the label in the training data
train_labels <- train_data$Y
train_features <- subset(train_data, select = c("v1", "v3"))
train_labels
train_features

# Check for Non-Linearity
# Plot the relationship between predictors and the response variable (Y)

plot(train_features$v1, train_labels)
plot(train_features$v3, train_labels)

# Apply squared
train_features$v1_squared <-(train_features$v1)^2
train_features$v3_squared <-(train_features$v3)^2
train_features

# Train the linear regression model
lm_model_transformed <- lm(train_labels ~ ., data = train_features)

# Step 2: Assumption on Residuals

# Check for heteroscedasticity (varying spread of residuals)
plot(predict(lm_model_transformed), residuals(lm_model_transformed))

# Check for normality of residuals
hist(residuals(lm_model_transformed))
qqnorm(residuals(lm_model_transformed))

# Train the model 
lm_model <- lm(train_labels ~ ., data = train_features, method = "qr")

# Performance evaluation of the model
predicted_values <- predict(lm_model, train_features)

# Residual Sum of Squares (RSS):
rss <- sum((train_labels - predicted_values)^2)
cat("Residual Sum of Squares (RSS)", rss, "\n")

# Mean Squared Error (MSE):
mse <- mean((train_labels - predicted_values)^2)
cat("Mean Squared Error (MAE):", mse, "\n")

# Root Mean Square Error (RMSE)
rmse <- sqrt(mse)
cat("Root Mean Square Error (RMSE):", rmse, "\n")

# Mean Absolute Error (MAE)
mae <- mean(abs(train_labels - predicted_values))
cat("Mean Absolute Error (MAE):", mae, "\n")

# R-squared (coefficient of determination):
r_squared <- cor(train_labels, predicted_values)^2
cat("R-squared :", r_squared, "\n")

test_features <- subset(test_data, select = c("v1", "v3"))

# Apply squared
test_features$v1_squared <- (test_features$v1)^2
test_features$v3_squared <- (test_features$v3)^2
test_features

# Get predictions on the test set
test_predictions <- predict(lm_model, newdata = test_features)

# Save the predictions to a CSV file
write.csv(test_predictions, file = "LR_v1v3_predictors.csv", row.names = FALSE)

#=================================================================================#
# 7. Train the linear regression model with v2,v3 predictors
#=================================================================================#

# Separating features from the label in the training data
train_labels <- train_data$Y
train_features <- subset(train_data, select = c("v2","v3"))
train_labels
train_features

# Check for Non-Linearity
# Plot the relationship between predictors and the response variable (Y)

plot(train_features$v2, train_labels)
plot(train_features$v3, train_labels)

# Apply squared
train_features$v2_squared <-(train_features$v2)^2
train_features$v3_squared <-(train_features$v3)^2
train_features

# Train the linear regression model
lm_model_transformed <- lm(train_labels ~ ., data = train_features)

# Step 2: Assumption on Residuals

# Check for heteroscedasticity (varying spread of residuals)
plot(predict(lm_model_transformed), residuals(lm_model_transformed))

# Check for normality of residuals
hist(residuals(lm_model_transformed))
qqnorm(residuals(lm_model_transformed))

# Train the model 
lm_model <- lm(train_labels ~ ., data = train_features, method = "qr")

# Performance evaluation of the model
predicted_values <- predict(lm_model, train_features)

# Residual Sum of Squares (RSS):
rss <- sum((train_labels - predicted_values)^2)
cat("Residual Sum of Squares (RSS)", rss, "\n")

# Mean Squared Error (MSE):
mse <- mean((train_labels - predicted_values)^2)
cat("Mean Squared Error (MAE):", mse, "\n")

# Root Mean Square Error (RMSE)
rmse <- sqrt(mse)
cat("Root Mean Square Error (RMSE):", rmse, "\n")

# Mean Absolute Error (MAE)
mae <- mean(abs(train_labels - predicted_values))
cat("Mean Absolute Error (MAE):", mae, "\n")

# R-squared (coefficient of determination):
r_squared <- cor(train_labels, predicted_values)^2
cat("R-squared :", r_squared, "\n")

test_features <- subset(test_data, select = c("v2","v3"))

# Apply squared
test_features$v2_squared <- (test_features$v2)^2
test_features$v3_squared <- (test_features$v3)^2
test_features

# Get predictions on the test set
test_predictions <- predict(lm_model, newdata = test_features)

# Save the predictions to a CSV file
write.csv(test_predictions, file = "LR_v2v3_predictors.csv", row.names = FALSE)

#=================================================================================#
# 8. Train the linear regression model with v3 predictors
#=================================================================================#

# Separating features from the label in the training data
train_labels <- train_data$Y
train_features <- subset(train_data, select = c("v3"))
train_labels
train_features

# Check for Non-Linearity
# Plot the relationship between predictors and the response variable (Y)

plot(train_features$v3, train_labels)

# Apply squared
train_features$v3_squared <-(train_features$v3)^2
train_features

# Train the linear regression model
lm_model_transformed <- lm(train_labels ~ ., data = train_features)

# Step 2: Assumption on Residuals

# Check for heteroscedasticity (varying spread of residuals)
plot(predict(lm_model_transformed), residuals(lm_model_transformed))

# Check for normality of residuals
hist(residuals(lm_model_transformed))
qqnorm(residuals(lm_model_transformed))

# Train the model 
lm_model <- lm(train_labels ~ ., data = train_features, method = "qr")

# Performance evaluation of the model
predicted_values <- predict(lm_model, train_features)

# Residual Sum of Squares (RSS):
rss <- sum((train_labels - predicted_values)^2)
cat("Residual Sum of Squares (RSS)", rss, "\n")

# Mean Squared Error (MSE):
mse <- mean((train_labels - predicted_values)^2)
cat("Mean Squared Error (MAE):", mse, "\n")

# Root Mean Square Error (RMSE)
rmse <- sqrt(mse)
cat("Root Mean Square Error (RMSE):", rmse, "\n")

# Mean Absolute Error (MAE)
mae <- mean(abs(train_labels - predicted_values))
cat("Mean Absolute Error (MAE):", mae, "\n")

# R-squared (coefficient of determination):
r_squared <- cor(train_labels, predicted_values)^2
cat("R-squared :", r_squared, "\n")

test_features <- subset(test_data, select = c("v3"))

# Apply squared
test_features$v3_squared <- (test_features$v3)^2
test_features

# Get predictions on the test set
test_predictions <- predict(lm_model, newdata = test_features)

# Save the predictions to a CSV file
write.csv(test_predictions, file = "LR_v3_predictors.csv", row.names = FALSE)

#=================================================================================#
# -------------------------------- KNN Regression ------------------------------- #
#=================================================================================#

#=================================================================================#
# 1. Train the KNN regression model with all the predictors
#=================================================================================#

# Separating features from the label in the training data
train_labels <- train_data$Y
train_features <- subset(train_data, select = 1:9)
train_labels
train_features

# Train the KNN regression model
k <- 9  # Choose the desired number of neighbors
knn_model <- knn.reg(train = train_features, test = NULL, y = train_labels, k = k)

# Predicted values
predicted_values <- knn_model$pred
predicted_values

# Residual Sum of Squares (RSS):
rss <- sum((train_labels - predicted_values)^2)
cat("Residual Sum of Squares (RSS)", rss, "\n")

# Mean Squared Error (MSE):
mse <- mean((train_labels - predicted_values)^2)
cat("Mean Squared Error (MAE):", mse, "\n")

# Root Mean Square Error (RMSE)
rmse <- sqrt(mse)
cat("Root Mean Square Error (RMSE):", rmse, "\n")

# Mean Absolute Error (MAE)
mae <- mean(abs(train_labels - predicted_values))
cat("Mean Absolute Error (MAE):", mae, "\n")

# R-squared (coefficient of determination):
r_squared <- cor(train_labels, predicted_values)^2
cat("R-squared :", r_squared, "\n")

# Convert the test features to matrices
test_features <- subset(test_data, select = 1:9)

# Get predictions on the test set
test_predictions <- knn.reg(train = train_features, test = test_features, y = train_labels, k = k)
test_predictions

# Get the predicted values from the KNN Reg object
test_predictions <- test_predictions$pred
test_predictions

# Create a data frame with the predicted values
predictions_df <- data.frame(Predicted_Values = test_predictions)

# Save the predictions to a CSV file
write.csv(predictions_df, file = "knn_all_predictors.csv", row.names = FALSE)

#=================================================================================#
# 2. Train the KNN regression model with v3 predictors
#=================================================================================#

# Separating features from the label in the training data
train_labels <- train_data$Y
train_features <- subset(train_data, select = c(3))
train_labels
train_features

# Train the KNN regression model
k <- 5 # Choose the desired number of neighbors
knn_model <- knn.reg(train = train_features, test = NULL, y = train_labels, k = k)

# Predicted values
predicted_values <- knn_model$pred
predicted_values

# Residual Sum of Squares (RSS):
rss <- sum((train_labels - predicted_values)^2)
cat("Residual Sum of Squares (RSS)", rss, "\n")

# Mean Squared Error (MSE):
mse <- mean((train_labels - predicted_values)^2)
cat("Mean Squared Error (MAE):", mse, "\n")

# Root Mean Square Error (RMSE)
rmse <- sqrt(mse)
cat("Root Mean Square Error (RMSE):", rmse, "\n")

# Mean Absolute Error (MAE)
mae <- mean(abs(train_labels - predicted_values))
cat("Mean Absolute Error (MAE):", mae, "\n")

# R-squared (coefficient of determination):
r_squared <- cor(train_labels, predicted_values)^2
cat("R-squared :", r_squared, "\n")

# Convert the test features to matrices
test_features <- subset(test_data, select = c(3))
test_features

# Get predictions on the test set
test_predictions <- knn.reg(train = train_features, test = test_features, y = train_labels, k = k)
test_predictions

# Get the predicted values from the KNN Reg object
test_predictions <- test_predictions$pred
test_predictions

# Create a data frame with the predicted values
predictions_df <- data.frame(Predicted_Values = test_predictions)

# Save the predictions to a CSV file
write.csv(predictions_df, file = "KNNR_v3.csv", row.names = FALSE)

#=================================================================================#
# 3. Train the KNN regression model with v1,v2,v3,v4,v6,v8,v9 predictors
#=================================================================================#

# Separating features from the label in the training data
train_labels <- train_data$Y
train_features <- subset(train_data, select = c(1,2,3,4,6,8,9))
train_labels
train_features

# Train the KNN regression model
k <- 5 # Choose the desired number of neighbors
knn_model <- knn.reg(train = train_features, test = NULL, y = train_labels, k = k)

# Predicted values
predicted_values <- knn_model$pred
predicted_values

# Residual Sum of Squares (RSS):
rss <- sum((train_labels - predicted_values)^2)
cat("Residual Sum of Squares (RSS)", rss, "\n")

# Mean Squared Error (MSE):
mse <- mean((train_labels - predicted_values)^2)
cat("Mean Squared Error (MAE):", mse, "\n")

# Root Mean Square Error (RMSE)
rmse <- sqrt(mse)
cat("Root Mean Square Error (RMSE):", rmse, "\n")

# Mean Absolute Error (MAE)
mae <- mean(abs(train_labels - predicted_values))
cat("Mean Absolute Error (MAE):", mae, "\n")

# R-squared (coefficient of determination):
r_squared <- cor(train_labels, predicted_values)^2
cat("R-squared :", r_squared, "\n")

# Convert the test features to matrices
test_features <- subset(test_data, select = c(1,2,3,4,6,8,9))
test_features

# Get predictions on the test set
test_predictions <- knn.reg(train = train_features, test = test_features, y = train_labels, k = k)
test_predictions

# Get the predicted values from the KNN Reg object
test_predictions <- test_predictions$pred
test_predictions

# Create a data frame with the predicted values
predictions_df <- data.frame(Predicted_Values = test_predictions)

# Save the predictions to a CSV file
write.csv(predictions_df, file = "KNNR_v1v2v3v4v6v8v9.csv", row.names = FALSE)

#=================================================================================#
# 4. Train the KNN regression model with v1,v2,v3,v4,v6,v8 predictors
#=================================================================================#

# Separating features from the label in the training data
train_labels <- train_data$Y
train_features <- subset(train_data, select = c(1,2,3,4,6,8))
train_labels
train_features

# Train the KNN regression model
k <- 5 # Choose the desired number of neighbors
knn_model <- knn.reg(train = train_features, test = NULL, y = train_labels, k = k)

# Predicted values
predicted_values <- knn_model$pred
predicted_values

# Residual Sum of Squares (RSS):
rss <- sum((train_labels - predicted_values)^2)
cat("Residual Sum of Squares (RSS)", rss, "\n")

# Mean Squared Error (MSE):
mse <- mean((train_labels - predicted_values)^2)
cat("Mean Squared Error (MAE):", mse, "\n")

# Root Mean Square Error (RMSE)
rmse <- sqrt(mse)
cat("Root Mean Square Error (RMSE):", rmse, "\n")

# Mean Absolute Error (MAE)
mae <- mean(abs(train_labels - predicted_values))
cat("Mean Absolute Error (MAE):", mae, "\n")

# R-squared (coefficient of determination):
r_squared <- cor(train_labels, predicted_values)^2
cat("R-squared :", r_squared, "\n")

# Convert the test features to matrices
test_features <- subset(test_data, select = c(1,2,3,4,6,8))
test_features

# Get predictions on the test set
test_predictions <- knn.reg(train = train_features, test = test_features, y = train_labels, k = k)
test_predictions

# Get the predicted values from the KNN Reg object
test_predictions <- test_predictions$pred
test_predictions

# Create a data frame with the predicted values
predictions_df <- data.frame(Predicted_Values = test_predictions)

# Save the predictions to a CSV file
write.csv(predictions_df, file = "KNNR_v1v2v3v4v6v8.csv", row.names = FALSE)

#=================================================================================#
# 5. Train the KNN regression model with v1,v2,v3,v4 predictors
#=================================================================================#

# Separating features from the label in the training data
train_labels <- train_data$Y
train_features <- subset(train_data, select = c(1,2,3,4))
train_labels
train_features

# Train the KNN regression model
k <- 5 # Choose the desired number of neighbors
knn_model <- knn.reg(train = train_features, test = NULL, y = train_labels, k = k)

# Predicted values
predicted_values <- knn_model$pred
predicted_values

# Residual Sum of Squares (RSS):
rss <- sum((train_labels - predicted_values)^2)
cat("Residual Sum of Squares (RSS)", rss, "\n")

# Mean Squared Error (MSE):
mse <- mean((train_labels - predicted_values)^2)
cat("Mean Squared Error (MAE):", mse, "\n")

# Root Mean Square Error (RMSE)
rmse <- sqrt(mse)
cat("Root Mean Square Error (RMSE):", rmse, "\n")

# Mean Absolute Error (MAE)
mae <- mean(abs(train_labels - predicted_values))
cat("Mean Absolute Error (MAE):", mae, "\n")

# R-squared (coefficient of determination):
r_squared <- cor(train_labels, predicted_values)^2
cat("R-squared :", r_squared, "\n")

# Convert the test features to matrices
test_features <- subset(test_data, select = c(1,2,3,4))
test_features

# Get predictions on the test set
test_predictions <- knn.reg(train = train_features, test = test_features, y = train_labels, k = k)
test_predictions

# Get the predicted values from the KNN Reg object
test_predictions <- test_predictions$pred
test_predictions

# Create a data frame with the predicted values
predictions_df <- data.frame(Predicted_Values = test_predictions)

# Save the predictions to a CSV file
write.csv(predictions_df, file = "KNNR_v1v2v3v4.csv", row.names = FALSE)

#=================================================================================#
# 6. Train the KNN regression model with v1,v2,v3 predictors
#=================================================================================#

# Separating features from the label in the training data
train_labels <- train_data$Y
train_features <- subset(train_data, select = c(1,2,3))
train_labels
train_features

# Train the KNN regression model
k <- 5 # Choose the desired number of neighbors
knn_model <- knn.reg(train = train_features, test = NULL, y = train_labels, k = k)

# Predicted values
predicted_values <- knn_model$pred
predicted_values

# Residual Sum of Squares (RSS):
rss <- sum((train_labels - predicted_values)^2)
cat("Residual Sum of Squares (RSS)", rss, "\n")

# Mean Squared Error (MSE):
mse <- mean((train_labels - predicted_values)^2)
cat("Mean Squared Error (MAE):", mse, "\n")

# Root Mean Square Error (RMSE)
rmse <- sqrt(mse)
cat("Root Mean Square Error (RMSE):", rmse, "\n")

# Mean Absolute Error (MAE)
mae <- mean(abs(train_labels - predicted_values))
cat("Mean Absolute Error (MAE):", mae, "\n")

# R-squared (coefficient of determination):
r_squared <- cor(train_labels, predicted_values)^2
cat("R-squared :", r_squared, "\n")

# Convert the test features to matrices
test_features <- subset(test_data, select = c(1,2,3))
test_features

# Get predictions on the test set
test_predictions <- knn.reg(train = train_features, test = test_features, y = train_labels, k = k)
test_predictions

# Get the predicted values from the KNN Reg object
test_predictions <- test_predictions$pred
test_predictions

# Create a data frame with the predicted values
predictions_df <- data.frame(Predicted_Values = test_predictions)

# Save the predictions to a CSV file
write.csv(predictions_df, file = "KNNR_v1v2v3.csv", row.names = FALSE)

#=================================================================================#
# 7. Train the KNN regression model with v2,v3 predictors
#=================================================================================#

# Separating features from the label in the training data
train_labels <- train_data$Y
train_features <- subset(train_data, select = c(2,3))
train_labels
train_features

# Train the KNN regression model
k <- 4 # Choose the desired number of neighbors
knn_model <- knn.reg(train = train_features, test = NULL, y = train_labels, k = k)

# Predicted values
predicted_values <- knn_model$pred
predicted_values

# Residual Sum of Squares (RSS):
rss <- sum((train_labels - predicted_values)^2)
cat("Residual Sum of Squares (RSS)", rss, "\n")

# Mean Squared Error (MSE):
mse <- mean((train_labels - predicted_values)^2)
cat("Mean Squared Error (MAE):", mse, "\n")

# Root Mean Square Error (RMSE)
rmse <- sqrt(mse)
cat("Root Mean Square Error (RMSE):", rmse, "\n")

# Mean Absolute Error (MAE)
mae <- mean(abs(train_labels - predicted_values))
cat("Mean Absolute Error (MAE):", mae, "\n")

# R-squared (coefficient of determination):
r_squared <- cor(train_labels, predicted_values)^2
cat("R-squared :", r_squared, "\n")

# Convert the test features to matrices
test_features <- subset(test_data, select = c(2,3))
test_features

# Get predictions on the test set
test_predictions <- knn.reg(train = train_features, test = test_features, y = train_labels, k = k)
test_predictions

# Get the predicted values from the KNN Reg object
test_predictions <- test_predictions$pred
test_predictions

# Create a data frame with the predicted values
predictions_df <- data.frame(Predicted_Values = test_predictions)

# Save the predictions to a CSV file
write.csv(predictions_df, file = "KNNR_v2v3.csv", row.names = FALSE)

#=================================================================================#
#                                      END
#=================================================================================#
