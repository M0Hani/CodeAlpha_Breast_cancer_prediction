# Importing libraries to use
  library(tidyverse)
  library(gplots)
  library(ggplot2)
  library(reshape2)
  library(dplyr)
  library(caret)
  library(e1071)
  library(rpart)
  library(randomForest)
  
#==================================================================
# Reading data
  df <- data.frame(read.csv("BreastCancer.csv"))
  
#==================================================================
# Data pre-processing
  str(df)
  summary(df)
  null_values <- colSums(is.na(df))
  print(null_values) #no null values
  df <- unique(df) #dropping duplicates
  rownames(df) <- NULL #re-indexing
  str(df)
  dfcat <- data.frame(lapply(df, as.factor)) #Categorizing data
  dfcat$diagnose <- recode(dfcat$diagnose, "1" = "Malignant", "0" = "Benign")
  
#==================================================================
# Visualizing data
  #Pie chart
  counts <- table(dfcat$diagnose)
  perc <- counts / sum(counts)
  pie_ <- data.frame( cls <- names(perc), Percentage <- perc)
  ggplot(pie_, aes(x = "", y = Percentage, fill = cls)) +
    geom_bar(stat = "identity", width = 1) + coord_polar(theta = "y") +
    geom_text(aes(label = paste0(round(Percentage * 100), "%")),
              position = position_stack(vjust = 0.5)) + labs(fill = "Class")
  
  #Bar plot
  bar_ <- data.frame(cls <- names(counts), count <- counts)
  ggplot(bar_, aes(x = cls, y = count)) +
    geom_bar(stat = "identity") +
    geom_text(aes(label = count), vjust = -0.5) +
    xlab("Class") + ylab("Count") +
    ggtitle("Bar plot of Diagnoses")
  
  #Heat-map
  cor_data <- reshape2::melt(cor(df))
  ggplot(cor_data, aes(x = Var1, y = Var2, fill = value)) +
    geom_tile() + geom_text(aes(label = round(value, 2), x = Var1, y = Var2), 
                            hjust = 0.5, vjust = 0.5, size = 2) +
    scale_fill_gradient2(low = "#D73027", mid = "#FEE090", high = "#4575B4", 
                         midpoint = 0, limits = c(-1, 1), name = "Correlation") +
    theme_minimal() + coord_equal() + labs(title = "Heatmap", x = "", y = "") + 
    theme(axis.text.x = element_text(angle = 90))
  
#==================================================================
# Prepare for modeling
  set.seed(123)
  train_indices <- sample(1:nrow(dfcat), nrow(dfcat)*0.7)
  train_data <- dfcat[train_indices, ]
  test_data <- dfcat[-train_indices, ]

# Model training and evaluation
  # 1. Support Vector Machines (SVM)
  svm_model <- svm(diagnose ~ ., data = train_data)
  svm_trn_pred <- predict(svm_model, newdata = train_data)
  svm_tst_pred <- predict(svm_model, newdata = test_data)
  svm_trn_acc <- sum(svm_trn_pred == train_data$diagnose) / length(svm_trn_pred)
  svm_tst_acc <- sum(svm_tst_pred == test_data$diagnose) / length(svm_tst_pred)
  
  # 2. Decision Tree (CART)
  dt_model <- rpart(diagnose ~ ., data = train_data)
  dt_trn_pred <- predict(dt_model, newdata = train_data, type = "class")
  dt_tst_pred <- predict(dt_model, newdata = test_data, type = "class")
  dt_trn_acc <- sum(dt_trn_pred == train_data$diagnose) / length(dt_trn_pred)
  dt_tst_acc <- sum(dt_tst_pred == test_data$diagnose) / length(dt_tst_pred)
  
  # 3. Random Forest
  rf_model <- randomForest(diagnose ~ ., data = train_data)
  rf_trn_pred <- predict(rf_model, newdata = train_data)
  rf_tst_pred<- predict(rf_model, newdata = test_data)
  rf_trn_acc <- sum(rf_trn_pred == train_data$diagnose) / length(rf_trn_pred)
  rf_tst_acc <- sum(rf_tst_pred == test_data$diagnose) / length(rf_tst_pred)
  
  # Print the accuracies
  cat("SVM - Training Accuracy:", svm_trn_acc, "\n")
  cat("SVM - Testing Accuracy:", svm_tst_acc, "\n")
  cat("Decision Tree - Training Accuracy:", dt_trn_acc, "\n")
  cat("Decision Tree - Testing Accuracy:", dt_tst_acc, "\n")
  cat("Random Forest - Training Accuracy:", rf_trn_acc, "\n")
  cat("Random Forest - Testing Accuracy:", rf_tst_acc, "\n")