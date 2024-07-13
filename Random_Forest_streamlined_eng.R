# Install if necessary:

# Load libraries:

library(ggplot2)
library(cowplot)
library(randomForest)
library(caret)
library(ROCR)
library(psych)
library(readr)

# The CSV data should contain column headers to identify the data.
# Recommendation: have 20 or more rows; the more training data,
# the more accurate it will be in the long run!

# Training file path: (Can have any name, just make sure to provide the full path)
trainingPath = 
  "trainSet.csv"
# Prediction file path: (Data WITHOUT the variable to predict) can be added later
predictionPath = 
  "PredictSet.csv"

# Variable to predict: (change the value with the column name)
varPred <- "Variable"

# Number of trees: #Adjust this value based on the "Random Forest Error Percentage" table
varTree <- 500

# Read CSV data:
data <- read.csv(direccion, header=TRUE)
data <- data[sample(1:nrow(data)), ] #shuffle rows 
#   (important to avoid prediction errors)
Partition <- createDataPartition(data[[varPred]], p=0.70, list = FALSE)
data <- data[Partition,] #70% training data
test <- data[-Partition,] #30% test data

# Since we use the random forest method, we define a seed 
# to reproduce our results; these lines can be removed
randomValue=2
set.seed(randomValue)

# Adjust data since the RandomForest function only works with factors.
headers <- colnames(data)
for (header in headers) {
  if(header == varPred){
    data[[header]] <- as.factor(data[[header]])
    test[[header]] <- as.factor(test[[header]])
  }
  
}
# View data structure:
str(data)
str(test)

# Correlation table between numeric columns (Strictly numeric!):
corPlot(data[, -which(names(data) == varPred)], 
        stars = TRUE, 
        gr = colorRampPalette(heat.colors(40))) #the more color and stars, the higher the correlation

# Define the best number of nodes per tree
# If you get an error like the following, run these lines again.
# (Error in mtry < 1 || mtry > p: 'length = 2' in coercion to 'logical(1)')
mtry <- tuneRF(x = data[, -which(names(data) == varPred)], 
               y = data[[varPred]], 
               ntreeTry=varTree,
               stepFactor=1.5, 
               improve=0.05)
best.m <- mtry[mtry[, 2] == min(mtry[, 2]), 1]
print(mtry)
print(best.m)

# Prediction model
#mtry and ntree will depend on the data, defaults are 4 and 500
#mtry: number of variables used in each tree (auto-adjusts with best.m)
#ntree: total number of trees used, defined by the "varTree" variable
model <- randomForest(formula = as.formula(paste(varPred, "~ .")),
                      data = data,
                      proximity = FALSE,
                      importance = TRUE,
                      replace = TRUE,
                      ntree = varTree,
                      mtry = best.m)
print(model) #oob should be less than 10%

# Error percentage corresponding to the number of trees
plot(model, main="Random Forest Error Percentage")

#varImpPlot(model) #to see all variables
#You can change n.var to get more important variables, not exceeding
#the number of columns
varImpPlot(model,
           sort = TRUE,
           n.var = 10,
           main = "The 10 most important variables")
#left table = how bad the model performs without each variable
#right table = how important the values of each variable are

# Prediction table with ROC
# ROC is a graph that shows the performance of the classification model
# the larger the area under the curve, the better the prediction performance
pred2 = predict(modelo, type="prob")
perf = prediction(pred2[,2], data[[varPred]])
# auc = area under the curve
auc = performance(perf, "auc")
print(auc)
pred3 = performance(perf, "tpr", "fpr")
plot(pred3, main="ROC curve of the Random Forest", col=2, lwd=2)
abline(a=0, b=1, lwd=2, lty=2, col="gray")

# Prediction with the model on the test data:
# Important that the length of the data must be equal to the test data!
# Accuracy decreases as the amount of data to predict decreases,
pred <- predict(modelo, test)
# compare the predicted values (assigned to the test)
# with the reference values.
cM <- confusionMatrix(data = pred, reference = test[[varPred]])
# Get values from the confusion matrix
# TP: True Positives
# TN: True Negatives
# FP: False Positives
# FN: False Negatives
TP <- cM$table[2, 2]
TN <- cM$table[1, 1]
FP <- cM$table[1, 2]
FN <- cM$table[2, 1]

# Confusion matrix
fourfoldplot(cM$table, main = "Confusion Matrix on test data")
df_cM <- as.data.frame(cM$table)
ggplot(df_cM, aes(x = Reference, y = Prediction, fill = Freq)) +
  geom_tile() +
  scale_fill_gradient(low = "white", high = "green") +
  labs(x = "Actual Value", y = "Prediction", title = "Confusion Matrix on test data") +
  geom_text(aes(label = c(TN, FP, FN, TP)), vjust = 1.5, color = "black")

# accuracy = number of times a statement is correct
accuracy <- (TP + TN) / (TP + TN + FP + FN)
# precision = number of times a positive is positive
precision <- TP / (TP + FP)
# recall = number of truly positive data compared to predicted data
recall <- TP / (TP + FN)
# f1-score = balance between precision and recall
f1_score <- 2 * (precision * recall) / (precision + recall)

cat("Accuracy:", accuracy, "\n")
cat("Precision:", precision, "\n")
cat("Recall:", recall, "\n")
cat("F1-Score:", f1_score, "\n")


# ADDITIONAL NOTES
# If the model is very inaccurate, you can try using only the most important columns to create 
# a new random forest model, removing the less important columns.
# using this instruction in the training and test data reading:
# csv_data <- csv_data[, !(names(csv_data) == "Name of the unimportant column")]
# a few "n" times to remove the less important variables, and then
# recalculate the random forest and the prediction

# Use NewData to assign a prediction to unknown data:
newdata <- read.csv(direccionPrediccion, header=TRUE)
newdata <- newdata[, !(names(newdata) == varPred)] #TEST #remove results column
newdata[[varPred]] <- predict(modelo, newdata) #assign prediction values
print(newdata)
write_csv(newdata, "path/to/file.csv") #Save newdata predictions


# Save model for future predictions:
full_path <- "path/where/you/want/to/save/prediction_model.rds"
saveRDS(modelo, file = full_path)
load_model <- readRDS(full_path)
# This way you don't need to recalculate the model and can apply it to new data
# like the "newdata" implementation above.

