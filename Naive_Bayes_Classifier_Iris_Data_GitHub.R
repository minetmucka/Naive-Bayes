#############################################################################################################################################################
## Objective:  Machine learning of iris species classification with naive Bayes inference                                                                   #
## Data source: iris data set (included in R)	                                                                                                              #
## Please install "e1071" package: install.packages("e1071")                                                                                                #
#############################################################################################################################################################
install.packages("e1071")   

## load the library
library(e1071)

## DATA EXPLORATION
## load the iris data in R
data(iris)
## explore the data set
str(iris)
dim(iris)

## BUILD MODEL
## randomly choose 50% of the data set as training data (Why 50% instead of more?)
set.seed(27)
iris.train.indices <- sample(1:nrow(iris), 0.5*nrow(iris), replace=F)
iris.train <- iris[iris.train.indices,]
dim(iris.train)
summary(iris.train$Species)
## Grab the remaining 50% for the testing data
iris.test <- iris[-iris.train.indices,]
dim(iris.test)
summary(iris.test$Species)
## You could also do this
#random.rows.test <- setdiff(1:nrow(iris),random.rows.train)
#iris.test <- iris[random.rows.test,]

## fitting the decision model on the training set
iris.nb.model <- naiveBayes(Species ~ ., data = iris.train)

## MODEL EVALUATION
## Predict test set species using our naive bayes model
iris.nb.predictions <- predict(iris.nb.model, iris.test, type = "class")

## calculate the confusion matrix
iris.nb.confusion <- table(iris.nb.predictions, iris.test$Species)
print(iris.nb.confusion)
## calculate the accuracy, precision, recall, and F1
iris.nb.accuracy <- sum(diag(iris.nb.confusion)) / sum(iris.nb.confusion)
print(iris.nb.accuracy)

iris.nb.precision <- iris.nb.confusion[2,2] / sum(iris.nb.confusion[2,])
print(iris.nb.precision)

iris.nb.recall <- iris.nb.confusion[2,2] / sum(iris.nb.confusion[,2])
print(iris.nb.recall)

iris.nb.F1 <- 2 * iris.nb.precision * iris.nb.recall / (iris.nb.precision + iris.nb.recall)
print(iris.nb.F1)

