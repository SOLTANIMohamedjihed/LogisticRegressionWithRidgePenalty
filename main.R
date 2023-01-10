

library(dplyr)
library(glmnet)
library(ROCR)
library(ggplot2)


#standardized training data

diabetes_train_std <- read.csv("diabetes_train-std.csv", header=TRUE, sep=",")
str(diabetes_train_std)


y_train_std <- as.factor(diabetes_train_std[,9])
x_train_std <- as.matrix(diabetes_train_std[,1:8])


#cross-validation
cv_train_std <- cv.glmnet(x_train_std, y_train_std, type.measure="class", nfolds=10, family="binomial")
lambda <- cv_train_std$lambda.min
lambda


plot(cv_train_std)


### fit the model

#ridge penalty alpha=0 &  the lasso penalty alpha=1

std_ridge_logit <- glmnet(x_train_std, y_train_std, family="binomial", alpha=0)


#predicting with the training set
SRL_pred_train <- predict(std_ridge_logit, x_train_std, type="class", s=lambda)

#report mean error rate (fraction of incorrect labels)
confusion_matrix_train <- table(y_train_std, SRL_pred_train)
confusion_matrix_train


error_rate_train <- (32+62)/400.0
error_rate_train

###Log-transformed data

diabetes_train_log <- read.csv("diabetes_train-log.csv", header=TRUE, sep=",")
#  dim = (400, 9)***

#separating response and predictor variables
y_train_log <- as.factor(diabetes_train_log[,9])
x_train_log <- as.matrix(diabetes_train_log[,1:8])

#cross-validation
cv_train_log <- cv.glmnet(x_train_log, y_train_log, type.measure="class", nfolds=10, family="binomial")
lambda_log <- cv_train_log$lambda.min
lambda_log

#fit the model
#ridge penalty alpha=0 ,  lasso penalty : alpha=1
log_ridge_logit <- glmnet(x_train_log, y_train_log, family="binomial", alpha=0)

#predicting with the log training set
LogRL_pred_train <- predict(log_ridge_logit, x_train_log, type="class", s=lambda_log)
log_confusion_matrix_train <- table(y_train_log, LogRL_pred_train)
log_confusion_matrix_train

#error rate
log_error_train <- (40+77)/400.0
log_error_train

diabetes_test_log <- read.csv("diabetes_test-log.csv", header=TRUE, sep=",")
#dim = (357, 9)

y_test_log <- as.factor(diabetes_test_log[,9])
x_test_log <- as.matrix(diabetes_test_log[,1:8])

#predicting with LOG test set
LogRL_pred_test <- predict(log_ridge_logit, x_test_log, type="class", s=lambda)
log_confusion_matrix_test <- table(y_test_log, LogRL_pred_test)
log_confusion_matrix_test

log_error_test <- (50+34)/357.0
log_error_test

###Plot the receiver operating characteristic (ROC) curves

#ROC curve for standardized data
prob_std <- predict(std_ridge_logit, x_test_std, type="response", s=lambda)
pred_std <- prediction(prob_std, y_test_std)
perf_std <- performance(pred_std, measure = "tpr", x.measure = "fpr")

#true positive rate
tpr.points1 <- attr(perf_std, "y.values")[[1]]
#tpr.points

#false positive rate
fpr.points1 <- attr(perf_std,"x.values")[[1]]
#fpr.points

#area under the curve
auc1 <- attr(performance(pred_std, "auc"), "y.values")[[1]]
formatted_auc1 <- signif(auc1, digits=3)


roc.data1 <- data.frame(fpr=fpr.points1, tpr=tpr.points1, model="GLM")


ggplot(roc.data1, aes(x=fpr, ymin=0, ymax=tpr)) +
    geom_ribbon(alpha=0.2) +
    geom_line(aes(y=tpr)) +
    ggtitle(paste0("ROC Curve for Standardized Data w/ AUC=", formatted_auc1))



#ROC curve for log data
#prob <- predict(fit, newdata=test, type="response")
prob <- predict(log_ridge_logit, x_test_log, type="response", s=lambda)
pred <- prediction(prob, y_test_log)
perf <- performance(pred, measure = "tpr", x.measure = "fpr")

#true positive rate
tpr.points <- attr(perf, "y.values")[[1]]
#tpr.points

#false positive rate
fpr.points <- attr(perf,"x.values")[[1]]
#fpr.points

#area under the curve
auc <- attr(performance(pred, "auc"), "y.values")[[1]]
formatted_auc <- signif(auc, digits=3)


roc.data <- data.frame(fpr=fpr.points, tpr=tpr.points, model="GLM")


ggplot(roc.data, aes(x=fpr, ymin=0, ymax=tpr)) +
    geom_ribbon(alpha=0.2) +
    geom_line(aes(y=tpr)) +
    ggtitle(paste0("ROC Curve for Log-Transformed Data w/ AUC=", formatted_auc))



## Generate and plot the lift curves for the two logistic regression models on the same plot.

#lift performance of standardized data
lift.perf1 <- performance(pred_std, "lift", "rpp")
plot(lift.perf1, colorize=T, main="Regularized Logit Performance", sub="Standardized vs. Logit Transformation")

#lift performance of log data
lift.perf <- performance(pred, "lift", "rpp")
plot(lift.perf, colorize=F, add=TRUE)



###RPubs


