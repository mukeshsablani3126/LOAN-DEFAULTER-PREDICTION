rm(list=ls(all=T))
setwd("C:/edwisor")
getwd()

#Load libraries
library("respond")
library("ggplot2")
library("scales")
library("psych")
library("gplots")
library("corrgram")
library("DataCombine")
library("randomForest")
library("splitstackshape")
library("ccipes")
library("e1071")
library("C50")
library("caret")
library("dummies")
library("sampling")
library("DMwR")
library("rpart")
library("Rcurl")
library("unbalanced")
library("data.table")
library("recipes")
library("caret")
## Read the data
loan = read.csv("bank-loan.csv", header = T, na.strings = c(" ", "", "NA")) # stringsAsFactors
# lets preview the training data
head(loan)
###################  Exploratory data analysis ###########################
dim(loan)

# Structure of data
str(loan)

#Summary of datasets
summary(loan)


# list types for each attribute
sapply(loan, class)


#changing datatype of ed variable to factor datatype.
loan$ed=as.factor(loan$ed)
typeof(loan)

#Unique values in a column
unique(loan$ed)

###############################Missing value analysis###########################
#finding missing value
sum(is.na(loan))


library(tidyr)
# remove missing value
loan=na.omit(loan)
library(tidyr)
dim(loan)

########## OUTLIER ANALYSIS ##########################################

#boxplot

boxplot(loan$income,
        main = "Boxplot for income",
        ylab = "income",
        col = "orange",
        border = "brown",
        horizontal = FALSE,
        notch = FALSE
)
loan$default=as.factor(loan$default)

#selecting only numeric
numeric_index = sapply(loan,is.numeric)


#subset of numeric data
numeric_data = loan[,numeric_index]

#saving the column names of numeric data
cnames = colnames(numeric_data) 
cnames

#remove outliers

for(i in cnames){
  print(i)
  val = loan[,i][loan[,i] %in% boxplot.stats(loan[,i])$out]
  #print(length(val))
  loan = loan[which(!loan[,i] %in% val),]
}

dim(loan)

############## Feature Selection ###############

#selecting only numeric
numeric_index = sapply(loan,is.numeric)

#subset of numeric data
numeric_data = loan[,numeric_index]

#correlation plot


corrgram(loan[,numeric_index], order = F,
         upper.panel=panel.pie, text.panel=panel.txt, main="Correlation plot")
cor_mat = cor(numeric_data)
cor_mat = round(cor_mat, 2)
#here, we can see that no dependencies between  independent variable.so all variables need to be considered.

#########################Feature Scaling####################################################################################################################################################################

#normality check
hist(loan$income)

#data found not normally distributed
#take subset by removing ed variable
loan = subset(loan,select = -c(ed))

#to check range before normalization
loan_num = subset(loan, select=-default) 

#subset of numeric data
range(loan_num)

#saving the column names of numeric data
cnames = colnames(loan_num) 

#Normalization
for(i in cnames){
  print(i)
  loan[,i] = (loan[,i] - min(loan[,i]))/
    (max(loan[,i] - min(loan[,i])))
}

#to check range after Normalization

loan_num = subset(loan, select=-default)  

#subset of numeric data
range(loan_num)
#saving output in csv format
write.csv(loan,"finalloan data- R.csv",row.names =F )

###################################Model Development#######################################
#Clean the environment
rmExcept("loan")
df=loan

#Divide data into train and test using stratified sampling method
set.seed(1234)
train_index = sample(1:nrow(loan), 0.8 * nrow(loan))
train_loan = loan[train_index,]
test = loan[-train_index,]

#Logistic Regression
logit_model = glm(default ~ ., data = loan, family = "binomial")

#summary of the model
summary(logit_model)

#predict using logistic regression
logit_Predictions = predict(logit_model, newdata = test, type = "response")

#convert prob
logit_Predictions = ifelse(logit_Predictions > 0.5, 1, 0)

#confusion matrix
ConfMatrix_lg = table(test$default, logit_Predictions)
confusionMatrix(ConfMatrix_lg)

ConfMatrix_lg

#fpr=9.18%
#tpr=54.76%
#tnr=30.95%

#Accuracy= 80%


#False Negative rate
#FNR = FN/FN+TP 
#fnr=45.23%

#ROC Curve
library(ROCR)
data(ROCR.simple)
pred <- prediction( ROCR.simple$predictions, ROCR.simple$labels)

roc <- performance(pred,"tpr","fpr")
plot(roc,
     colorize=T,
     main="ROC -Curve")   
abline(a=0,b=1)

#AUC curve
auc<- performance(pred,"auc")
auc<-unlist(slot(auc,"y.values"))
auc<- round(auc,4)
legend(.6,.2,auc,title="AUC",cex=4)


##Decision tree for classification
library(dplyr)
str(loan)
#Develop Model on training data
C50_model = C5.0(default ~., loan, trials = 100, rules = TRUE)

#summary of DT model
summary(C50_model)

#write rules into disk
write(capture.output(summary(C50_model)), "c50Rules.txt")
#Lets predict for test cases
C50_Predictions = predict(C50_model, test[,-8], type = "class")


##Evaluate the performance of classification model
ConfMatrix_C50 = table(test$default, C50_Predictions)

ConfMatrix_C50

#fpr= 5.1%
#tpr=73.8%
#tnr=94.98%


#Accuracy= 88.57%
#False Negative rate
#FNR = FN/FN+TP
#fnr=26.19%


#ROC Curve
library(ROCR)
data(ROCR.simple)
pred <- prediction( ROCR.simple$predictions, ROCR.simple$labels)
roc <- performance(pred,"tpr","fpr")
plot(roc,
     colorize=T,
     main="ROC -Curve")   
abline(a=0,b=1)

#AUC curve
auc<- performance(pred,"auc")
auc<-unlist(slot(auc,"y.values"))
auc<- round(auc,4)
legend(.6,.2,auc,title="AUC",cex=4)

library(randomForest)
library(caret)



###Random Forest
RF_model = randomForest(default ~ ., loan, importance = TRUE, ntree = 500)

#Predict test data using random forest model
RF_Predictions = predict(RF_model, test[,-8])

##Evaluate the performance of classification model
ConfMatrix_RF = table(test$default, RF_Predictions)
confusionMatrix(ConfMatrix_RF)

ConfMatrix_RF

#fpr= 0.0%
#tpr=100%
#tnr=100%

#Accuracy= 100%
#False Negative rate
#FNR = FN/FN+TP 
#fnr =0.0%

#ROC Curve

library(ROCR)
data(ROCR.simple)
pred <- prediction( ROCR.simple$predictions, ROCR.simple$labels)
roc <- performance(pred,"tpr","fpr")
plot(roc,
     colorize=T,
     main="ROC -Curve")   
abline(a=0,b=1)

#AUC curve
auc<- performance(pred,"auc")
auc<-unlist(slot(auc,"y.values"))
auc<- round(auc,4)
legend(.6,.2,auc,title="AUC",cex=4)


#naive Bayes
library(e1071)
library(caret)

#Develop model
NB_model = naiveBayes(default ~ ., data = loan)

#predict on test cases #raw
NB_Predictions = predict(NB_model, test[,1:8], type = 'class')

#Look at confusion matrix
Conf_matrix = table(observed = test[,8], predicted = NB_Predictions)
confusionMatrix(Conf_matrix)

Conf_matrix



#fpr(recall)= 69.04%
#tpr =90.8%
#tnr=30.9%

#Accuracy= 72.8%
#False Negative rate
#FNR = FN/FN+TP 
#fnr =91.8%

library(ROCR)
data(ROCR.simple)
pred <- prediction( ROCR.simple$predictions, ROCR.simple$labels)
perf <- performance(pred,"tpr","fpr")
plot(perf)

#AUC curve
auc<- performance(pred,"auc")
auc<-unlist(slot(auc,"y.values"))
auc<- round(auc,4)
legend(.6,.2,auc,title="AUC",cex=4)

#saving output in csv format
write.csv(RF_Predictions,"finalloan result- R.csv",row.names =F )

# Here we can see easily all the model accuracy and compare them Random forest model give high accuracy so 
#  we will freeze random forest model 
