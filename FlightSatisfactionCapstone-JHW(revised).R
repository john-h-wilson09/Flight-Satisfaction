library(tidyverse)
library(dslabs)
library(caret)
library(purrr)
library(knitr)
library(tinytex)
library(rpart)
library(data.table)
options(digits = 4)    


# Flight Satisfaction dataset:
# https://www.kaggle.com/teejmahal20/airline-passenger-satisfaction/data

dl <- tempfile()
download.file("https://github.com/john-h-wilson09/Flight-Satisfaction/archive/master.zip", dl)

validation <- read.csv(unzip(dl,"Flight-Satisfaction-master/test.csv"))
practice <- read.csv(unzip(dl,"Flight-Satisfaction-master/train.csv"))
rm(dl)
str(practice) #see structure of the dataset

# Changing arrival status to factor of levels late or ontime
validation$Arrival.Delay.in.Minutes <- as.factor(ifelse(validation$Arrival.Delay.in.Minutes>0, "Late","OnTime"))
practice$Arrival.Delay.in.Minutes <- as.factor(ifelse(practice$Arrival.Delay.in.Minutes>0, "Late","OnTime")) 

# Remove all NA values from data tables
practice2 <- na.omit(setDT(practice), cols = c(1:25))
validation2 <- na.omit(setDT(validation), cols = c(1:25))

# Form test and train set within the practice dataset
set.seed(1, sample.kind = "Rounding")
test2_index <- createDataPartition(practice2$satisfaction,times=1,p=0.5,list=FALSE)
train2_set <- practice2[-test2_index,]
test2_set <- practice2[test2_index,]

# Regressions
lda2_fit <- train(satisfaction ~ ., method = "lda", data=train2_set)
lda2_pred <- predict(lda2_fit, test2_set)
lda2_acc <- mean(lda2_pred==test2_set$satisfaction)
varImp(lda2_fit) #shows online.boarding was most important variable

mods = c("glm","qda","rpart")
mod_acc <- map(mods, function(mod){
  fit2 <- train(satisfaction ~ ., method = mod, data=train2_set)
  pred <- predict(fit2,test2_set)
  mean(pred==test2_set$satisfaction)
})

loess_fit <- train(satisfaction ~ Flight.Distance+Age+Online.boarding+On.board.service+
                     Inflight.entertainment+Seat.comfort, method = "gamLoess", data=train2_set)
loess_pred <- predict(loess_fit,test2_set)
loess_acc <- mean(loess_pred==test2_set$satisfaction)

# Will take a few minutes to run
knn2_fit <- train(satisfaction ~ ., method = "knn", data=train2_set, 
                 tuneGrid = data.frame(k=113))
knn2_preds <- predict(knn2_fit,test2_set)
knn2_acc <- mean(knn2_preds==test2_set$satisfaction)

# See the distribution of satisfaction across online boarding rating
train2_set %>% ggplot(aes(Online.boarding, fill=satisfaction), xlab="Online Boarding Rating") + 
  geom_bar() + theme(legend.position = c(.05, .95), legend.justification = c("left", "top"))
OnlineB_preds <- ifelse(test2_set$Online.boarding >3,"satisfied","neutral or dissatisfied")
OnlineB_acc <- mean(OnlineB_preds==test2_set$satisfaction)

# Simplified Model - Objective Inputs
# Selecting the columns to keep for modeling
validation <- validation2 %>% dplyr::select(satisfaction, Age, Class, 
                                            Arrival.Delay.in.Minutes, Type.of.Travel, 
                                            Customer.Type, Flight.Distance) 
practice <- practice2 %>% dplyr::select(satisfaction, Age, Class, 
                                        Arrival.Delay.in.Minutes, Type.of.Travel, 
                                        Customer.Type, Flight.Distance)

# Form test and train set within the practice dataset
set.seed(1, sample.kind = "Rounding")
test_index <- createDataPartition(practice$satisfaction,times=1,p=0.5,list=FALSE)
train_set <- practice[-test_index,]
test_set <- practice[test_index,]

# Regressions
lda_fit <- train(satisfaction ~ ., method = "lda", data=train_set)
lda_preds <- predict(lda_fit, test_set)
lda_acc <- mean(lda_preds==test_set$satisfaction)
varImp(lda_fit) #shows class of ticket most important

# Regressions
models = c("glm","qda","rpart")
model_acc <- map(models, function(mod){
  fit <- train(satisfaction ~ ., method = mod, data=train_set)
  preds <- predict(fit,test_set)
  mean(preds==test_set$satisfaction)
})

loe_fit <- train(satisfaction ~ Flight.Distance+Age, method = "gamLoess", data=train_set)
loe_preds <- predict(loe_fit,test_set)
loe_acc <- mean(loe_preds==test_set$satisfaction)

# Will take a few minutes to run
knn_fit <- train(satisfaction ~ ., method = "knn", data=train_set, 
                 tuneGrid = data.frame(k=seq(73,133,20)))
knn_preds <- predict(knn_fit,test_set)
knn_acc <- mean(knn_preds==test_set$satisfaction)
plot(knn_fit)

# Displaying satisfacion based on ticket class
train2_set %>% ggplot(aes(Class,fill=satisfaction),xlab="Class") + geom_bar() +
  theme(legend.position = c(.95, .95), 
        legend.justification = c("right", "top")) #Business likely to be dissatisfied/neutral

# Applying theory that business class will be satisfied and the other neutral/dissatisfied
class_preds <- ifelse(test2_set$Class=="Business","satisfied","neutral or dissatisfied")
class_acc <- mean(class_preds==test2_set$satisfaction)

# Combining all results for simplified models
Results <- data.frame(
  Models = c("LDA","GLM","QDA","Rpart","Loess","Class Pred"),
  Accuracy = c(lda_acc,model_acc[[1]],model_acc[[2]],model_acc[[3]],
               loe_acc,class_acc)) %>% arrange(desc(Accuracy))

# Validation - LDA was most accurate model
val_preds <- predict(lda_fit,validation)
val_acc <- mean(val_preds==validation$satisfaction)

# Combined results for using all inputs
Results2 <- data.frame(
  Models = c("LDA","GLM","QDA","Rpart","Loess","OnlineBoarding Pred"),
  Accuracy = c(lda2_acc,mod_acc[[1]],mod_acc[[2]],mod_acc[[3]],
               loess_acc,OnlineB_acc)) %>% arrange(desc(Accuracy))

# Validation - GLM was most accurate model
val2_fit <- train(satisfaction ~ ., method="glm", data=train2_set)
val2_preds <- predict(val2_fit,validation2)
val2_acc <- mean(val2_preds==validation2$satisfaction)
