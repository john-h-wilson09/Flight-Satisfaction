library(tidyverse)
library(dslabs)
library(broom)
library(lubridate)
library(caret)
library(stringr)
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

# Selecting the columns to keep for modeling
validation <- validation %>% dplyr::select(satisfaction, Gender, Age, Class, Arrival.Delay.in.Minutes,
                                    Flight.Distance) 
practice <- practice %>% dplyr::select(satisfaction, Gender, Age, Class, Arrival.Delay.in.Minutes,
                                           Flight.Distance) 

# Changing arrival status to factor of levels late or ontime
validation$Arrival.Delay.in.Minutes <- as.factor(ifelse(validation$Arrival.Delay.in.Minutes>0, "Late","OnTime"))
validation <- validation %>% filter(!(is.na(validation$Arrival.Delay.in.Minutes))) #remove NA values
practice$Arrival.Delay.in.Minutes <- as.factor(ifelse(practice$Arrival.Delay.in.Minutes>0, "Late","OnTime")) 
practice <- practice %>% filter(!(is.na(practice$Arrival.Delay.in.Minutes))) #remove NA values

# Changing int class to num
num <- c(3,6)
practice[,num] <- apply(practice[,num],2,as.numeric) 
validation[,num] <- apply(validation[,num],2,as.numeric)

# Form test and train set within the practice dataset
set.seed(1, sample.kind = "Rounding")
test_index <- createDataPartition(practice$satisfaction,times=1,p=0.5,list=FALSE)
train_set <- practice[-test_index,]
test_set <- practice[test_index,]

# Regressions
lda_fit <- train(satisfaction ~ ., method = "lda", data=train_set)
lda_preds <- predict(lda_fit, test_set)
lda_acc <- mean(lda_preds==test_set$satisfaction)
varImp(lda_fit) #shows gender plays no influence in satisfaction so gender will be removed to increase calc speed
plot(train_set$Gender,train_set$satisfaction,xlab="Gender",ylab="Response")
train_set <- train_set[-2] #removed gender variable

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
knn_fit <- train(satisfaction ~ ., method = "knn", data=train_set, tuneGrid = data.frame(k=seq(70,130,20)))
knn_preds <- predict(knn_fit,test_set)
knn_acc <- mean(knn_preds==test_set$satisfaction)
plot(knn_fit)

# Satisfaction determined by class
train_set %>% ggplot(aes(Class,fill=satisfaction),lab="Class") + geom_bar() +
  theme(legend.position = c(.95, .95), 
        legend.justification = c("right", "top")) #Business likely to be dissatisfied/neutral
class_preds <- ifelse(test_set$Class=="Business","satisfied","neutral or dissatisfied")
class_acc <- mean(class_preds==test_set$satisfaction)

# Model Results
Results <- data.frame(Models = c("LDA","GLM","QDA","Rpart","Loess","Knn","Class Pred"),
                      Accuracy = c(lda_acc,model_acc[[1]],model_acc[[2]],model_acc[[3]],
                                   loe_acc,knn_acc,class_acc)) %>% arrange(desc(Accuracy))

# Validation - QDA was most accurate model
qda_fit <- train(satisfaction ~  ., method = "qda", data=train_set)
val_preds <- predict(qda_fit,validation)
val_acc <- mean(val_preds==validation$satisfaction)
