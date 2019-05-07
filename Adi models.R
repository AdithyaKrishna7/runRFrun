rm(list = ls())

library(dplyr)
library(tidyverse)
library(sqldf)
library(ggplot2)
library(corrplot)
library(psych)
library(hydroGOF)
library(mice)
library(randomForest)
library(DataExplorer)
library(mltools)
library(caret)
library(data.table)
library(xgboost)


base_data <- readRDS("~/runRFrun/AT2_train_STUDENT.rds")
pred <- readRDS("~/runRFrun/AT2_test_STUDENT.rds")

base_data <- base_data %>% distinct(.)

names(base_data)

str(base_data)

glimpse(base_data)

plot_missing(base_data)

#base_data$user_id <- as.numeric(as.character(base_data$user_id))
#base_data$item_id <- as.numeric(as.character(base_data$item_id))

base_data$item_imdb_mature_rating <- factor(base_data$item_imdb_mature_rating, ordered = F)
base_data$age_band <- factor(base_data$age_band, ordered = F)

base_data$video_release_date <- NULL
base_data$imdb_url <- NULL
base_data$release_date <- NULL
base_data$timestamp <- NULL
base_data$zip_code <- NULL

log_vars <- unlist(lapply(base_data, is.logical))  
base_log <- data.frame(base_data[ , log_vars])
base_non_log <- data.frame(base_data[ , !log_vars])

for(i in 1:ncol(base_log)){
  base_log[, i] <- as.numeric(as.logical(base_log[, i]))
}

base_data <- cbind(base_non_log, base_log)
rm(base_log, base_non_log)

str(base_data)
#------------------------------------------------------------------------------------------------
#Missing value imputation

base_data$item_id <- as.numeric(as.character(base_data$item_id))

num_cols <- unlist(lapply(base_data, is.numeric))

base_comp <- base_data[complete.cases(base_data),]

base_na <- base_data[!complete.cases(base_data),]

(nrow(base_comp) + nrow(base_na))/nrow(base_data) #1

plot_missing(base_comp)
plot_missing(base_na)

na_cols <- apply(is.na(base_na), 2, any)
View(na_cols)
na_cols[5] <- TRUE

base_f <- base_na[,!na_cols]
base_no <- base_na[,na_cols]

base_na <- cbind(base_f, base_no)

rm(base_f, base_no)

base_avg <- base_data[, na_cols]

avgs <- base_avg %>% summarise_all(list(avg = ~mean), na.rm = T)

names(base_na)
names(avgs)

base_na <- as.data.frame(base_na)

for(i in 31:ncol(base_na)){
  base_na[is.na(base_na[,i]), i] <- as.numeric(avgs[,i-30])
}

base_f <- base_comp[,!na_cols]
base_no <- base_comp[,na_cols]

base_comp <- cbind(base_f, base_no)

base_data <- rbind(base_comp, base_na)

rm(base_avg, base_comp, base_f, base_na, base_no)

plot_missing(base_data)

#------------------------------------------------------------------------------------------------
#Check the same for predicted

pred$item_imdb_mature_rating <- factor(pred$item_imdb_mature_rating, ordered = F)
pred$age_band <- factor(pred$age_band, ordered = F)

pred$video_release_date <- NULL
pred$imdb_url <- NULL
pred$release_date <- NULL
pred$timestamp <- NULL
pred$zip_code <- NULL

pred$item_id <- as.numeric(as.character(pred$item_id))

num_cols <- unlist(lapply(pred, is.numeric))

pred_comp <- pred[complete.cases(pred),]

pred_na <- pred[!complete.cases(pred),]

(nrow(pred_comp) + nrow(pred_na))/nrow(pred) #1

plot_missing(pred_comp)
plot_missing(pred_na)

na_cols <- apply(is.na(pred_na), 2, any)
View(na_cols)
na_cols[5] <- TRUE

pred_f <- pred_na[,!na_cols]
pred_no <- pred_na[,na_cols] #Two extra columns with NAs compared to base data

pred_na <- cbind(pred_f, pred_no)
names(pred_na)
rm(pred_f, pred_no)

pred_avg <- pred[, na_cols]

names(pred_na)
names(avgs)

pred_na <- pred_na[,c(1:27,29:30,28,31:43)]

names(pred_na[,30:43])
names(avgs)

pred_na <- as.data.frame(pred_na)

for(i in 30:ncol(pred_na)){
  pred_na[is.na(pred_na[,i]), i] <- as.numeric(avgs[,i-29])
}

pred_f <- pred_comp[,!na_cols]
pred_no <- pred_comp[,na_cols]

pred_comp <- cbind(pred_f, pred_no)

pred <- rbind(pred_comp, pred_na)

plot_missing(pred)

user_gender_item_mean_rating_avg <- mean(base_data$user_gender_item_imdb_mean_rating, na.rm = T)
user_age_band_item_mean_rating_avg <- mean(base_data$user_age_band_item_mean_rating, na.rm = T)

names(pred_na)

#User age band rating and gender rating replacement
pred_na[is.na(pred_na[,28]), 28] <- user_age_band_item_mean_rating_avg
pred_na[is.na(pred_na[,29]), 29] <- user_gender_item_mean_rating_avg

pred <- rbind(pred_comp, pred_na)

plot_missing(pred)

rm(pred_avg, pred_comp, pred_f, pred_na, pred_no)

#Export tables
write.table(base_data, "D:/UTS/MDSI/Sem 01/Data, Algos, and Meaning/Assignments/DAM Assignment 2/student_data/student_data/base.csv", sep = ",", row.names = F)

write.table(pred, "D:/UTS/MDSI/Sem 01/Data, Algos, and Meaning/Assignments/DAM Assignment 2/student_data/student_data/pred.csv", sep = ",", row.names = F)

#------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------
#Linear regression

base_data <- read.table("D:/UTS/MDSI/Sem 01/Data, Algos, and Meaning/Assignments/DAM Assignment 2/student_data/student_data/base.csv", sep = ",", header = T)

pred <- read.table("D:/UTS/MDSI/Sem 01/Data, Algos, and Meaning/Assignments/DAM Assignment 2/student_data/student_data/pred.csv", sep = ",", header = T)

set.seed(1058)

train_indices <- sample(seq_len(nrow(base_data)), size = floor(0.75*nrow(base_data)))

train <- base_data[train_indices,]
test <- base_data[-train_indices,]


#Attempt 1
lin <- lm(rating ~ gender + age + occupation, data = train)
summary(lin)

test$pred <- predict(lin, newdata = test, type = "response")

rmse(test$pred, test$rating) #1.108914

#Attempt 2
lin <- lm(rating ~ gender + age + occupation + item_mean_rating, data = train)
summary(lin)

test$pred <- predict(lin, newdata = test, type = "response")

rmse(test$pred, test$rating) #0.9856837

#Attempt 3
lin <- lm(rating ~ gender + age + occupation + item_mean_rating + user_gender_item_mean_rating, data = train)
summary(lin)

test$pred <- predict(lin, newdata = test, type = "response")

rmse(test$pred, test$rating) #0.9732259

#Attempt 4
lin <- lm(rating ~ gender + age + occupation + item_mean_rating + user_gender_item_mean_rating, data = train)
summary(lin)

test$pred <- predict(lin, newdata = test, type = "response")

rmse(test$pred, test$rating) #0.9732259

#Attempt 5
lin <- lm(rating ~ gender + age + occupation + item_mean_rating + user_gender_item_mean_rating + user_age_band_item_mean_rating, data = train)
summary(lin)

test$pred <- predict(lin, newdata = test, type = "response")

rmse(test$pred, test$rating) #0.9517137

#Attempt 6
lin <- lm(rating ~ gender + age + occupation + item_mean_rating + user_gender_item_mean_rating + user_age_band_item_mean_rating + item_imdb_rating_of_ten, data = train)
summary(lin)

test$pred <- predict(lin, newdata = test, type = "response")

rmse(test$pred, test$rating) #0.9583025

#Attempt 7		
lin <- lm(rating ~ gender + age + occupation + item_mean_rating + user_gender_item_mean_rating + user_age_band_item_mean_rating + item_imdb_rating_of_ten + item_imdb_mature_rating + item_imdb_length, data = train)		
summary(lin)		

test$pred <- predict(lin, newdata = test, type = "response")

rmse(test$pred, test$rating) #0.9585021

test$pred <- NULL

#-------------------------------------------------------------------------------------------------
#Exploratory factor analysis

#One hot encode all factor variables
#Convert user_id and ZIP code to non factors
train$user_id <- as.numeric(as.character(train$user_id))

train_new <- one_hot(as.data.table(train))

test$user_id <- as.numeric(as.character(test$user_id))

test_new <- one_hot(as.data.table(test))

train_fact <- train_new %>% select(-user_id, -rating, -movie_title, -item_id)

#Exploratory factor analysis

parallel <- fa.parallel(train_fact, fm = 'minres', fa = 'fa')

factors <- fa(train_fact, nfactors = 7, rotate = "oblimin")

print(factors$loadings,cutoff = 0.3)

fa.diagram(factors)

train_fact <- cbind(train_new %>% select(user_id, rating, movie_title, item_id), factors$scores)

test_fact <- test_new %>% select(-user_id, -rating, -movie_title, -item_id)

test_fact <- factor.scores(as.data.table(test_fact), factors)

test_new <- cbind(test_new %>% select(user_id, rating, movie_title, item_id), test_fact$scores)

#Linear regression using factors

lin <- lm(rating ~ MR1 + MR2 + MR4 + MR6 + MR7, data = train_fact)
summary(lin)

test_new$pred <- predict(lin, newdata = test_new, type = "response")
rmse(test_new$rating, test_new$pred) #1.015678

#Check

#Random forest attempt
train_fact$movie_title <- NULL
test_new$movie_title <- NULL
test_new$pred <- NULL

rand <- randomForest(rating ~ . - user_id - item_id, data = train_fact, importance=TRUE, xtest=test_new[,-c(1:3)],ntree=250, keep.forest = T)

rmse(rand$test$predicted, test$rating) #1.032


#-------------------------------------------------------------------------------------------------
#Predictions
#-------------------------------------------------------------------------------------------------

names(pred)

str(pred)

glimpse(pred)

log_vars <- unlist(lapply(pred, is.logical))  
pred_log <- data.frame(pred[ , log_vars])
pred_non_log <- data.frame(pred[ , !log_vars])

for(i in 1:ncol(pred_log)){
  pred_log[, i] <- as.numeric(as.logical(pred_log[, i]))
}

pred <- cbind(pred_non_log, pred_log)
rm(pred_log, pred_non_log)

#-------------------------------------------------------------------------------------------------
#Factor scoring for pred
pred$user_id <- as.numeric(as.character(pred$user_id))

pred$pred <- predict(lin, newdata = pred, type = "response")

pred_exp <- pred %>% select(user_id, item_id, pred)

write.table(pred_exp, "C:/Users/Adithya/Desktop/export.csv", row.names = F, sep = ",")

#-------------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------------
#Engineering new variables
#-------------------------------------------------------------------------------------------------

base_data <- read.table("D:/UTS/MDSI/Sem 01/Data, Algos, and Meaning/Assignments/DAM Assignment 2/student_data/student_data/base.csv", sep = ",", header = T)

pred <- read.table("D:/UTS/MDSI/Sem 01/Data, Algos, and Meaning/Assignments/DAM Assignment 2/student_data/student_data/pred.csv", sep = ",", header = T)


base_data <- base_data %>% 
                mutate(item_length_band = case_when(item_imdb_length < 50 ~ "Less than 50",
                                                    item_imdb_length < 100 ~ "50 to 100",
                                                    item_imdb_length < 150 ~ "100 to 150",
                                                    item_imdb_length < 200 ~ "150 to 200",
                                                    item_imdb_length > 200 ~ "More than 200"))

base_data$item_length_band <- as.factor(base_data$item_length_band)

user_att_mean <- base_data %>% group_by(age, occupation, gender) %>% 
                                summarise(user_rat = mean(rating))

item_att_mean <- base_data %>% group_by(unknown, action, adventure, animation, childrens, comedy, crime, documentary, drama, fantasy, film_noir, horror,  musical,  mystery, romance,  sci_fi,  thriller, war, western, item_length_band, item_imdb_mature_rating) %>% summarise(item_rat = mean(rating))


base_data <- base_data %>% left_join(user_att_mean, by = c("age", "occupation", "gender"))

nrow(base_data %>% filter(is.na(user_rat))) #0

base_data <- base_data %>% left_join(item_att_mean, by = c("unknown", "action", "adventure", "animation", "childrens", "comedy", "crime", "documentary", "drama", "fantasy", "film_noir", "horror",  "musical",  "mystery", "romance",  "sci_fi",  "thriller", "war", "western", "item_length_band", "item_imdb_mature_rating"))

nrow(base_data %>% filter(is.na(item_rat))) #0


#Test-train split
set.seed(1058)

train_indices <- sample(seq_len(nrow(base_data)), size = floor(0.75*nrow(base_data)))

train <- base_data[train_indices,]
test <- base_data[-train_indices,]


lin <- lm(rating ~ item_rat + user_rat, data = train)
summary(lin)

test$pred <- predict(lin, newdata = test, type = "response")

rmse(test$pred, test$rating) #1.108914

#XGB attempt
row.has.na <- apply(train, 1, function(x){any(is.na(x))})

train <- train[!row.has.na,]

row.has.na <- apply(test, 1, function(x){any(is.na(x))})

test <- test[!row.has.na,]

target <- train$rating

feature_names <- names(train[,c(8,46:47)])

dtrain <- xgb.DMatrix(data = as.matrix(train[,c(8,46:47)]), label = target)
dtest <- xgb.DMatrix(data=as.matrix(test[,feature_names]), missing=NA)

params <- list(booster = "gblinear", objective = "reg:linear", eta=0.3, gamma=0, max_depth=6, min_child_weight=1, subsample=1, colsample_bytree=1)

foldsCV <- createFolds(target, k=10, list=TRUE, returnTrain=FALSE)

xgb_cv <- xgb.cv(data=dtrain,
                 params=params,
                 nrounds=5000,
                 prediction=TRUE,
                 maximize=FALSE,
                 folds=foldsCV,
                 early_stopping_rounds = 30,
                 print_every_n = 5
)

print(xgb_cv$evaluation_log[which.min(xgb_cv$evaluation_log$test_rmse_mean)])

nrounds <- xgb_cv$best_iteration #2686

xgb <- xgb.train(params = params
                 , data = dtrain
                 # , watchlist = list(train = dtrain)
                 , nrounds = nrounds
                 , verbose = 1
                 , print_every_n = 5
                 #, feval = amm_mae
)

importance_matrix <- xgb.importance(feature_names,model=xgb)

preds <- predict(xgb,dtest)

test$pred <- preds

rmse(test$pred, test$rating)

#Predicting on the predict dataset
pred <- pred %>% 
            mutate(item_length_band = case_when(item_imdb_length < 50 ~ "Less than 50",
                                      item_imdb_length < 100 ~ "50 to 100",
                                      item_imdb_length < 150 ~ "100 to 150",
                                      item_imdb_length < 200 ~ "150 to 200",
                                      item_imdb_length > 200 ~ "More than 200"))

pred$item_length_band <- as.factor(pred$item_length_band)

pred <- pred %>% left_join(user_att_mean, by = c("age", "occupation", "gender"))

pred <- pred %>% left_join(item_att_mean, by = c("unknown", "action", "adventure", "animation", "childrens", "comedy", "crime", "documentary", "drama", "fantasy", "film_noir", "horror",  "musical",  "mystery", "romance",  "sci_fi",  "thriller", "war", "western", "item_length_band"))

row.has.na <- apply(pred, 1, function(x){any(is.na(x))})

pred <- pred[!row.has.na,]

dpred <- xgb.DMatrix(data=as.matrix(pred[,feature_names]), missing=NA)

preds <- predict(xgb,dpred)

pred$pred <- preds

results_df <- pred[,c(1,28,47)]
results_df$user_item <- paste(results_df$user_id,results_df$item_id,sep = "_")
results_df <- results_df[,c(-1,-2)]

nrow(results_df %>% filter(is.na(pred)))

results_df <- results_df %>% 
      mutate(pred = case_when((pred < 1 | is.na(pred)) ~ 1, pred > 5 ~ 5, TRUE ~ pred)) 

names(results_df)[1] <- "rating"

write.table(results_df, "D:/UTS/MDSI/Sem 01/Data, Algos, and Meaning/Assignments/DAM Assignment 2/student_data/student_data/submission.csv", sep = ",", row.names = F)
