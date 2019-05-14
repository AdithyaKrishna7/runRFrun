# == Libraries ================================================================
# Load packages
library(tidyverse)
library(lubridate)
library(forcats)
library(caret)
library(DataExplorer)
library(corrplot)
library(recommenderlab)
library(reshape2)
library(recosystem)
library(parallel)
library(doParallel)
library(xgboost)
library(gbm)

## Allow multiple cores for processing - Run Code for all Models

cluster <- makeCluster(detectCores() - 1)
registerDoParallel(cluster)


# stuTrain <- readRDS('AT2_train_STUDENT.rds')
# stuTest <- readRDS('AT2_test_STUDENT.rds')
# scrape <- readRDS('scrape.rds')
# 
# # #======================== create user mean rating (train) and add to treain and test data set ========================#
# # stuTrain$user_id <- as.factor(stuTrain$user_id)
# # stuTest$user_id <- as.factor(stuTest$user_id)
# 
# stuTrain$hrStamp <- as.factor(hour(ymd_hms(stuTrain$timestamp)))
# stuTest$hrStamp <- as.factor(hour(ymd_hms(stuTest$timestamp)))
# stuTrain$relMonth <- as.factor(month(ymd_hms(as.POSIXct.Date(stuTrain$release_date))))
# stuTest$relMonth <- as.factor(month(ymd_hms(as.POSIXct.Date(stuTest$release_date))))
# 
# 
# stuTrain$relMonth <- fct_explicit_na(stuTrain$relMonth)
# stuTest$relMonth <- fct_explicit_na(stuTest$relMonth)
# 
# stuTrain <- stuTrain %>% 
#   mutate(item_length_band = case_when(item_imdb_length < 50 ~ "Less than 50",
#                                       item_imdb_length < 100 ~ "50 to 100",
#                                       item_imdb_length < 150 ~ "100 to 150",
#                                       item_imdb_length < 200 ~ "150 to 200",
#                                       item_imdb_length > 200 ~ "More than 200",
#                                       is.na(item_imdb_length) ~ "Unknown"))
# 
# stuTrain$item_length_band <- as.factor(stuTrain$item_length_band)
# 
# stuTest <- stuTest %>% 
#   mutate(item_length_band = case_when(item_imdb_length < 50 ~ "Less than 50",
#                                       item_imdb_length < 100 ~ "50 to 100",
#                                       item_imdb_length < 150 ~ "100 to 150",
#                                       item_imdb_length < 200 ~ "150 to 200",
#                                       item_imdb_length > 200 ~ "More than 200",
#                                       is.na(item_imdb_length) ~ "Unknown"))
# stuTest$item_length_band <- as.factor(stuTest$item_length_band)
# 
# stuTrain <- stuTrain %>%
#   mutate(imdb_top1000Avg = case_when(item_imdb_top_1000_voters_average < 3 ~ "Poor",
#                                 item_imdb_top_1000_voters_average < 5 ~ "Avergae",
#                                 item_imdb_top_1000_voters_average < 7 ~ "Good",
#                                 item_imdb_top_1000_voters_average < 10 ~ "Excellent",
#                                 is.na(item_imdb_top_1000_voters_average) ~ "Unknown"))
# stuTrain$imdb_top1000Avg <- as.factor(stuTrain$imdb_top1000Avg)
# 
# stuTest <- stuTest %>%
#   mutate(imdb_top1000Avg = case_when(item_imdb_top_1000_voters_average < 3 ~ "Poor",
#                                 item_imdb_top_1000_voters_average < 5 ~ "Avergae",
#                                 item_imdb_top_1000_voters_average < 7 ~ "Good",
#                                 item_imdb_top_1000_voters_average < 10 ~ "Excellent",
#                                 is.na(item_imdb_top_1000_voters_average) ~ "Unknown"))
# stuTest$imdb_top1000Avg <- as.factor(stuTest$imdb_top1000Avg)
# 
# stuTrain <- stuTrain %>%
#   mutate(imdb_rating = case_when(item_imdb_rating_of_ten < 3 ~ "Poor",
#                             item_imdb_rating_of_ten < 5 ~ "Avergae",
#                             item_imdb_rating_of_ten < 7 ~ "Good",
#                             item_imdb_rating_of_ten < 10 ~ "Excellent",
#                             is.na(item_imdb_rating_of_ten) ~ "Unknown"))
# stuTrain$imdb_rating<- as.factor(stuTrain$imdb_rating)
# 
# stuTest <- stuTest %>%
#   mutate(imdb_rating = case_when(item_imdb_rating_of_ten < 3 ~ "Poor",
#                             item_imdb_rating_of_ten < 5 ~ "Avergae",
#                             item_imdb_rating_of_ten < 7 ~ "Good",
#                             item_imdb_rating_of_ten < 10 ~ "Excellent",
#                             is.na(item_imdb_rating_of_ten) ~ "Unknown"))
# stuTest$imdb_rating<- as.factor(stuTest$imdb_rating)
# 
# userMeanRating_train <- stuTrain %>%
#   group_by(user_id)%>%
#   summarise(user_mean_rating = mean(rating))
# 
# userGenreRating <- stuTrain %>%
#   group_by(age_band, gender, action, adventure, animation, childrens, comedy, crime, drama, film_noir, horror,  musical,  mystery, romance,  sci_fi, thriller, war)%>%
#   summarise(user_genre_mean_rating = mean(rating))
# 
# userIndex <- stuTrain %>%
#   group_by(age_band, hrStamp, occupation,  imdb_top1000Avg) %>%
#   summarise(userIndexMeanRating = mean(rating))
# 
# itemIndex <- stuTrain %>%
#   group_by(item_mean_rating, item_imdb_length, item_imdb_mature_rating, imdb_rating) %>%
#   summarise(itemIndexMeanRating = mean(rating))
# 
# # Join values into stuTrain and stuTest
# 
# stuTrain <- stuTrain %>%
#   left_join(userMeanRating_train, by = 'user_id') %>%
#   left_join(userGenreRating, by = c("age_band", "gender", "action", "adventure", "animation", "childrens", "comedy", "crime", "drama", "film_noir", "horror",  "musical",  "mystery", "romance",  "sci_fi",  "thriller", "war"))%>%
#   left_join(itemIndex, by = c("item_mean_rating", "item_imdb_length", "item_imdb_mature_rating", "imdb_rating"))%>%
#   left_join(userIndex, by = c("age_band", "hrStamp","occupation", "imdb_top1000Avg"))
# 
# stuTest <- stuTest %>%
#   left_join(userMeanRating_train, by = 'user_id') %>%
#   left_join(userGenreRating, by = c("age_band", "gender", "action", "adventure", "animation", "childrens", "comedy", "crime", "drama", "film_noir", "horror",  "musical",  "mystery", "romance",  "sci_fi",  "thriller", "war"))%>%
#   left_join(itemIndex, by = c("item_mean_rating", "item_imdb_length", "item_imdb_mature_rating", "imdb_rating"))%>%
#   left_join(userIndex, by = c("age_band", "hrStamp","occupation",  "imdb_top1000Avg"))
# 
# summary(stuTrain)
# summary(stuTest)
# 
# # stuTest$user_genre_mean_rating[is.na(stuTest$user_genre_mean_rating)] <- stuTest$item_mean_rating[is.na(stuTest$user_genre_mean_rating)]
# 
# 
# ## Create Test / Train Split
# set.seed(11234)
# trSize <- floor(0.75*nrow(stuTrain))
# trIndex <- sample(seq_len(nrow(stuTrain)), size = trSize)
# 
# # trSet <- stuTrain[trIndex,]
# # tstSet <- stuTrain[-trIndex,]
# 
# # subet to only include the columns needed
# 
# stuTrain <- stuTrain[,c(1,6,7,55:58)]
# stuTest <- stuTest[,c(1,6,54:57)]
# 
# 
# trSet <- stuTrain[trIndex,]
# tstSet <- stuTrain[-trIndex,]
# 
# # ================== GBM MODEL ================
# 
# 
# newGBM <- gbm(rating ~ user_mean_rating + user_mean_rating + user_genre_mean_rating + 
#               itemIndexMeanRating + userIndexMeanRating,
#               data = trSet,
#               distribution = "gaussian",
#               interaction.depth = 10,
#               n.trees = 10000,
#               shrinkage = 0.01,
#               bag.fraction = 0.45,
#               cv.folds = 10,
#               n.cores = 3)
# newGBM
# 
# 
# 
# tstSet$prediction <- predict(newGBM, tstSet, n.tree = 10000) 
# ModelMetrics::rmse(tstSet$rating, tstSet$prediction)#0.9069582
# 
# 
# stuTest$rating <- predict(newGBM, stuTest, n.tree = 10000)
# # ======================== Create submission file ========================
# results_df <- stuTest[,c(1,2,7)]
# results_df$user_item <- paste(results_df$user_id,results_df$item_id,sep = "_")
# results_df <- results_df[,c(-1,-2)]
# summary(results_df)
# results_df <- results_df %>% mutate(rating = case_when((rating < 1 | is.na(rating)) ~ 1, rating > 5 ~ 5, TRUE ~ rating) ) 
# 
# # write.csv(results_df, file = "runRFrun_Submission_gbm_19-20190514.csv", row.names = FALSE)



  # ================== XGB MODEL ================

## Allow multiple cores for processing - Run Code for all Models

cluster <- makeCluster(detectCores() - 1)
registerDoParallel(cluster)


stuTrain <- readRDS('AT2_train_STUDENT.rds')
stuTest <- readRDS('AT2_test_STUDENT.rds')
scrape <- readRDS('scrape.rds')

# #======================== create user mean rating (train) and add to treain and test data set ========================#
# stuTrain$user_id <- as.factor(stuTrain$user_id)
# stuTest$user_id <- as.factor(stuTest$user_id)

stuTrain$hrStamp <- as.factor(hour(ymd_hms(stuTrain$timestamp)))
stuTest$hrStamp <- as.factor(hour(ymd_hms(stuTest$timestamp)))
stuTrain$relMonth <- as.factor(month(ymd_hms(as.POSIXct.Date(stuTrain$release_date))))
stuTest$relMonth <- as.factor(month(ymd_hms(as.POSIXct.Date(stuTest$release_date))))


stuTrain$relMonth <- fct_explicit_na(stuTrain$relMonth)
stuTest$relMonth <- fct_explicit_na(stuTest$relMonth)

stuTrain <- stuTrain %>% 
  mutate(item_length_band = case_when(item_imdb_length < 50 ~ "Less than 50",
                                      item_imdb_length < 100 ~ "50 to 100",
                                      item_imdb_length < 150 ~ "100 to 150",
                                      item_imdb_length < 200 ~ "150 to 200",
                                      item_imdb_length > 200 ~ "More than 200",
                                      is.na(item_imdb_length) ~ "Unknown"))

stuTrain$item_length_band <- as.factor(stuTrain$item_length_band)

stuTest <- stuTest %>% 
  mutate(item_length_band = case_when(item_imdb_length < 50 ~ "Less than 50",
                                      item_imdb_length < 100 ~ "50 to 100",
                                      item_imdb_length < 150 ~ "100 to 150",
                                      item_imdb_length < 200 ~ "150 to 200",
                                      item_imdb_length > 200 ~ "More than 200",
                                      is.na(item_imdb_length) ~ "Unknown"))
stuTest$item_length_band <- as.factor(stuTest$item_length_band)

stuTrain <- stuTrain %>%
  mutate(imdb_top1000Avg = case_when(item_imdb_top_1000_voters_average < 3 ~ "Poor",
                                     item_imdb_top_1000_voters_average < 5 ~ "Avergae",
                                     item_imdb_top_1000_voters_average < 7 ~ "Good",
                                     item_imdb_top_1000_voters_average < 10 ~ "Excellent",
                                     is.na(item_imdb_top_1000_voters_average) ~ "Unknown"))
stuTrain$imdb_top1000Avg <- as.factor(stuTrain$imdb_top1000Avg)

stuTest <- stuTest %>%
  mutate(imdb_top1000Avg = case_when(item_imdb_top_1000_voters_average < 3 ~ "Poor",
                                     item_imdb_top_1000_voters_average < 5 ~ "Avergae",
                                     item_imdb_top_1000_voters_average < 7 ~ "Good",
                                     item_imdb_top_1000_voters_average < 10 ~ "Excellent",
                                     is.na(item_imdb_top_1000_voters_average) ~ "Unknown"))
stuTest$imdb_top1000Avg <- as.factor(stuTest$imdb_top1000Avg)

stuTrain <- stuTrain %>%
  mutate(imdb_rating = case_when(item_imdb_rating_of_ten < 3 ~ "Poor",
                                 item_imdb_rating_of_ten < 5 ~ "Avergae",
                                 item_imdb_rating_of_ten < 7 ~ "Good",
                                 item_imdb_rating_of_ten < 10 ~ "Excellent",
                                 is.na(item_imdb_rating_of_ten) ~ "Unknown"))
stuTrain$imdb_rating<- as.factor(stuTrain$imdb_rating)

stuTest <- stuTest %>%
  mutate(imdb_rating = case_when(item_imdb_rating_of_ten < 3 ~ "Poor",
                                 item_imdb_rating_of_ten < 5 ~ "Avergae",
                                 item_imdb_rating_of_ten < 7 ~ "Good",
                                 item_imdb_rating_of_ten < 10 ~ "Excellent",
                                 is.na(item_imdb_rating_of_ten) ~ "Unknown"))
stuTest$imdb_rating<- as.factor(stuTest$imdb_rating)

userMeanRating_train <- stuTrain %>%
  group_by(user_id)%>%
  summarise(user_mean_rating = mean(rating))

userGenreRating <- stuTrain %>%
  group_by(age_band, gender, action, adventure, animation, childrens, comedy, crime, drama, film_noir, horror,  musical,  mystery, romance,  sci_fi, thriller, war)%>%
  summarise(user_genre_mean_rating = mean(rating))

userIndex <- stuTrain %>%
  group_by(age_band, hrStamp, occupation,  imdb_top1000Avg) %>%
  summarise(userIndexMeanRating = mean(rating))

itemIndex <- stuTrain %>%
  group_by(item_mean_rating, item_imdb_length, item_imdb_mature_rating, imdb_rating) %>%
  summarise(itemIndexMeanRating = mean(rating))

# Join values into stuTrain and stuTest

stuTrain <- stuTrain %>%
  left_join(userMeanRating_train, by = 'user_id') %>%
  left_join(userGenreRating, by = c("age_band", "gender", "action", "adventure", "animation", "childrens", "comedy", "crime", "drama", "film_noir", "horror",  "musical",  "mystery", "romance",  "sci_fi",  "thriller", "war"))%>%
  left_join(itemIndex, by = c("item_mean_rating", "item_imdb_length", "item_imdb_mature_rating", "imdb_rating"))%>%
  left_join(userIndex, by = c("age_band", "hrStamp","occupation", "imdb_top1000Avg"))

stuTest <- stuTest %>%
  left_join(userMeanRating_train, by = 'user_id') %>%
  left_join(userGenreRating, by = c("age_band", "gender", "action", "adventure", "animation", "childrens", "comedy", "crime", "drama", "film_noir", "horror",  "musical",  "mystery", "romance",  "sci_fi",  "thriller", "war"))%>%
  left_join(itemIndex, by = c("item_mean_rating", "item_imdb_length", "item_imdb_mature_rating", "imdb_rating"))%>%
  left_join(userIndex, by = c("age_band", "hrStamp","occupation",  "imdb_top1000Avg"))

summary(stuTrain)
summary(stuTest)

# stuTest$user_genre_mean_rating[is.na(stuTest$user_genre_mean_rating)] <- stuTest$item_mean_rating[is.na(stuTest$user_genre_mean_rating)]


## Create Test / Train Split
set.seed(11234)
trSize <- floor(0.75*nrow(stuTrain))
trIndex <- sample(seq_len(nrow(stuTrain)), size = trSize)

# trSet <- stuTrain[trIndex,]
# tstSet <- stuTrain[-trIndex,]

# subet to only include the columns needed

stuTrain_xgb <- stuTrain[,c(1,6,7,55:58)]
stuTest_xgb <- stuTest[,c(1,6,54:57)]


trSet <- stuTrain_xgb[trIndex,]
tstSet <- stuTrain_xgb[-trIndex,]



target <- trSet$rating
feature_names <- names(trSet[,c(4:7)])


dtrain <- xgb.DMatrix(data = as.matrix(trSet[,feature_names]), label = target)
dtest <- xgb.DMatrix(data=as.matrix(tstSet[,feature_names]), missing=NA)
#
params <- list(booster = "gbtree", 
               objective = "reg:linear", 
               #objective = "reg:squarederror",
               eta=0.1, gamma=0, 
               max_depth=6, 
               min_child_weight=1, 
               subsample=0.5, 
               colsample_bytree=1)

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

nrounds <- xgb_cv$best_iteration #45

xgb <- xgb.train(params = params
                 , data = dtrain
                 # , watchlist = list(train = dtrain)
                 , nrounds = nrounds
                 , verbose = 1
                 , print_every_n = 5
                 #, feval = amm_mae
)

importance_matrix <- xgb.importance(feature_names,model=xgb)
xgb
preds <- predict(xgb,dtest)

tstSet$prediction <- preds
ModelMetrics::rmse(tstSet$rating, tstSet$prediction)#0.9091451

# Predicting on stuTest
dpred <- xgb.DMatrix(data=as.matrix(stuTest_xgb[,feature_names]), missing=NA)
preds <- predict(xgb,dpred)

stuTest_xgb$rating <- preds

# ======================== Create submission file ========================
results_df_XGB <- stuTest_xgb[,c(1,2,7)]
results_df_XGB$user_item <- paste(results_df_XGB$user_id,results_df_XGB$item_id,sep = "_")
results_df_XGB <- results_df_XGB[,c(-1,-2)]
summary(results_df_XGB)
results_df_XGB <- results_df_XGB %>% mutate(rating = case_when((rating < 1 | is.na(rating)) ~ 1, rating > 5 ~ 5, TRUE ~ rating) ) 

write.csv(results_df_XGB, file = "runRFrun_Submission_XGB-KM_01-20190515.csv", row.names = FALSE)


####TRY XGB###

stuTest$rating <- predict(newGBM, stuTest)