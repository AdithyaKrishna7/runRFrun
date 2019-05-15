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

####################### MODEL 15 GBM ntrees=10K - submitted to Kaggle on 13/05/2019 8.45am RMSE =  0.93172, test Train RMSE = 0.9069937 ########################

# Load train and test datasets

stuTrain <- readRDS('AT2_train_STUDENT.rds')
stuTest <- readRDS('AT2_test_STUDENT.rds')
#scrape <- readRDS('scrape.rds')


#########################################################################################
########################################## EDA ##########################################
#########################################################################################





#########################################################################################
######################### Feature Engineering and data cleaning #########################
#########################################################################################

# convert user and item IDs to characters for consistency
stuTrain$user_id <- as.character(stuTrain$user_id)
stuTest$user_id <- as.character(stuTest$user_id)

#Extract time of day of rating for use in model
stuTrain$hrStamp <- as.factor(hour(ymd_hms(stuTrain$timestamp)))
stuTest$hrStamp <- as.factor(hour(ymd_hms(stuTest$timestamp)))

#Extract Month item was released for use in model
stuTrain$relMonth <- as.factor(month(ymd_hms(as.POSIXct.Date(stuTrain$release_date))))
stuTest$relMonth <- as.factor(month(ymd_hms(as.POSIXct.Date(stuTest$release_date))))


# stuTrain$relMonth <- fct_explicit_na(stuTrain$relMonth)
# stuTest$relMonth <- fct_explicit_na(stuTest$relMonth)

# Band item_imdb_length to overcome missing values issue

stuTrain <- stuTrain %>% 
  mutate(item_length_band = case_when(item_imdb_length < 50 ~ "Less than 50",
                                      item_imdb_length < 100 ~ "50 to 100",
                                      item_imdb_length < 150 ~ "100 to 150",
                                      item_imdb_length < 200 ~ "150 to 200",
                                      item_imdb_length > 200 ~ "More than 200"))

stuTrain$item_length_band <- as.factor(stuTrain$item_length_band)

stuTest <- stuTest %>% 
  mutate(item_length_band = case_when(item_imdb_length < 50 ~ "Less than 50",
                                      item_imdb_length < 100 ~ "50 to 100",
                                      item_imdb_length < 150 ~ "100 to 150",
                                      item_imdb_length < 200 ~ "150 to 200",
                                      item_imdb_length > 200 ~ "More than 200"))
stuTest$item_length_band <- as.factor(stuTest$item_length_band)


## Calculate user mean rating

userMeanRating_train <- stuTrain %>%
  group_by(user_id)%>%
  summarise(user_mean_rating = mean(rating))

## Calculate user Genre Rating, genre Unknown and Western dropped due to low prevelance
userGenreRating <- stuTrain %>%
  group_by(age_band, gender, action, adventure, animation, childrens, comedy, crime, drama, film_noir, horror,  musical,  mystery, romance,  sci_fi, thriller, war)%>%
  summarise(user_genre_mean_rating = mean(rating))

## Caluculate userIndex score based on age, gender, occupation and hour of the day rating was provided
userIndex <- stuTrain %>%
  group_by(age, gender,occupation,hrStamp) %>%
  summarise(userIndexMeanRating = mean(rating))


## Caluculate itemIndex score based on item_mean_rating, item_imdb_length, item_imdb_mature_rating
itemIndex <- stuTrain %>%
  group_by(item_mean_rating, item_imdb_length, item_imdb_mature_rating) %>%
  summarise(itemIndexMeanRating = mean(rating))

# Join engineered varaibles into stuTrain and stuTest

stuTrain <- stuTrain %>%
  left_join(userMeanRating_train, by = 'user_id') %>%
  left_join(userGenreRating, by = c("age_band", "gender", "action", "adventure", "animation", "childrens", "comedy", "crime", "drama", "film_noir", "horror",  "musical",  "mystery", "romance",  "sci_fi",  "thriller", "war"))%>%
  left_join(itemIndex, by = c("item_mean_rating", "item_imdb_length", "item_imdb_mature_rating"))%>%
  left_join(userIndex, by = c("age", "gender","occupation", "hrStamp"))

stuTest <- stuTest %>%
  left_join(userMeanRating_train, by = 'user_id') %>%
  left_join(userGenreRating, by = c("age_band", "gender", "action", "adventure", "animation", "childrens", "comedy", "crime", "drama", "film_noir", "horror",  "musical",  "mystery", "romance",  "sci_fi",  "thriller", "war"))%>%
  left_join(itemIndex, by = c("item_mean_rating", "item_imdb_length", "item_imdb_mature_rating"))%>%
  left_join(userIndex, by = c("age", "gender","occupation", "hrStamp"))

summary(stuTrain)
summary(stuTest)


## Create Test / Train Split
set.seed(42)
trSize <- floor(0.8*nrow(stuTrain))
trIndex <- sample(seq_len(nrow(stuTrain)), size = trSize)

trSet <- stuTrain[trIndex,]
tstSet <- stuTrain[-trIndex,]

# ================== GBM MODEL ================

newGBM <- gbm(rating ~ user_mean_rating + user_genre_mean_rating + itemIndexMeanRating + 
                userIndexMeanRating,
              data = trSet,
              distribution = "gaussian",
              interaction.depth = 10,
              n.trees = 10000,
              shrinkage = 0.01,
              bag.fraction = 0.45,
              n.cores = 3)
newGBM


# Make predictions for Test Set
tstSet$prediction <- predict(newGBM, tstSet, n.tree = 10000)

#check for RMSE
ModelMetrics::rmse(tstSet$rating, tstSet$prediction) ## Test/Train RMSE = 0.9071594

# Make prediction for submission test set
stuTest$rating <- predict(newGBM, stuTest, n.tree = 10000)
# ======================== Create submission file ========================

# create results dataframe with final rating, user_id and item_id
results_df <- stuTest[,c(1,6,56)]

#create variable <user_id>_<item_id> for submission
results_df$user_item <- paste(results_df$user_id,results_df$item_id,sep = "_")

#drop user_id and item_id
results_df <- results_df[,c(-1,-2)]

#check summary for missing values
summary(results_df)

#Optimise ratings to 1 or 5 when prediction is lower than 1 or higher than 5
results_df <- results_df %>% mutate(rating = case_when((rating < 1 | is.na(rating)) ~ 1, rating > 5 ~ 5, TRUE ~ rating) ) 

#Write output CSV file for submission

write.csv(results_df, file = "runRFrun_Submission_gbm_15-20190513.csv", row.names = FALSE)

## Kaggle RMSE for above results is 0.93172