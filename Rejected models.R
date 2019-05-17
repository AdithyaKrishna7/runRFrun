#rm(list = ls())
#== Libraries ================================================================
#Load packages
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
library(sqldf)
library(randomForest)
library(mltools)

#Original data
stuTrain <- readRDS('AT2_train_STUDENT.rds')
stuTest <- readRDS('AT2_test_STUDENT.rds')
scrape <- readRDS('scrape.rds')


######################### Feature Engineering and data cleaning #########################
#########################################################################################

stuTrain <- readRDS('AT2_train_STUDENT.rds')
stuTest <- readRDS('AT2_test_STUDENT.rds')

#Convert user and item IDs to characters for consistency
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

#Band item_imdb_length to overcome missing values issue

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


#Calculate user mean rating

userMeanRating_train <- stuTrain %>%
  group_by(user_id)%>%
  summarise(user_mean_rating = mean(rating))

#Calculate user Genre Rating, genre Unknown and Western dropped due to low prevelance
userGenreRating <- stuTrain %>%
  group_by(age_band, gender, action, adventure, animation, childrens, comedy, crime, drama, film_noir, horror,  musical,  mystery, romance,  sci_fi, thriller, war)%>%
  summarise(user_genre_mean_rating = mean(rating))

#Calculate userIndex score based on age, gender, occupation and hour of the day rating was provided
userIndex <- stuTrain %>%
  group_by(age, gender,occupation,hrStamp) %>%
  summarise(userIndexMeanRating = mean(rating))


#Calculate itemIndex score based on item_mean_rating, item_imdb_length, item_imdb_mature_rating
itemIndex <- stuTrain %>%
  group_by(item_mean_rating, item_imdb_length, item_imdb_mature_rating) %>%
  summarise(itemIndexMeanRating = mean(rating))

#Join engineered varaibles into stuTrain and stuTest

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


#Create Test / Train Split
set.seed(42)
trSize <- floor(0.8*nrow(stuTrain))
trIndex <- sample(seq_len(nrow(stuTrain)), size = trSize)

trSet <- stuTrain[trIndex,]
tstSet <- stuTrain[-trIndex,]


#===============================================================================================
#== Rejected models ============================================================================

##### MODEL 0 Linear Regression ################################################################

lin <- lm(rating ~ item_mean_rating + user_mean_rating + user_genre_mean_rating + itemIndexMeanRating + userIndexMeanRating, data = trSet)		
summary(lin)		

tstSet$pred <- predict(lin, newdata = tstSet, type = "response")

rmse(tstSet$pred, tstSet$rating) #0.984289

tstSet$pred <- NULL

stuTest$pred <- predict(lin, newdata = stuTest, type = "response")

results_df <- stuTest %>% select(user_id, item_id, pred)
results_df$user_item <- paste(results_df$user_id,results_df$item_id,sep = "_")
results_df <- results_df[,c(-1,-2)]

names(results_df)[1] <- "rating"

nrow(results_df %>% filter(is.na(rating)))

results_df <- results_df %>% 
  mutate(rating = case_when(rating < 1 ~ 1, 
                            rating > 5 ~ 5, 
                            is.na(rating) ~ 3.528011, 
                            TRUE ~ rating)) 

write.table(results_df, "D:/UTS/MDSI/Sem 01/Data, Algos, and Meaning/Assignments/DAM Assignment 2/student_data/student_data/submission_3.csv", sep = ",", row.names = F)

##### MODEL 1 Random Forest ####################################################################

rftrSet <- trSet %>% select(user_id, item_id, rating, item_mean_rating, user_mean_rating, user_genre_mean_rating, itemIndexMeanRating, userIndexMeanRating)

rftstSet <- tstSet %>% select(user_id, item_id, rating, item_mean_rating, user_mean_rating, user_genre_mean_rating, itemIndexMeanRating, userIndexMeanRating)

rand <- randomForest(rating ~ . - item_id - user_id, data = rftrSet, importance=TRUE, xtest=rftstSet[,-c(1:3)],ntree=250, keep.forest = T)

rmse(rand$test$predicted, rftstSet$rating) #0.9360473


##### MODEL 2 Recommendation model (User Based Collaborative Filtering) - submitted to Kaggle on 01/05/2019 #######

stuTrain <- readRDS('AT2_train_STUDENT.rds')
stuTest <- readRDS('AT2_test_STUDENT.rds')
scrape <- readRDS('scrape.rds')

## create simple user / movie / rating matrix

#ratMat_train <- as.data.frame(stuTrain[,c(1, 6,7)])
ratMat_train <- as.data.frame(stuTrain[,c(1:4,6,7,32:33)])

#ratMat_test <- as.data.frame(stuTest[,c(1,6)])
ratMat_test <- as.data.frame(stuTest[,c(1:4,6,31:32)])


#train_data <- acast(ratMat_train, ratMat_train$user_id~ratMat_train$item_id)
train_data <- acast(ratMat_train, user_id~item_id, value.var = "rating")

rating_mat_train <- as.matrix(train_data)

realRatMat <- as(rating_mat_train,"realRatingMatrix")
realRatMat_normalised <- normalize(realRatMat)

#test data as a rating matrix   
test_data <- acast(ratMat_test, user_id~item_id)
rating_mat_test <- as.matrix(test_data)

realRatMat_test <- as(rating_mat_test, "realRatingMatrix")


hist(getRatings(realRatMat),binwidth=1,col="green",breaks=15,main="Histogram of Ratings",xlab="Ratings")

qplot(getRatings(realRatMat_normalised),binwidth=1,main="Histogram of Normalized Ratings",xlab="Ratings")

iSuggest <- Recommender(realRatMat[1:nrow(realRatMat)],method="UBCF", 
                        param=list(normalize="Z-score", method="Cosine"))
print(iSuggest)


# predict using iSuggest model to get ratings and top 25
iSuggest_ratings <-predict(iSuggest,realRatMat[1:nrow(realRatMat)],type="ratings")
iSuggest_ratings #943 x 1682 rating matrix of class ‘realRatingMatrix’ with 1505603 ratings.

#iSuggest_top10movies <-predict(iSuggest,realRatMat[1:nrow(realRatMat)],type="topNList",n=25)


#recommendation list

iSuggest_list <- as(iSuggest_ratings, "list")
head(iSuggest_list, 10)


modelRating <- NULL
results_df <- data.frame()

for(i in 1:length(ratMat_test[,2])){
  userid <- ratMat_test[i,1]
  movieid <- ratMat_test[i,5]
  
  user_n <- as.data.frame(iSuggest_list[[userid]])
  user_n$id<-row.names(user_n)
  
  x = user_n[user_n$id==movieid,1]
  
  if (length(x)==0)
  {
    modelRating[i] <- 1
  }
  else
  {
    modelRating[i] <-x
  }
  results_df[i, "rating"] <- modelRating[i]
  results_df[i, "user_item"] <- paste(userid, movieid,sep = "_")
}


#write.csv(results_df[,c(1,2)], file = "runRFrun_second_submission-20190502.csv", row.names = FALSE)

fsvd <- funkSVD(realRatMat,k=2,verbose = TRUE)
test_svd <- tcrossprod(fsvd$U,fsvd$V)
sqrt(MSE(rating_mat_train,test_svd))

#RMSE = 0.8890359

##### MODEL 3 Recommendation model (Item Based Collaborative Filtering) ##############################################################################

#ratings matrix
ratMat_train <- as.data.frame(stuTrain[,c(1, 6,7)])
ratMat_test <- as.data.frame(stuTest[,c(1,6)])

#test & tran split

trainset_size <- floor(0.75 * nrow(ratMat_train))
set.seed(42) 
trainset_indices <- sample(seq_len(nrow(ratMat_train)), size = trainset_size)
trainset <- ratMat_train[trainset_indices, ]
testset <- ratMat_train[-trainset_indices, ]

#IBCF

train_data <- acast(trainset, trainset$item_id~trainset$user_id)
rating_mat_train <- as.matrix(train_data)

realRatMat <- as(rating_mat_train,"realRatingMatrix")
realRatMat_normalised <- normalize(realRatMat)

hist(getRatings(realRatMat),binwidth=1,col="green",breaks=15,main="Histogram of Ratings",xlab="Ratings")
qplot(getRatings(realRatMat_normalised),binwidth=1,main="Histogram of Normalized Ratings",xlab="Ratings")

#Recommender 
iSuggest <- Recommender(realRatMat[1:nrow(realRatMat)],method="IBCF", 
                        param=list(normalize="Z-score", method="Cosine", nn=5, minRating=1))

iSuggest_ratings <-predict(iSuggest,realRatMat[1:nrow(realRatMat)],type="ratings")
iSuggest_ratings #1643 x 943 rating matrix of class ‘realRatingMatrix’ with 404354 ratings

#Recommendation list

iSuggest_list <- as(iSuggest_ratings, "list")
head(iSuggest_list, 10)

modelRating <- NULL


for(i in 1:length(testset[,2])){
  userid <- testset[i,1] #vector of the user id from the test set
  movieid <- testset[i,2] #vector of the movie id from the test set
  
  user_n <- as.data.frame(iSuggest_list[[userid]]) #getting the predictions based on userid
  user_n$id<-row.names(user_n) #adding movie id labels/names
  
  x = user_n[user_n$id==movieid,1]
  
  if (length(x)==0)
  {
    modelRating[i] <- 1 #if the length of x is zero then no ratings otherwise x
  }
  else
  {
    modelRating[i] <-x
  }
}

rmse(modelRating, testset$rating) #2.3


##### MODEL 4 Matrix Factorisation - submitted to Kaggle on 02/05/2019#########################

r = Reco()
trainMemory = data_memory(user_index = trSet$user_id, item_index = trSet$item_id, rating = trSet$rating, index1 = TRUE)


opt <- r$tune(trainMemory, 
              opts = list(
                dim      = c(65), # number of latent factors    
                costp_l2 = c(0.01, 0.1), # L2 regularization cost for user factors         
                costq_l2 = c(0.01, 0.1), # L2 regularization cost for item factors       
                costp_l1 = 0, # L1 regularization cost for user factors                    
                costq_l1 = 0, # L1 regularization cost for item factors                           
                lrate    = c(0.01, 0.1), # learning rate, which can be thought of as the step size in gradient descent.          
                lrate    = c(0.1, 0.2,0.5), # learning rate, which can be thought of as the step size in gradient descent.          
                
                nthread  = 4,  # number of threads for parallel computing
                nfold = 10, # number of folds in cross validation.  
                niter    = 10, #  number of iterations
                verbose  = FALSE
              ))
opt = r$tune(trainMemory, opts = list(
  dim = c(200),
  lrate = c(0.001, 0.1),
  nthread = 4,
  nfold = 5,
  niter = 10))

r$train(trainMemory, 
        opts = c(opt$min,                    
                 niter = 100, nthread = 4)) 
#rm(trainMemory)

testMemory <- data_memory(user_index = tstSet$user_id, item_index = tstSet$item_id,rating = tstSet$rating, index1 = TRUE)
tstSet$prediction <- r$predict(testMemory, out_memory())

tstSet <- tstSet %>% mutate(prediction = case_when(prediction < 1 ~ 1, prediction > 5 ~ 5, TRUE ~ prediction) ) 

ModelMetrics::rmse(tstSet$rating, tstSet$prediction)

validateMem <- data_memory(user_index = stuTest$user_id, item_index = stuTest$item_id, index1 = TRUE)
stuTest$rating <- r$predict(validateMem,out_memory())



results_df <- stuTest[,c(1,6,49)]
results_df$user_item <- paste(results_df$user_id,results_df$item_id,sep = "_")
results_df <- results_df[,c(-1,-2)]
results_df <- results_df %>% mutate(rating = case_when(rating < 1 ~ 1, rating > 5 ~ 5, TRUE ~ rating) ) 

#write.csv(results_df, file = "runRFrun_Submission_MF_1-20190502.csv", row.names = FALSE)

##### MODEL 5 GBM - submitted to Kaggle on 06/05/2019 RMSE = 0.97817 ########################

trainControls <- trainControl(
  method = "repeatedcv",
  number = 10,
  repeats = 10,
  allowParallel = TRUE)

gbmFit1 <- train(rating ~ itemIndexMeanRating + userIndexMeanRating, data = trSet, 
                 method = "gbm", 
                 trControl = trainControls,
                 verbose = FALSE)

tstSet$prediction <- predict(gbmFit1, tstSet)
stuTest$rating <- predict(gbmFit1, stuTest)
ModelMetrics::rmse(tstSet$rating, tstSet$prediction)


results_df <- stuTest[,c(1,6,51)]
results_df$user_item <- paste(results_df$user_id,results_df$item_id,sep = "_")
results_df <- results_df[,c(-1,-2)]
results_df <- results_df %>% mutate(rating = case_when((rating < 1 | is.na(rating)) ~ 1, rating > 5 ~ 5, TRUE ~ rating) ) 

#write.csv(results_df, file = "runRFrun_Submission_gbm-20190506.csv", row.names = FALSE)

##### MODEL 5a GBM - submitted to Kaggle on 06/05/2019 10.40pm RMSE = 0.96958 , test Train RMSE = 0.9682333########################

#Import preprocessed data - Run Code for all Models
stuTrain <- readRDS('AT2_train_STUDENT.rds')
stuTest <- readRDS('AT2_test_STUDENT.rds')
scrape <- readRDS('scrape.rds')


tempDf <- stuTrain[, c("user_id", "occupation", "gender", "age", "item_id", "item_imdb_mature_rating", "rating", "item_imdb_length", "unknown", "action", "adventure", "animation", "childrens", "comedy", "crime", "documentary","drama", "fantasy", "film_noir", "horror",  "musical",  "mystery", "romance",  "sci_fi",  "thriller", "war", "western")]

userIndex <- tempDf %>%
  group_by(age, gender, occupation)%>%
  summarise(userIndexMeanRating = mean(rating))

itemIndex <- tempDf %>%
  group_by(unknown, action, adventure, animation, childrens, comedy, crime, documentary, drama, fantasy, film_noir, horror,  musical,  mystery, romance,  sci_fi,  thriller, war, western, item_imdb_length, item_imdb_mature_rating) %>%
  summarise(itemIndexMeanRating = mean(rating))

stuTrain <- stuTrain %>%
  left_join(itemIndex, by = c("unknown", "action", "adventure", "animation", "childrens", "comedy", "crime", "documentary", "drama", "fantasy", "film_noir", "horror", "musical", "mystery", "romance", "sci_fi", "thriller", "war", "western", "item_imdb_length", "item_imdb_mature_rating"))%>%
  left_join(userIndex, by = c("age", "gender", "occupation"))

stuTest <- stuTest %>%
  left_join(itemIndex, by = c("unknown", "action", "adventure", "animation", "childrens", "comedy", "crime", "documentary", "drama", "fantasy", "film_noir", "horror", "musical", "mystery", "romance", "sci_fi", "thriller", "war", "western", "item_imdb_length", "item_imdb_mature_rating"))%>%
  left_join(userIndex, by = c("age", "gender", "occupation"))

## Create Test / Train Split
set.seed(42)
trSize <- floor(0.8*nrow(stuTrain))
trIndex <- sample(seq_len(nrow(stuTrain)), size = trSize)

trSet <- stuTrain[trIndex,]
tstSet <- stuTrain[-trIndex,]


trainControls <- trainControl(
  method = "repeatedcv",
  number = 10,
  repeats = 5,
  allowParallel = TRUE)

gbmFit1 <- train(rating ~ itemIndexMeanRating + userIndexMeanRating, data = trSet, 
                 method = "gbm", 
                 trControl = trainControls,
                 verbose = FALSE)
gbmFit1

tstSet$prediction <- predict(gbmFit1, tstSet)
stuTest$rating <- predict(gbmFit1, stuTest)
ModelMetrics::rmse(tstSet$rating, tstSet$prediction)


results_df <- stuTest[,c(1,6,51)]
results_df$user_item <- paste(results_df$user_id,results_df$item_id,sep = "_")
results_df <- results_df[,c(-1,-2)]
results_df <- results_df %>% mutate(rating = case_when((rating < 1 | is.na(rating)) ~ 1, rating > 5 ~ 5, TRUE ~ rating) ) 

#write.csv(results_df, file = "runRFrun_Submission_gbm_2-20190506.csv", row.names = FALSE)


##### MODEL 6 XGBoost Model ####################################################################


## Allow multiple cores for processing - Run Code for all Models

cluster <- makeCluster(detectCores() - 1)
registerDoParallel(cluster)


stuTrain <- readRDS('AT2_train_STUDENT.rds')
stuTest <- readRDS('AT2_test_STUDENT.rds')
scrape <- readRDS('scrape.rds')

# create user mean rating (train) and add to treain and test data set #
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

#======================== Create submission file========================

results_df_XGB <- stuTest_xgb[,c(1,2,7)]
results_df_XGB$user_item <- paste(results_df_XGB$user_id,results_df_XGB$item_id,sep = "_")
results_df_XGB <- results_df_XGB[,c(-1,-2)]
summary(results_df_XGB)
results_df_XGB <- results_df_XGB %>% mutate(rating = case_when((rating < 1 | is.na(rating)) ~ 1, rating > 5 ~ 5, TRUE ~ rating) ) 

write.csv(results_df_XGB, file = "runRFrun_Submission_XGB-KM_01-20190515.csv", row.names = FALSE)


####TRY XGB###

stuTest$rating <- predict(newGBM, stuTest)