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

## Allow multiple cores for processing - Run Code for all Models

cluster <- makeCluster(detectCores() - 1)
registerDoParallel(cluster)

# == Import preprocessed data - Run Code for all Models==================
stuTrain <- readRDS('AT2_train_STUDENT.rds')
stuTest <- readRDS('AT2_test_STUDENT.rds')
scrape <- readRDS('scrape.rds')


#========================Feature Engineering - Run Code for all Models========================#
#
#

tempDf <- stuTrain[, c("user_id", "occupation", "gender", "age", "item_id", "rating", "item_imdb_length", "unknown", "action", "adventure", "animation", "childrens", "comedy", "crime", "documentary","drama", "fantasy", "film_noir", "horror",  "musical",  "mystery", "romance",  "sci_fi",  "thriller", "war", "western")]

#genre.occupation.length.mean.rating <- tempDf %>%
 # group_by(unknown, action, adventure, animation, childrens, comedy, crime, documentary, drama, fantasy, film_noir, horror,  musical,  mystery, romance,  sci_fi,  thriller, war, western, occupation, item_imdb_length) %>%
  #summarise(mean.genre.occ.length.rating = mean(rating))

userIndex <- tempDf %>%
  group_by(age, gender, occupation)%>%
  summarise(userIndexMeanRating = mean(rating))

itemIndex <- tempDf %>%
  group_by(unknown, action, adventure, animation, childrens, comedy, crime, documentary, drama, fantasy, film_noir, horror,  musical,  mystery, romance,  sci_fi,  thriller, war, western, item_imdb_length) %>%
  summarise(itemIndexMeanRating = mean(rating))

stuTrain <- stuTrain %>%
  left_join(itemIndex, by = c("unknown", "action", "adventure", "animation", "childrens", "comedy", "crime", "documentary", "drama", "fantasy", "film_noir", "horror", "musical", "mystery", "romance", "sci_fi", "thriller", "war", "western", "item_imdb_length"))%>%
  left_join(userIndex, by = c("age", "gender", "occupation"))

stuTest <- stuTest %>%
  left_join(itemIndex, by = c("unknown", "action", "adventure", "animation", "childrens", "comedy", "crime", "documentary", "drama", "fantasy", "film_noir", "horror", "musical", "mystery", "romance", "sci_fi", "thriller", "war", "western", "item_imdb_length"))%>%
  left_join(userIndex, by = c("age", "gender", "occupation"))

## Create Test / Train Split
set.seed(42)
trSize <- floor(0.8*nrow(stuTrain))
trIndex <- sample(seq_len(nrow(stuTrain)), size = trSize)

trSet <- stuTrain[trIndex,]
tstSet <- stuTrain[-trIndex,]


##### MODEL 1 Recommendation model (User Based Collaborative Filtering) - submitted to Kaggle on 01/05/2019 #######

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

######################## MODEL 2 Matrix Factoriasation - submitted to Kaggle on 02/05/2019#########################3

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


######################## MODEL 3 GBM - submitted to Kaggle on 06/05/2019 RMSE = 0.97817 ########################

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

######################## MODEL 3a GBM - submitted to Kaggle on 06/05/2019 10.40pm RMSE = 0.96958 , test Train RMSE = 0.9682333########################
# == Import preprocessed data - Run Code for all Models==================
stuTrain <- readRDS('AT2_train_STUDENT.rds')
stuTest <- readRDS('AT2_test_STUDENT.rds')
scrape <- readRDS('scrape.rds')


#
#

tempDf <- stuTrain[, c("user_id", "occupation", "gender", "age", "item_id", "item_imdb_mature_rating", "rating", "item_imdb_length", "unknown", "action", "adventure", "animation", "childrens", "comedy", "crime", "documentary","drama", "fantasy", "film_noir", "horror",  "musical",  "mystery", "romance",  "sci_fi",  "thriller", "war", "western")]

#genre.occupation.length.mean.rating <- tempDf %>%
# group_by(unknown, action, adventure, animation, childrens, comedy, crime, documentary, drama, fantasy, film_noir, horror,  musical,  mystery, romance,  sci_fi,  thriller, war, western, occupation, item_imdb_length) %>%
#summarise(mean.genre.occ.length.rating = mean(rating))

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


#==========================================================================#

######################## MODEL 5 nuralNet -  ########################

nn.grid <- expand.grid(.decay = c(0.5), .size = c(3))
nn.Control <- trainControl(method = 'cv', number = 5, allowParallel = TRUE)

nn.train <- train(rating ~ 
                    age+gender+occupation+
                    unknown+action+adventure+animation+
                    childrens+comedy+crime+documentary+
                    drama+fantasy+film_noir+horror+
                    musical+ mystery+romance+ sci_fi+ 
                    thriller+war+western,
                    data = trSet, method = "nnet",
                    maxit = 1000, tuneGrid = nn.grid, trace = F, linout = 1) 
nn.train

tstSet$predict <- predict(nn.train, newdata = tstSet)
ModelMetrics::rmse(tstSet$rating, tstSet$predict)

######################## MODEL 6 GBM with user mean rating - submitted to Kaggle on 07/05/2019 10.19pm RMSE = 0.94430 , test Train RMSE = 0.9358683 ########################
# == Import preprocessed data - Run Code for all Models==================
stuTrain <- readRDS('AT2_train_STUDENT.rds')
stuTest <- readRDS('AT2_test_STUDENT.rds')
scrape <- readRDS('scrape.rds')

#======================== create user mean rating (train) and add to treain and test data set ========================#

userMeanRating_train <- stuTrain %>%
  group_by(user_id)%>%
  summarise(user_mean_rating = mean(rating))

stuTrain <- stuTrain %>% 
  left_join(userMeanRating_train, by = 'user_id')

stuTest <- stuTest %>% 
  left_join(userMeanRating_train, by = 'user_id')

tempDf <- stuTrain[, c("user_id", "occupation", "gender", "age", "item_id", "user_mean_rating", "item_mean_rating","item_imdb_mature_rating", "rating", "item_imdb_length", "unknown", "action", "adventure", "animation", "childrens", "comedy", "crime", "documentary","drama", "fantasy", "film_noir", "horror",  "musical",  "mystery", "romance",  "sci_fi",  "thriller", "war", "western")]

userIndex <- tempDf %>%
  group_by(age, gender, occupation, user_mean_rating)%>%
  summarise(userIndexMeanRating = mean(rating))

itemIndex <- tempDf %>%
  group_by(unknown, action, adventure, animation, childrens, comedy, crime, documentary, drama, fantasy, film_noir, horror,  musical,  mystery, romance,  sci_fi,  thriller, war, western, item_mean_rating, item_imdb_length, item_imdb_mature_rating) %>%
  summarise(itemIndexMeanRating = mean(rating))

stuTrain <- stuTrain %>%
  left_join(itemIndex, by = c("unknown", "action", "adventure", "animation", "childrens", "comedy", "crime", "documentary", "drama", "fantasy", "film_noir", "horror", "musical", "mystery", "romance", "sci_fi", "thriller", "war", "western", "item_mean_rating", "item_imdb_length", "item_imdb_mature_rating"))%>%
  left_join(userIndex, by = c("age", "gender", "occupation", "user_mean_rating"))

stuTest <- stuTest %>%
  left_join(itemIndex, by = c("unknown", "action", "adventure", "animation", "childrens", "comedy", "crime", "documentary", "drama", "fantasy", "film_noir", "horror", "musical", "mystery", "romance", "sci_fi", "thriller", "war", "western", "item_mean_rating", "item_imdb_length", "item_imdb_mature_rating"))%>%
  left_join(userIndex, by = c("age", "gender", "occupation", "user_mean_rating"))


## Create Test / Train Split
set.seed(42)
trSize <- floor(0.75*nrow(stuTrain))
trIndex <- sample(seq_len(nrow(stuTrain)), size = trSize)

trSet <- stuTrain[trIndex,]
tstSet <- stuTrain[-trIndex,]


trainControls <- trainControl(
  method = "repeatedcv",
  number = 10,
  repeats = 10,
  allowParallel = TRUE)



newGBM <- train(rating ~ itemIndexMeanRating + userIndexMeanRating, data = trSet, 
                 method = "gbm", 
                 trControl = trainControls,
                 bag.fraction=0.40,
                 #n.trees = 200,
                #tuneGrid = tune_grid,
                 verbose = FALSE)
newGBM

tstSet$prediction <- predict(newGBM, tstSet)
stuTest$rating <- predict(newGBM, stuTest)
ModelMetrics::rmse(tstSet$rating, tstSet$prediction)


results_df <- stuTest[,c(1,6,52)]
results_df$user_item <- paste(results_df$user_id,results_df$item_id,sep = "_")
results_df <- results_df[,c(-1,-2)]
results_df <- results_df %>% mutate(rating = case_when((rating < 1 | is.na(rating)) ~ 1, rating > 5 ~ 5, TRUE ~ rating) ) 

#write.csv(results_df, file = "runRFrun_Submission_gbm_3-20190507.csv", row.names = FALSE)

####################### MODEL 7 GBM new engineered variables - submitted to Kaggle on 10/05/2019 12.15am RMSE =  0.94298, test Train RMSE = 0.9297203 ########################
# == Import preprocessed data - Run Code for all Models==================
stuTrain <- readRDS('AT2_train_STUDENT.rds')
stuTest <- readRDS('AT2_test_STUDENT.rds')
scrape <- readRDS('scrape.rds')

#======================== create user mean rating (train) and add to treain and test data set ========================#
stuTrain$user_id <- as.character(stuTrain$user_id)

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
                                      item_imdb_length > 200 ~ "More than 200"))

stuTrain$item_length_band <- as.factor(stuTrain$item_length_band)

stuTest <- stuTest %>% 
  mutate(item_length_band = case_when(item_imdb_length < 50 ~ "Less than 50",
                                      item_imdb_length < 100 ~ "50 to 100",
                                      item_imdb_length < 150 ~ "100 to 150",
                                      item_imdb_length < 200 ~ "150 to 200",
                                      item_imdb_length > 200 ~ "More than 200"))
stuTest$item_length_band <- as.factor(stuTest$item_length_band)

userMeanRating_train <- stuTrain %>%
  group_by(user_id)%>%
  summarise(user_mean_rating = mean(rating))

userGenreRating <- stuTrain %>%
  group_by(age_band, gender, unknown, action, adventure, animation, childrens, comedy, crime, documentary, drama, fantasy, film_noir, horror,  musical,  mystery, romance,  sci_fi,  thriller, war, western)%>%
  summarise(user_genre_mean_rating = mean(rating))

userIndex <- stuTrain %>%
  group_by(occupation,hrStamp) %>%
  summarise(userIndexMeanRating = mean(rating))

itemIndex <- stuTrain %>%
  group_by(item_mean_rating, item_imdb_length, item_imdb_mature_rating) %>%
  summarise(itemIndexMeanRating = mean(rating))


# Join values into stuTrain and stuTest

stuTrain <- stuTrain %>%
  left_join(userMeanRating_train, by = 'user_id') %>%
  left_join(userGenreRating, by = c("age_band", "gender", "unknown", "action", "adventure", "animation", "childrens", "comedy", "crime", "documentary", "drama",    "fantasy", "film_noir", "horror",  "musical",  "mystery", "romance",  "sci_fi",  "thriller", "war", "western"))%>%
  left_join(itemIndex, by = c("item_mean_rating", "item_imdb_length", "item_imdb_mature_rating"))%>%
  left_join(userIndex, by = c("occupation", "hrStamp"))

stuTest <- stuTest %>%
  left_join(userMeanRating_train, by = 'user_id') %>%
  left_join(userGenreRating, by = c("age_band", "gender", "unknown", "action", "adventure", "animation", "childrens", "comedy", "crime", "documentary", "drama",    "fantasy", "film_noir", "horror",  "musical",  "mystery", "romance",  "sci_fi",  "thriller", "war", "western"))%>%
  left_join(itemIndex, by = c("item_mean_rating", "item_imdb_length", "item_imdb_mature_rating"))%>%
  left_join(userIndex, by = c("occupation", "hrStamp"))

summary(stuTrain)
summary(stuTest)

stuTest$user_genre_mean_rating[is.na(stuTest$user_genre_mean_rating)] <- stuTest$item_mean_rating[is.na(stuTest$user_genre_mean_rating)]


## Create Test / Train Split
set.seed(42)
trSize <- floor(0.75*nrow(stuTrain))
trIndex <- sample(seq_len(nrow(stuTrain)), size = trSize)

trSet <- stuTrain[trIndex,]
tstSet <- stuTrain[-trIndex,]


trainControls <- trainControl(
  method = "cv",
  number = 10,
  repeats = 10,
  allowParallel = TRUE)


newGBM <- train(rating ~ user_mean_rating + user_genre_mean_rating + itemIndexMeanRating + userIndexMeanRating, data = trSet, 
                method = "gbm", 
                trControl = trainControls,
                bag.fraction=0.55,
                #n.trees = 200,
                #tuneGrid = tune_grid,
                verbose = TRUE)

newGBM

tstSet$prediction <- predict(newGBM, tstSet)
ModelMetrics::rmse(tstSet$rating, tstSet$prediction)


stuTest$rating <- predict(newGBM, stuTest)


results_df <- stuTest[,c(1,6,56)]
results_df$user_item <- paste(results_df$user_id,results_df$item_id,sep = "_")
results_df <- results_df[,c(-1,-2)]
summary(results_df)
results_df <- results_df %>% mutate(rating = case_when((rating < 1 | is.na(rating)) ~ 1, rating > 5 ~ 5, TRUE ~ rating) ) 

#write.csv(results_df, file = "runRFrun_Submission_gbm_7-20190509.csv", row.names = FALSE)

####################### MODEL 10 GBM new engineered variables - submitted to Kaggle on 11/05/2019 11.15pm RMSE =  0.0.93923, test Train RMSE = 0..9186373 ########################


stuTrain <- readRDS('AT2_train_STUDENT.rds')
stuTest <- readRDS('AT2_test_STUDENT.rds')
scrape <- readRDS('scrape.rds')

#======================== create user mean rating (train) and add to treain and test data set ========================#
stuTrain$user_id <- as.factor(stuTrain$user_id)
stuTest$user_id <- as.factor(stuTest$user_id)

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
                                      item_imdb_length > 200 ~ "More than 200"))

stuTrain$item_length_band <- as.factor(stuTrain$item_length_band)

stuTest <- stuTest %>% 
  mutate(item_length_band = case_when(item_imdb_length < 50 ~ "Less than 50",
                                      item_imdb_length < 100 ~ "50 to 100",
                                      item_imdb_length < 150 ~ "100 to 150",
                                      item_imdb_length < 200 ~ "150 to 200",
                                      item_imdb_length > 200 ~ "More than 200"))
stuTest$item_length_band <- as.factor(stuTest$item_length_band)

userMeanRating_train <- stuTrain %>%
  group_by(user_id)%>%
  summarise(user_mean_rating = mean(rating))

userGenreRating <- stuTrain %>%
  group_by(age_band, gender, action, adventure, animation, childrens, comedy, crime, drama, film_noir, horror,  musical,  mystery, romance,  sci_fi,  thriller, war, western)%>%
  summarise(user_genre_mean_rating = mean(rating))

userIndex <- stuTrain %>%
  group_by(age_band, gender,occupation,hrStamp) %>%
  summarise(userIndexMeanRating = mean(rating))

itemIndex <- stuTrain %>%
  group_by(item_mean_rating, item_imdb_length, item_imdb_mature_rating) %>%
  summarise(itemIndexMeanRating = mean(rating))

# Join values into stuTrain and stuTest

stuTrain <- stuTrain %>%
  left_join(userMeanRating_train, by = 'user_id') %>%
  left_join(userGenreRating, by = c("age_band", "gender", "action", "adventure", "animation", "childrens", "comedy", "crime", "drama", "film_noir", "horror",  "musical",  "mystery", "romance",  "sci_fi",  "thriller", "war", "western"))%>%
  left_join(itemIndex, by = c("item_mean_rating", "item_imdb_length", "item_imdb_mature_rating"))%>%
  left_join(userIndex, by = c("age_band", "gender","occupation", "hrStamp"))

stuTest <- stuTest %>%
  left_join(userMeanRating_train, by = 'user_id') %>%
  left_join(userGenreRating, by = c("age_band", "gender", "action", "adventure", "animation", "childrens", "comedy", "crime", "drama", "film_noir", "horror",  "musical",  "mystery", "romance",  "sci_fi",  "thriller", "war", "western"))%>%
  left_join(itemIndex, by = c("item_mean_rating", "item_imdb_length", "item_imdb_mature_rating"))%>%
  left_join(userIndex, by = c("age_band", "gender","occupation", "hrStamp"))



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
              n.trees = 1000,
              shrinkage = 0.01,
              bag.fraction = 0.45,
              #cv.folds = 10,
              n.cores = 3)
newGBM



tstSet$prediction <- predict(newGBM, tstSet, n.tree = 1000)
ModelMetrics::rmse(tstSet$rating, tstSet$prediction)


stuTest$rating <- predict(newGBM, stuTest, n.tree = 1000)
# ======================== Create submission file ========================
results_df <- stuTest[,c(1,6,56)]
results_df$user_item <- paste(results_df$user_id,results_df$item_id,sep = "_")
results_df <- results_df[,c(-1,-2)]
summary(results_df)
results_df <- results_df %>% mutate(rating = case_when((rating < 1 | is.na(rating)) ~ 1, rating > 5 ~ 5, TRUE ~ rating) ) 

# write.csv(results_df, file = "runRFrun_Submission_gbm_10-20190511.csv", row.names = FALSE)



####################### MODEL 12 GBM new engineered variables - submitted to Kaggle on 11/05/2019 11.39pm RMSE =  0.93785, test Train RMSE = 0.9105955 ########################

stuTrain <- readRDS('AT2_train_STUDENT.rds')
stuTest <- readRDS('AT2_test_STUDENT.rds')
scrape <- readRDS('scrape.rds')

#======================== create user mean rating (train) and add to treain and test data set ========================#
stuTrain$user_id <- as.factor(stuTrain$user_id)
stuTest$user_id <- as.factor(stuTest$user_id)

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
                                      item_imdb_length > 200 ~ "More than 200"))

stuTrain$item_length_band <- as.factor(stuTrain$item_length_band)

stuTest <- stuTest %>% 
  mutate(item_length_band = case_when(item_imdb_length < 50 ~ "Less than 50",
                                      item_imdb_length < 100 ~ "50 to 100",
                                      item_imdb_length < 150 ~ "100 to 150",
                                      item_imdb_length < 200 ~ "150 to 200",
                                      item_imdb_length > 200 ~ "More than 200"))
stuTest$item_length_band <- as.factor(stuTest$item_length_band)

userMeanRating_train <- stuTrain %>%
  group_by(user_id)%>%
  summarise(user_mean_rating = mean(rating))

userGenreRating <- stuTrain %>%
  group_by(age_band, gender, action, adventure, animation, childrens, comedy, crime, drama, film_noir, horror,  musical,  mystery, romance,  sci_fi, thriller, war)%>%
  summarise(user_genre_mean_rating = mean(rating))

userIndex <- stuTrain %>%
  group_by(age, gender,occupation,hrStamp) %>%
  summarise(userIndexMeanRating = mean(rating))

itemIndex <- stuTrain %>%
  group_by(item_mean_rating, item_imdb_length, item_imdb_mature_rating) %>%
  summarise(itemIndexMeanRating = mean(rating))

# Join values into stuTrain and stuTest

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

# stuTest$user_genre_mean_rating[is.na(stuTest$user_genre_mean_rating)] <- stuTest$item_mean_rating[is.na(stuTest$user_genre_mean_rating)]


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
              n.trees = 1000,
              shrinkage = 0.01,
              bag.fraction = 0.45,
              #cv.folds = 10,
              n.cores = 3)
newGBM



tstSet$prediction <- predict(newGBM, tstSet, n.tree = 1000)
ModelMetrics::rmse(tstSet$rating, tstSet$prediction)


stuTest$rating <- predict(newGBM, stuTest, n.tree = 1000)
# ======================== Create submission file ========================
results_df <- stuTest[,c(1,6,56)]
results_df$user_item <- paste(results_df$user_id,results_df$item_id,sep = "_")
results_df <- results_df[,c(-1,-2)]
summary(results_df)
results_df <- results_df %>% mutate(rating = case_when((rating < 1 | is.na(rating)) ~ 1, rating > 5 ~ 5, TRUE ~ rating) ) 

# write.csv(results_df, file = "runRFrun_Submission_gbm_12-20190511.csv", row.names = FALSE)

