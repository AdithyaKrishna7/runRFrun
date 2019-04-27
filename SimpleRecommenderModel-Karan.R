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

# == Import preprocessed data =================================================
stuTrain <- readRDS('AT2_train_STUDENT.rds')
stuTest <- readRDS('AT2_test_STUDENT.rds')
scrape <- readRDS('scrape.rds')

##### Recommendation model #######

## create simple user / movie / rating matrix

ratMat_train <- as.data.frame(stuTrain[,c(1, 6,7)])
ratMat_test <- as.data.frame(stuTest[,c(1,6)])

train_data <- acast(ratMat_train, ratMat_train$user_id~ratMat_train$item_id)
rating_mat_train <- as.matrix(train_data)

realRatMat <- as(rating_mat_train,"realRatingMatrix")
realRatMat_normalised <- normalize(realRatMat)

hist(getRatings(realRatMat),binwidth=1,col="green",breaks=15,main="Histogram of Ratings",xlab="Ratings")

qplot(getRatings(realRatMat_normalised),binwidth=1,main="Histogram of Normalized Ratings",xlab="Ratings")

iSuggest <- Recommender(realRatMat[1:nrow(realRatMat)],method="UBCF", 
                        param=list(normalize="Z-score", method="Cosine", nn=5, minRating=1))
print(iSuggest)


# predict using iSuggest model to get ratings and top 25
iSuggest_ratings <-predict(iSuggest,realRatMat[1:nrow(realRatMat)],type="ratings")
iSuggest_ratings #943 x 1682 rating matrix of class ‘realRatingMatrix’ with 1505603 ratings.

#iSuggest_top10movies <-predict(iSuggest,realRatMat[1:nrow(realRatMat)],type="topNList",n=25)

# recommendation list

iSuggest_list <- as(iSuggest_ratings, "list")
head(iSuggest_list, 10)


modelRating <- NULL

for(i in 1:length(ratMat_test[,2])){
  userid <- ratMat_test[i,1]
  movieid <- ratMat_test[i,2]
  
  user_n <- as.data.frame(iSuggest_list[[userid]])
  user_n$id<-row.names(user_n)
  
  x = user_n[user_n$id==movieid,1]
  
  if (length(x)==0)
  {
    modelRating[i] <- 0
  }
  else
  {
    modelRating[i] <-x
  }
}