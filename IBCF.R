install.packages('mltools')
library(recommenderlab)
library(reshape2)
library(ggplot2)
library(mltools)
library(hydroGOF)


Tem <- readRDS('./Uni/DAM/AT2/student_data/AT2_train_STUDENT.rds')
Tem_t <- readRDS('./Uni/DAM/AT2/student_data/AT2_test_STUDENT.rds')


#ratings matrix
ratMat_train <- as.data.frame(Tem[,c(1, 6,7)])
ratMat_test <- as.data.frame(Tem_t[,c(1,6)])

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

# recommendation list

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

