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

str(stuTrain)
summary(stuTrain)

str(stuTest)
summary(stuTest)

stuTrain <- readRDS('AT2_train_STUDENT.rds')
stuTest <- readRDS('AT2_test_STUDENT.rds')
scrape <- readRDS('scrape.rds')

stuTrain$user_id <- as.factor(stuTrain$user_id)
stuTest$user_id <- as.factor(stuTest$user_id)

stuTrain$item_id <- as.factor(stuTrain$item_id)
stuTest$item_id <- as.factor(stuTest$item_id)

# Questions
# How many unique users
# How many unique movies
# How many uniquer genre combinations?

unique(stuTrain$user_id) #943 users
length(unique(stuTrain$item_id)) #1682 items

stuTrain$user_id<-as.character(stuTrain$user_id)

usrCounts <- stuTrain %>%
  group_by(user_id) %>%
  summarise(user_count = n()) %>%
  top_n(250, user_count)

itemCounts <- stuTrain %>%
  group_by(item_id) %>%
  summarise(item_count = n()) %>%
  top_n(500,item_count)

sum(usrCounts$user_count) #49365
sum(itemCounts$item_count) #62611

## imblance in data set as top 25% of users account for > 50% of ratings and
## top 500 items account for 77% of the ratings in the train set.

df <- stuTrain %>%
  group_by(item_id, unknown, action, adventure, animation, childrens, comedy, crime, documentary, drama, fantasy, film_noir, horror,  musical,  mystery, romance,  sci_fi,  thriller, war, western) %>%
  summarise(genre_count = n())

df1 <- melt(df, "item_id", variable.name = "genre")

df1<-df1[df1$value==TRUE,c("item_id","genre")]
df1 <- df1 %>%
  group_by(item_id) %>%
  summarise(genre = paste(genre, collapse=' '))


df$genre <- as.factor(names(df[2:20])[apply(df[2:20],1,match,x=1)])


genreList <- df1 %>%
  group_by(genre) %>%
  summarise(genre_count = n()) %>% ## 256 unique Genre combinations
  top_n(100, genre_count)

#Top 100 genre combinations account for 92% of the items 

