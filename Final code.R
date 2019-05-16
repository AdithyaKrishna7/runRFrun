#rm(list = ls())
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

# == Import preprocessed data =================================================

# Webscraping dataa
stuTrain <- readRDS('AT2_train_STUDENT.rds')
stuTest <- readRDS('AT2_test_STUDENT.rds')
scrape <- readRDS('scrape.rds')

str(stuTrain)
summary(stuTrain)


str(stuTest)
summary(stuTest)


str(scrape)
summary(scrape)


stuTrain$user_id <- as.factor(stuTrain$user_id)
stuTest$user_id <- as.factor(stuTest$user_id)

stuTrain$item_id <- as.factor(stuTrain$item_id)
stuTest$item_id <- as.factor(stuTest$item_id)

# Converting ordered factors to unordered
stuTrain$item_imdb_mature_rating <- factor(stuTrain$item_imdb_mature_rating, ordered = F)
stuTrain$age_band <- factor(stuTrain$age_band, ordered = F)

stuTest$item_imdb_mature_rating <- factor(stuTrain$item_imdb_mature_rating, ordered = F)
stuTest$age_band <- factor(stuTrain$age_band, ordered = F)




# == EDA ================================================================
# Questions
# How many unique users
# How many unique movies
# How many uniquer genre combinations?

length(unique(stuTrain$user_id)) #943 users
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

sum(usrCounts$user_count) #49365 (61.64%)
sum(itemCounts$item_count) #62611 (77.76%)

## Imbalance in data set as top 25% of users account for > 50% of ratings and
## Top 500 items account for 77% of the ratings in the train set.

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

stuTrain <- stuTrain %>%
  left_join(df1, by = 'item_id')
stuTest <- stuTest %>%
  left_join(df1, by = 'item_id')

stuTrain$gender <- as.factor(stuTrain$genre)

#Top 100 genre combinations account for 92% of the items 

rm(usrCounts, itemCounts, df, df1, genreList)


# Converting logical values to numeric

# For the train dataset

logVars <- unlist(lapply(stuTrain, is.logical))  
baseLog <- data.frame(stuTrain[ , logVars])
baseNonLog <- data.frame(stuTrain[ , !logVars])

for(i in 1:ncol(baseLog)){
  baseLog[, i] <- as.numeric(as.logical(baseLog[, i]))
}

stuTrain <- cbind(baseNonLog, baseLog)
rm(baseLog, baseNonLog)

str(stuTrain)

# For the test dataset

logVars <- unlist(lapply(stuTest, is.logical))  
baseLog <- data.frame(stuTest[ , logVars])
baseNonLog <- data.frame(stuTest[ , !logVars])

for(i in 1:ncol(baseLog)){
  baseLog[, i] <- as.numeric(as.logical(baseLog[, i]))
}

stuTest <- cbind(baseNonLog, baseLog)
rm(baseLog, baseNonLog)

str(stuTest)

# -- User related ----------------------------------------------------------------
#Create a separate User related dataframe

user <- stuTrain %>% group_by(user_id, age, age_band, gender, zip_code, occupation) %>% summarise(count = n(), user_mean = mean(rating))

#1. User ID 

summary(user$count) #Min 13, max 593, median 52, mean 85

#2. User age 

summary(user$age) #Min 7, max 73, median 31, mean 34
boxplot(user$age)$stats[5,1] #70 

nrow(user %>% filter(age > 70)) #1 user

#3. User gender

summary(user$gender)
#Males 670, Females 273

#4. User ZIP code

zip <- (user %>% group_by(zip_code) %>% summarise(users_in_zip = n()))
#795 unique zips

summary(zip$users_in_zip) #Max 9, 75th perc is also 1
nrow(zip %>% filter(users_in_zip > 1)) #102 zips

#Very little variance - can be potentiall excluded from modelling

#5. Occupation

occ <- user %>% group_by(occupation) %>% summarise(count = n())
#21 different occupations

summary(occ$count) #Min 7 (doctor, homemaker), max 196 (student)

user <- user %>% 
  mutate(occu_class = case_when(occupation == "student" ~ "Student",
                                occupation %in% c("other", "none", "homemaker", "retired") ~ "Other",
                                TRUE ~ "Professional"))

rm(occ, zip)


# -- Movie related ----------------------------------------------------------------
#Create a movie related dataset

nrow(stuTrain %>% distinct(item_id)) #1682

nrow(stuTrain %>% distinct(movie_title)) #1664 - 18 titles have 2 item_ids mapped

names(stuTrain)

movie <- unique(stuTrain[,c(6, 9:12, 32:50, 14, 17:24)])

test <- movie %>% group_by(movie_title) %>% summarise(count = n()) %>% filter(count > 1) %>% arrange(-count) 

movieIssues <- inner_join(test, movie, by = "movie_title")

#Checking other variables in movie dataset
summary(movie)

#One movie with release date blank
View(movie %>% filter(is.na(release_date)))
#Movie title is unknown (id 267), most variables are blank - delete row?
View(stuTrain %>% filter(item_id == 267)) #9 rows - exclude?

#Most engineered IMDB columns have NAs

#Checking if user-movie mapping is distinct
nrow(stuTrain %>% distinct(user_id, item_id)) #80523

#Most watched title
movieCounts <- stuTrain %>% group_by(movie_title) %>% summarise(count = n()) %>% arrange(-count)
#Star Wars most watched

nrow(movieCounts %>% filter(count == 1)) #147 out of 1664 titles watched only by one user

summary(movieCounts$count) #Median 22; most movies are watched by few users only

rm(movieCounts, movieIssues, test)


