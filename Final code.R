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

#== Import preprocessed data =================================================

#Webscraping dataa
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

#Converting ordered factors to unordered
stuTrain$item_imdb_mature_rating <- factor(stuTrain$item_imdb_mature_rating, ordered = F)
stuTrain$age_band <- factor(stuTrain$age_band, ordered = F)

stuTest$item_imdb_mature_rating <- factor(stuTrain$item_imdb_mature_rating, ordered = F)
stuTest$age_band <- factor(stuTrain$age_band, ordered = F)




#== EDA ================================================================
#Questions
#How many unique users
#How many unique movies
#How many unique genre combinations?

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

##Imbalance in data set as top 25% of users account for > 50% of ratings and
##Top 500 items account for 77% of the ratings in the train set.

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


#Converting logical values to numeric

#For the train dataset

logVars <- unlist(lapply(stuTrain, is.logical))  
baseLog <- data.frame(stuTrain[ , logVars])
baseNonLog <- data.frame(stuTrain[ , !logVars])

for(i in 1:ncol(baseLog)){
  baseLog[, i] <- as.numeric(as.logical(baseLog[, i]))
}

stuTrain <- cbind(baseNonLog, baseLog)
rm(baseLog, baseNonLog)

str(stuTrain)

#For the test dataset

logVars <- unlist(lapply(stuTest, is.logical))  
baseLog <- data.frame(stuTest[ , logVars])
baseNonLog <- data.frame(stuTest[ , !logVars])

for(i in 1:ncol(baseLog)){
  baseLog[, i] <- as.numeric(as.logical(baseLog[, i]))
}

stuTest <- cbind(baseNonLog, baseLog)
rm(baseLog, baseNonLog)

str(stuTest)

#-- User related ----------------------------------------------------------------
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


#-- Movie related ----------------------------------------------------------------
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


#Genres vs ratings

genreRatings <- stuTrain %>% 
                    group_by(unknown, action, adventure, animation, childrens, comedy, crime, documentary, drama, fantasy, film_noir, horror,  musical,  mystery, romance,  sci_fi,  thriller, war, western) %>% 
                    summarise(genre_rating = mean(rating), 
                              votes = n(), 
                              dist_movies = n_distinct(item_id)) %>% 
                    arrange(-genre_rating)        


genreRatings$ratio <- (genreRatings$votes/genreRatings$dist_movies)

summary(genreRatings$ratio)

sum(genreRatings$dist_movies)

movieGenre <- stuTrain %>% distinct(item_id, movie_title, unknown, action, adventure, animation, childrens, comedy, crime, documentary, drama, fantasy, film_noir, horror,  musical,  mystery, romance,  sci_fi,  thriller, war, western)

movieGenre <- movieGenre %>% 
                  melt(id = c("item_id", "movie_title")) %>% 
                     filter(value == 1) %>%
                          select(-value) %>%
                              arrange(item_id)

movieStats <- sqldf("select a.*, b.rating 
                     from movieGenre a
                     left join stuTrain b
                     on a.item_id = b.item_id
                     order by a.item_id")

genreStats <- movieStats %>% 
                  group_by(variable) %>% 
                      summarise(avg_rating = mean(rating),
                                count = n(),
                                dist_movies = n_distinct(item_id)) %>%
                            mutate(rating_ratio = count/dist_movies) %>%
                                arrange(-avg_rating)

rm(genreRatings, genreStats, movieGenre, movieStats)

#== Plots ===================================================================

#Understanding various distributions

ggplot(movie, aes(x = item_mean_rating, y = item_imdb_rating_of_ten)) + geom_point() #Linear relationship as expected

ggplot(stuTrain, aes(x = item_mean_rating, y = item_imdb_length)) + geom_point() #No obvious pattern

ggplot(user, aes(x = age, y = user_mean, color = gender)) + geom_point() #No obvious pattern


ggplot(user, aes(x = age_band, y = user_mean, fill = age_band)) + 
  geom_boxplot(notch = F) + 
  facet_wrap(~occu_class, scales = "free") + 
  scale_fill_brewer(palette = "Set4") +
  theme_bw() +
  xlab("Age band of user") +
  ylab("Mean rating of user") +
  labs(fill = "Age band of user") + 
  ggtitle("Mean user rating across occupation and age bands") +
  theme(plot.title = element_text(hjust = 0.5), legend.position = "none") 


#Viewing behaviour of users based on timestamp

stuTrain$view_hour <- hour(as.character(stuTrain$timestamp))

stuTrain <- stuTrain %>% 
  mutate(view_cat = case_when(view_hour < 6 ~ "Late night (12AM - 6AM)",
                              view_hour < 11 ~ "Morning (6AM - 11AM)",
                              view_hour < 16 ~ "Afternoon (11AM - 4PM)",
                              view_hour < 20 ~ "Evening (4PM - 8PM)",
                              view_hour < 24 ~ "Night (8PM - 12AM)"))

stuTrain$view_cat <- factor(stuTrain$view_cat, ordered = T, levels = c("Morning (6AM - 11AM)", "Afternoon (11AM - 4PM)", "Evening (4PM - 8PM)", "Night (8PM - 12AM)", "Late night (12AM - 6AM)"))

viewCat <- stuTrain %>% group_by(view_cat) %>% summarise(freq = n())

#Bar chart

ggplot(viewCat, aes(x = view_cat, y = freq, fill = freq)) +
  geom_bar(width = 0.5, stat = "identity") +
  theme_bw() +
  scale_fill_gradient(high = "#132B43", low = "#56B1F7") +
  xlab("Time of the day") +
  ylab("Counts of viewings") +
  ggtitle("Movie viewings over time of day") +
  theme(plot.title = element_text(hjust = 0.5), legend.position = "none") 

rm(viewCat)

#== End of EDA ================================================

#==============================================================

#== Modelling =================================================

#########################################################################################
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

#================== GBM MODEL ================

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


#Make predictions for Test Set
tstSet$prediction <- predict(newGBM, tstSet, n.tree = 10000)

#Check for RMSE
ModelMetrics::rmse(tstSet$rating, tstSet$prediction) ## Test/Train RMSE = 0.9071594

#Make prediction for submission test set
stuTest$rating <- predict(newGBM, stuTest, n.tree = 10000)

#======================== Create submission file ========================

#Create results dataframe with final rating, user_id and item_id
results_df <- stuTest[,c(1,6,56)]

#Create variable <user_id>_<item_id> for submission
results_df$user_item <- paste(results_df$user_id,results_df$item_id,sep = "_")

#Drop user_id and item_id
results_df <- results_df[,c(-1,-2)]

#Check summary for missing values
summary(results_df)

#Optimise ratings to 1 or 5 when prediction is lower than 1 or higher than 5
results_df <- results_df %>% mutate(rating = case_when((rating < 1 | is.na(rating)) ~ 1, rating > 5 ~ 5, TRUE ~ rating) ) 

#Write output CSV file for submission

write.csv(results_df, file = "runRFrun_Submission_gbm_15-20190513.csv", row.names = FALSE)

## Kaggle RMSE for above results is 0.93172