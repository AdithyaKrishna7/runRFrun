# == Libraries ===============================================================
# Load packages
library(tidyverse)
library(lubridate)
library(forcats)
library(caret)
library(DataExplorer)
library(corrplot)

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

## Scrape

plot_missing(scrape)
plot_histogram(scrape)

scrapeCor <- cor(scrape[,c(-2,-3,-6)])
corrplot(scrapeCor)

#pretty much all variables are missing values, consider replacing with median values

#Train
plot_missing(stuTrain)
plot_histogram(stuTrain)
plot_correlation(stuTrain)
## in test and train video_release_date is all N/A can be dropped for both test and train
#logical values for category / genre (col 13:31) can be gathered into one factor variable
#some missing values. observations can be dropped or NAs replaced with median
