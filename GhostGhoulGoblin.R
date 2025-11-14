library(tidyverse)
library(patchwork)
library(ggplot2)
library(tidymodels)
library(vroom)
library(GGally)
library(rpart)
library(glmnet)
library(bonsai)
library(lightgbm)
library(agua)
library(h2o)
library(kknn)
library(discrim)
library(kernlab)
library(naivebayes)
library(themis)
library(beepr)

# Import train and test data
trainData <- vroom("train.csv")
testData <- vroom("test.csv")

# Recipe
# ggg_recipe <- recipe(type ~ bone_length + rotting_flesh + hair_length + has_soul + color,  
                     data = trainData) #%>%

  
# Prediction
predictions <- predict(final_fit, new_data = testData, type = "type")
  