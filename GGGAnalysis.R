library(vroom)
library(dplyr)
library(tidymodels)
library(discrim)
library(embed)
library(keras)
library(reticulate)
library(kernlab)
library(themis)
library(naivebayes)



# Upload Data ------------------------------------------------------------------
train_data <- vroom("train.csv")
test_data <- vroom("test.csv")


# Recipe -----------------------------------------------------------------------

ids <- train_data$id
train_data <- train_data[-1]
train_data$type <- as.factor(train_data$type)

my_recipe <- recipe(type~., data=train_data) %>%
  step_mutate_at(color, fn=factor) %>%
  #step_dummy(all_nominal_predictors()) %>%
  #step_smote(all_outcomes(), neighbors=3)
  step_downsample(all_outcomes())
  #step_normalize(all_numeric_predictors()) %>%
  #step_pca(all_numeric_predictors(), threshold=)







# # apply the recipe to your data1
# prep <- prep(my_recipe)
# baked <- bake(prep, new_data = train_data)

# SVN Poly ---------------------------------------------------------------------

svm_poly_mod <- svm_poly(degree=1, cost=.0131) %>%
  set_mode("classification") %>%
  set_engine("kernlab")

svm_poly_wf <- workflow() %>%
  add_recipe(my_recipe) %>%
  add_model(svm_poly_mod) %>%
  fit(data=train_data)

svm_poly_preds <- predict(svm_poly_wf, new_data=test_data, type="class")

kag_sub <- data.frame(id = test_data$id, type = svm_poly_preds$.pred_class)
vroom_write(x=kag_sub, file="./SvmPoly_down.csv", delim=",")



# SVN Linear -------------------------------------------------------------------

svm_linear_mod <- svm_linear(cost=.0131) %>%
  set_mode("classification") %>%
  set_engine("kernlab")

svm_linear_wf <- workflow() %>%
  add_recipe(my_recipe) %>%
  add_model(svm_linear_mod) %>%
  fit(data=train_data)

svm_linear_preds <- predict(svm_linear_wf, new_data=test_data, type="class")

kag_sub <- data.frame(id = test_data$id, type = svm_linear_preds$.pred_class)
vroom_write(x=kag_sub, file="./SvmLinear_down.csv", delim=",")







# Naive Bayes ------------------------------------------------------------------

# nb_mod <- naive_Bayes(Laplace=tune(), smoothness=tune()) %>%
#   set_mode("classification") %>%
#   set_engine("naivebayes")
# 
# nb_wf <- workflow() %>%
#   add_recipe(my_recipe) %>%
#   add_model(nb_mod)
# 
# # Set up grid of tuning values and K-fold
# tuning_grid <- grid_regular(Laplace(), smoothness(), levels=10)
# folds <- vfold_cv(train_data, v = 5, repeats=1)
# 
# # Find best tuning parameters
# cv_results <- nb_wf %>%
#   tune_grid(resamples=folds, grid=tuning_grid, metrics=metric_set(accuracy))
# best_tune <- cv_results %>%
#   select_best(metric="accuracy")
# 
# # Finalize workflow and predict
# final_wf <- nb_wf %>%
#   finalize_workflow(best_tune) %>%
#   fit(data=train_data)
# 
# # Make Predictions
# nb_preds <- predict(final_wf, new_data=test_data, type="class")
# 
# # Format for Kaggle
# kag_sub <- data.frame(id = test_data$id, type = nb_preds$.pred_class)
# vroom_write(x=kag_sub, file="./NaiveBayes_none.csv", delim=",")



# SVM Poly ---------------------------------------------------------------------

svm_poly_mod <- svm_poly(degree=tune(), cost=tune()) %>%
  set_mode("classification") %>%
  set_engine("kernlab")

svm_poly_wf <- workflow() %>%
  add_recipe(my_recipe) %>%
  add_model(svm_poly_mod)

# Set up grid of tuning values and K-fold
tuning_grid <- grid_regular(degree(), cost(), levels=10)
folds <- vfold_cv(train_data, v = 5, repeats=1)

# Find best tuning parameters
cv_results <- svm_poly_wf %>%
  tune_grid(resamples=folds, grid=tuning_grid, metrics=metric_set(accuracy))
best_tune <- cv_results %>%
  select_best(metric="accuracy")

# Finalize workflow and predict
final_wf <- svm_poly_wf %>%
  finalize_workflow(best_tune) %>%
  fit(data=train_data)

# Make Predictions
nb_preds <- predict(svm_poly_wf, new_data=test_data, type="class")

# Format for Kaggle
kag_sub <- data.frame(id = test_data$id, type = nb_preds$.pred_class)
vroom_write(x=kag_sub, file="./SvmPoly_none.csv", delim=",")


# SVM Linear -------------------------------------------------------------------

svm_linear_mod <- svm_linear(cost=tune()) %>%
  set_mode("classification") %>%
  set_engine("kernlab")

svm_linear_wf <- workflow() %>%
  add_recipe(my_recipe) %>%
  add_model(svm_linear_mod)

tuning_grid <- grid_regular(cost(), levels=10)
folds <- vfold_cv(train_data, v = 5, repeats=1)

cv_results <- svm_linear_wf %>%
  tune_grid(resamples=folds, grid=tuning_grid, metrics=metric_set(accuracy))
best_tune <- cv_results %>%
  select_best(metric="accuracy")

final_wf <- svm_linear_wf %>%
  finalize_workflow(best_tune) %>%
  fit(data=train_data)

# Make Predictions
nb_preds <- predict(final_wf, new_data=test_data, type="class")

# Format for Kaggle
kag_sub <- data.frame(id = test_data$id, type = nb_preds$.pred_class)
vroom_write(x=kag_sub, file="./SvmPoly_none.csv", delim=",")






# Make Predictions
nb_preds <- predict(svm_linear_wf, new_data=test_data, type="class")

# Format for Kaggle
kag_sub <- data.frame(id = test_data$id, type = nb_preds$.pred_class)
vroom_write(x=kag_sub, file="./SvmLinear_smote.csv", delim=",")

# Forest -----------------------------------------------------------------------

# forest_mod <- rand_forest(mtry = tune(), min_n=tune(), trees=tune()) %>%
#   set_engine("ranger") %>%
#   set_mode("classification")
# 
# # Create a workflow with model & recipe
# forest_wf <- workflow() %>%
#   add_recipe(my_recipe) %>%
#   add_model(forest_mod)
# 
# # Set up grid of tuning values and K-fold
# tuning_grid <- grid_regular(mtry(range=c(1,9)), min_n(), trees(), levels=3)
# 
# folds <- vfold_cv(train_data, v = 5, repeats=1)
# 
# # Find best tuning parameters
# cv_results <- forest_wf %>%
#   tune_grid(resamples=folds, grid=tuning_grid, metrics=metric_set(accuracy))
# best_tune <- cv_results %>%
#   select_best(metric="accuracy")
# 
# # Finalize workflow and predict
# final_wf <- forest_wf %>%
#   finalize_workflow(best_tune) %>%
#   fit(data=train_data)
# 
# # Make Predictions
# forest_preds <- predict(final_wf, new_data=test_data, type="class")
# 
# # Format for Kaggle
# kag_sub <- data.frame(id = test_data$id, type = forest_preds$.pred_class)
# vroom_write(x=kag_sub, file="./Forest_up.csv", delim=",")



# Logistic Regression ----------------------------------------------------------

# logreg_mod <- logistic_reg() %>%
#   set_engine("glm")
# 
# # Put into a workflow
# logreg_wf <- workflow() %>%
#   add_recipe(my_recipe) %>%
#   add_model(logreg_mod) %>%
#   fit(data=train_data)
# 
# # Make Predictions
# logreg_preds <- predict(logreg_wf, new_data=test_data, type="class")
# 
# # Format for Kaggle
# kag_sub <- data.frame(id = test_data$id,
#                       ACTION = logreg_preds$.pred_1)
# vroom_write(x=kag_sub, file="./LogregPreds.csv", delim=",")


# Pentalized Logistic Regression -----------------------------------------------

# p_logreg_mod <- logistic_reg(mixture=tune(), penalty=tune()) %>%
#   set_engine("glmnet")
# 
# p_logreg_wf <- workflow() %>%
#   add_recipe(my_recipe) %>%
#   add_model(p_logreg_mod)
# 
# tuning_grid <- grid_regular(penalty(), mixture(), levels = 3)
# folds <- vfold_cv(train_data, v = 5, repeats=1)
# 
# cv_results <- p_logreg_wf %>%
#   tune_grid(resamples=folds, grid=tuning_grid, metrics=metric_set(roc_auc))
# best_tune <- cv_results %>%
#   select_best(metric="roc_auc")
# final_wf <- p_logreg_wf %>%
#   finalize_workflow(best_tune) %>%
#   fit(data=train_data)
# 
# # Make Predictions
# p_logreg_preds <- predict(final_wf, new_data=test_data, type="prob")
# 
# # Format for Kaggle
# kag_sub <- data.frame(id = test_data$id,
#                       ACTION = p_logreg_preds$.pred_1)
# vroom_write(x=kag_sub, file="./PLogregPreds_pcd2.csv", delim=",")


