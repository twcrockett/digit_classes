library(tidyverse)  # For data manipulation and visualization
library(tidymodels) # For modeling framework
library(Rdimtools)  # For importing USPS digit data
library(rTensor)    # For tensor operations (HOSVD)
library(randomForest) # For Random Forest
library(e1071)      # For Naive Bayes and SVM
library(glmnet)     # For regularization in SVD approach
library(nnet)       # For multinomial logistic regression (multinom function)

set.seed(48)

data(usps); usps <- tibble(
  digit_id = 1:length(usps$label),
  label = as.factor(usps$label), # Make label a factor for classification
  digit_vector = split(usps$data, row(usps$data))
) %>%
  mutate(
    digit_matrix = map(digit_vector, ~matrix(.x, nrow = 16))
  )

usps_split <- initial_split(usps, prop = 0.8, strata = label)
usps_train <- training(usps_split)
usps_test <- testing(usps_split)
usps_folds <- vfold_cv(usps_train, v = 5, strata = label)

# Helper function to convert to feature matrix
create_feature_matrix <- function(data) {
  matrix(unlist(data$digit_vector), ncol = 256, byrow = TRUE) %>%
    as_tibble() %>%
    bind_cols(label = data$label)
}

# Create feature matrices for train and test sets
train_features <- create_feature_matrix(usps_train)
test_features <- create_feature_matrix(usps_test)


# SVD
