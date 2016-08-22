library(caret)
library(ggplot2)

set.seed(7210)
train_dat <- SLC14_1(250)
large_dat <- SLC14_1(10000)

rand_ctrl <- trainControl(method = "repeatedcv", repeats = 5,search = "random")
set.seed(308) 

rand_search <- train(y ~ ., data = train_dat,method = "svmRadial",
                          ## Create 20 random parameter values
                                             tuneLength = 20,
                                              metric = "RMSE",
                                              preProc = c("center", "scale"),
                                              trControl = rand_ctrl)
rand_search
plot(rand_search)
ggplot(rand_search) + scale_x_log10() + scale_y_log10()
getTrainPerf(rand_search)


## Define the resampling method
ctrl <- trainControl(method = "repeatedcv", repeats = 5)

## Use this function to optimize the model. The two parameters are 
## evaluated on the log scale given their range and scope. 

svm_fit_bayes <- function(logC, logSigma) {
## Use the same model code but for a single (C, sigma) pair. 
txt <- capture.output(
    mod <- train(y ~ ., data = train_dat,
                 method = "svmRadial",
                 preProc = c("center", "scale"),
                 metric = "RMSE",
                 trControl = ctrl,
                 tuneGrid = data.frame(C = exp(logC), sigma = exp(logSigma)))
    )
## The function wants to _maximize_ the outcome so we return 
## the negative of the resampled RMSE value. `Pred` can be used
## to return predicted values but we'll avoid that and use zero
list(Score = -getTrainPerf(mod)[, "TrainRMSE"], Pred = 0)
}


## Define the bounds of the search. 
lower_bounds <- c(logC = -5, logSigma = -9)
upper_bounds <- c(logC = 20, logSigma = -0.75)
bounds <- list(logC = c(lower_bounds[1], upper_bounds[1]),logSigma = c(lower_bounds[2], upper_bounds[2]))
 
## Create a grid of values as the input into the BO code
initial_grid <- rand_search$results[, c("C", "sigma", "RMSE")]
initial_grid$C <- log(initial_grid$C)
initial_grid$sigma <- log(initial_grid$sigma)
initial_grid$RMSE <- -initial_grid$RMSE
names(initial_grid) <- c("logC", "logSigma", "Value")

## Run the optimization with the initial grid and do
## 30 iterations. We will choose new parameter values
## using the upper confidence bound using 1 std. dev. 
   
library(rBayesianOptimization)
 
set.seed(8606)
ba_search <- BayesianOptimization(svm_fit_bayes,
                                  bounds = bounds,
                                  #init_grid_dt = initial_grid,
                                  init_points = 0,
                                  n_iter = 30,
                                  acq = "ucb", 
                                  kappa = 1,
                                  eps = 0.0,
                                  verbose = TRUE)
 
 set.seed(308)
 final_search <- train(y ~ ., data = train_dat,
                       method = "svmRadial",
                       tuneGrid = data.frame(C = exp(ba_search$Best_Par["logC"]),
                       sigma = exp(ba_search$Best_Par["logSigma"])),
                       metric = "RMSE",
                       preProc = c("center", "scale"),
                       trControl = ctrl)
 
 compare_models(final_search, rand_search)
 
 
 postResample(predict(rand_search, large_dat), large_dat$y)
 
 postResample(predict(final_search, large_dat), large_dat$y)

