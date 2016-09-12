library(data.table)
library(plyr)
library(dplyr)
library(tidyr)
library(ggplot2)
library(readr)
library(caret)
library(Matrix)
library(SOAR)
library(Amelia)
library(corrplot)
library(ROCR)
library(gridExtra)
library(randomForest)
example <- twoClassSim(1000, intercept = -10,linearVars = 20)
#splom(~example[, 1:6], groups = example$Class)

dim(example)

table(example$Class)

target<-as.factor(example$Class)
train_matrix <- as.data.frame(model.matrix(Class ~ .-1, data = example, sparse=FALSE))

length(target)
fitControl <- trainControl(method = "repeatedcv",
                           ## k-fold CV...
                           number = 3,
                           ## repeated ten times
                           repeats = 1,
                           ## Estimate class probabilities
                           classProbs = TRUE,
                           ## Evaluate performance using 
                           ## the following function
                           summaryFunction = twoClassSummary)


gbmGrid <-  expand.grid(interaction.depth = c(10,15,25),
                        n.trees = (1:5)*75,
                        shrinkage = c(0.01,0.05,0.1),
                        n.minobsinnode = c(20,25,30))



set.seed(998)
gbmFit <- train(y = target,
                x = train_matrix,
                method = "gbm",
                trControl = fitControl,
                verbose = FALSE,
                tuneGrid = gbmGrid,
                metric = "ROC")
gbmFit
plot(gbmFit)

gbmFit$results

## Random  --------

fitControl.random <- trainControl(method = "repeatedcv",
                           ## k-fold CV...
                           number = 3,
                           ## repeated ten times
                           repeats = 1,
                           ## Estimate class probabilities
                           classProbs = TRUE,
                           ## Evaluate performance using 
                           ## the following function
                           summaryFunction = twoClassSummary,
                           search = "random")



set.seed(998)
gbmFit.random <- train(y = target,
                x = train_matrix,
                method = "gbm",
                trControl = fitControl.random,
                verbose = FALSE,
                tuneLength = 30,
                metric = "ROC")
gbmFit.random
plot(gbmFit.random)

fit_results<-as.data.frame(gbmFit.random$results)
library(mgcv)

m.gam <- gam(ROC ~ s(interaction.depth, k = 4) +
               s(n.trees, k = 4) +
               s(shrinkage) +
               s(n.minobsinnode, k = 4) ,
               data = fit_results)

par(mfrow = c(2, 2))
for (i in 1:4) {
  plot(m.gam, select = i)
}

### ---------- adaptative ....

# http://topepo.github.io/caret/adaptive.html

fitControl.adaptative <- trainControl(method = "adaptive_cv",
                            number = 3,
                            repeats = 1,
                            ## Estimate class probabilities
                            classProbs = TRUE,
                            ## Evaluate performance using 
                            ## the following function
                            summaryFunction = twoClassSummary,
                            ## Adaptive resampling information:
                            adaptive = list(min = 2,
                                            alpha = 0.05,
                                            method = "gls",
                                            complete = TRUE))

set.seed(998)
gbmFit.adaptative <- train(y = target,
                       x = train_matrix,
                       method = "gbm",
                       trControl = fitControl.adaptative,
                       verbose = FALSE,
                       tuneLength = 30,
                       metric = "ROC")
gbmFit.adaptative
plot(gbmFit.adaptative)




## ---------  info --------------  


thresh_code <- getModelInfo("rf", regex = FALSE)[[1]]
# rf model modified with following parameters
#         thresh_code$type
#         thresh_code$parameters
#         thresh_code$grid
#         thresh_code$loop
#         thresh_code$fit
#         thresh_code$predict
#         thresh_code$prob

thresh_code$type <- c("Classification") #...1
## Add the threshold as another tuning parameter
thresh_code$parameters <- data.frame(parameter = c("mtry", "threshold"),  #...2
                                     class = c("numeric", "numeric"),
                                     label = c("#Randomly Selected Predictors",
                                               "Probability Cutoff"))
## The default tuning grid code:
thresh_code$grid <- function(x, y, len = NULL, search = "grid") {  #...3
  p <- ncol(x)
  if(search == "grid") {
    grid <- expand.grid(mtry = seq(1,floor(sqrt(p)),1),
                        threshold = seq(.01, .99, length = len))
  } else {
    grid <- expand.grid(mtry = sample(1:p, size = len),
                        threshold = runif(1, 0, size = len))
  }
  grid
}

## Here we fit a single random forest model (with a fixed mtry)
## and loop over the threshold values to get predictions from the same
## randomForest model.
thresh_code$loop = function(grid) {    #...4
  library(plyr)
  loop <- ddply(grid, c("mtry"),
                function(x) c(threshold = max(x$threshold)))
  submodels <- vector(mode = "list", length = nrow(loop))
  for(i in seq(along = loop$threshold)) {
    index <- which(grid$mtry == loop$mtry[i])
    cuts <- grid[index, "threshold"]
    submodels[[i]] <- data.frame(threshold = cuts[cuts != loop$threshold[i]])
  }
  list(loop = loop, submodels = submodels)
}

## Fit the model independent of the threshold parameter
thresh_code$fit = function(x, y, wts, param, lev, last, classProbs, ...) {  #...5
  if(length(levels(y)) != 2)
    stop("This works only for 2-class problems")
  randomForest(x, y, mtry = param$mtry, ...)
}

## Now get a probability prediction and use different thresholds to
## get the predicted class
thresh_code$predict = function(modelFit, newdata, submodels = NULL) {    #...6
  class1Prob <- predict(modelFit,
                        newdata,
                        type = "prob")[, modelFit$obsLevels[1]]
  ## Raise the threshold for class #1 and a higher level of
  ## evidence is needed to call it class 1 so it should 
  ## decrease sensitivity and increase specificity
  out <- ifelse(class1Prob >= modelFit$tuneValue$threshold,
                modelFit$obsLevels[1],
                modelFit$obsLevels[2])
  if(!is.null(submodels)) {
    tmp2 <- out
    out <- vector(mode = "list", length = length(submodels$threshold))
    out[[1]] <- tmp2
    for(i in seq(along = submodels$threshold)) {
      out[[i+1]] <- ifelse(class1Prob >= submodels$threshold[[i]],
                           modelFit$obsLevels[1],
                           modelFit$obsLevels[2])
    }
  }
  out
}                  
## The probabilities are always the same but we have to create
## multitple versions of the probs to evaluate the data across
## thresholds
thresh_code$prob = function(modelFit, newdata, submodels = NULL) {   #...7
  out <- as.data.frame(predict(modelFit, newdata, type = "prob"))
  if(!is.null(submodels)) {
    probs <- out
    out <- vector(mode = "list", length = length(submodels$threshold)+1)
    out <- lapply(out, function(x) probs)
  }
  out
}



### for summaryFunction in trControl 
fourStats <- function (data, lev = levels(data$obs), model = NULL) {
  ## This code will get use the area under the ROC curve and the
  ## sensitivity and specificity values using the current candidate
  ## value of the probability threshold.
  out <- c(twoClassSummary(data, lev = levels(data$obs), model = NULL))
  
  ## The best possible model has sensitivity of 1 and specificity of 1. 
  ## How far are we from that value?
  coords <- matrix(c(1, 1, out["Spec"], out["Sens"]),
                   ncol = 2,
                   byrow = TRUE)
  colnames(coords) <- c("Spec", "Sens")
  rownames(coords) <- c("Best", "Current")
  c(out, Dist = dist(coords)[1])
}

fitControl <- trainControl(method = "repeatedcv",
                           ## k-fold CV...
                           number = 3,
                           ## repeated ten times
                           repeats = 1,
                           ## Estimate class probabilities
                           classProbs = TRUE,
                           ## Evaluate performance using 
                           ## the following function
                           summaryFunction = fourStats)

data.thresh <-cbind(target,train_matrix)


tune.grid <- expand.grid(mtry= c(1,4,7),threshold= seq(.01, .99, length = 20))
#mtry_grid <- data.frame(mtry = c(1:10))

set.seed(949)
mod1 <- train(target~. , 
              data = data.thresh,
              method = thresh_code,  # modified model -- in lieu of "rf"
              ## Minimize the distance to the perfect model
              metric = "Dist",
              maximize = FALSE,
              tuneGrid=tune.grid,
              #tuneLength = 20,   #  20,
              ntree = 200,          # 1000,
              trControl = fitControl)

mod1
plot(mod1)                        

# ----  XGBoosting -----------              

fiveStats <- function(...) c(twoClassSummary(...), defaultSummary(...))
fourStats <- function (data, lev = levels(data$obs), model = NULL)
{
  
  accKapp <- postResample(data[, "pred"], data[, "obs"])
  out <- c(accKapp,
           sensitivity(data[, "pred"], data[, "obs"], lev[1]),
           specificity(data[, "pred"], data[, "obs"], lev[2]))
  names(out)[3:4] <- c("Sens", "Spec")
  out
}


fitControl <- trainControl(method = "repeatedcv",
                           ## k-fold CV...
                           number = 3,
                           ## repeated ten times
                           repeats = 1,
                           ## Estimate class probabilities
                           classProbs = TRUE,
                           ## Evaluate performance using 
                           ## the following function
                           summaryFunction = fiveStats,
                           search = "random")

xgb.grid <- expand.grid( nrounds = 10,
                         eta = c(0.01, 0.03),
                         max_depth = c(4:6)*2,
                         gamma = c( 0.0, 0.2),
                         subsample = c(0.6, 0.9),
                         colsample_bytree = c( 0.5,0, 0.8),
                         min_child_weight = seq(1,30,10),
                         max_delta_step = seq(1,30,10))


                        
set.seed(45)
xgb_tune <-train(y=target,
                 x=train_matrix,
                 method="xgbTree",
                 trControl=fitControl,
                 #tuneGrid=xgb.grid,
                 tuneLength = 30,
                 verbose=T,
                 metric="Kappa",
                 nthread =3
)
xgb_tune$bestTune
plot(xgb_tune)



fit_results<-as.data.frame(xgb_tune$results)
library(mgcv)

m.gam <- gam(Kappa ~ s(eta, k = 4) +
               s(max_depth, k = 4) +
               s(gamma) +
               s(colsample_bytree, k = 4) +
               s(min_child_weight, k = 4) +
               s(nrounds, k = 4),
             data = fit_results)

par(mfrow = c(2, 3))
for (i in 1:6) {
  plot(m.gam, select = i)
}
