#Illustrative Example Optimizing probability thresholds for class imbalances
library(caret)
library(pROC)
library(reshape2)
library(ggplot2)

setwd("~/Documents/ACS/Acxion2")
trainingSet <- dget("trainingSet.csv")
testingSet <- dget("testingSet.csv")
dim(trainingSet)

trainingSet[1:2,c(1,20)] # sample

table(trainingSet$Class)
dim(testingSet)
table(testingSet$Class)

#There is almost a 9:1 imbalance in these data. 
#Let's use a standard random forest model with these data using the default value of mtry. 
#We'll also use repeated 10-fold cross validation to get a sense of performance:
set.seed(949)
mod0 <- train(Class ~ ., data = trainingSet,
              method = "rf",
              metric = "ROC",
              tuneGrid = data.frame(mtry = 3),
              ntree = 1000,
              trControl = trainControl(method = "repeatedcv",
                                       repeats = 5,
                                       classProbs = TRUE,
                                       summaryFunction = twoClassSummary))


getTrainPerf(mod0)

## Get the ROC curve using TestingSet
roc0 <- roc(testingSet$Class,
            predict(mod0, testingSet, type = "prob")[,1],
            levels = rev(levels(testingSet$Class)))
roc0

## Now plot
plot(roc0, print.thres = c(.5), type = "S", 
     print.thres.pattern = "%.3f (Spec = %.2f, Sens = %.2f)", 
     print.thres.cex = .8,
     legacy.axes = TRUE)

# ****The area under the ROC curve is very high, indicating that the model has very good 
# predictive power for these data. 
# The plot shows the default probability cut off value of 50%. 
# The sensitivity and specificity values associated with this point indicate 
# that performance is **not** that good when an actual call needs to be made on a sample.

# One of the most common ways to deal with this is to determine an alternate probability 
# cut off using the ROC curve. But to do this well, another set of data (not the test set) 
# is needed to set the cut off and the test set is used to validate it. We don't have a 
# lot of data this is difficult since we will be spending some of our data just to get 
# a single cut off value.
# Alternatively the model can be tuned, using resampling, to determine any model tuning 
# parameters as well as an appropriate cut off for the probabilities.

## Get the model code for the original random forest method:
# Suppose the model has one tuning parameter and we want to look at four candidate values for tuning. 
# Suppose we also want to tune the probability cut off over 20 different thresholds. 
# Now we have to look at 20×4=80 different models (and that is for each resample). 
# One other feature that has been opened up his ability to use sequential parameters: 
#     these are tuning parameters that don't require a completely new model fit to produce predictions. 
# In this case, we can fit one random forest model and get it's predicted class probabilities and 
# evaluate the candidate probability cutoffs using these same hold-out samples. Here is what the 
# model code looks like:

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
    grid <- expand.grid(mtry = floor(sqrt(p)),
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
## mulitple versions of the probs to evaluate the data across
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
##------------- Modeling with customized RF model ----------

set.seed(949)
mod1 <- train(Class ~ ., data = trainingSet,
              method = thresh_code,  # modified model -- in lieu of "rf"
              ## Minimize the distance to the perfect model
              metric = "Dist",
              maximize = FALSE,
              tuneLength = 10,   #  20,
              ntree = 200,          # 1000,
              trControl = trainControl(method = "cv",  #      "repeatedcv",
                                       #       repeats = 5,
                                       classProbs = TRUE,
                                       summaryFunction = fourStats))

mod1

metrics <- mod1$results[, c(2, 4:6)]
metrics <- melt(metrics, id.vars = "threshold",
                variable.name = "Resampled",
                value.name = "Data")

ggplot(metrics, aes(x = threshold, y = Data, color = Resampled)) +
  geom_line() +
  ylab("") + xlab("Probability Cutoff") +
  theme(legend.position = "top")
