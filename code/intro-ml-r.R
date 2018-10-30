# Brandon's RBitmoji id
my_id <- "1551b314-5e8a-4477-aca2-088c05963111-v1"

# List of required (CRAN) packages
pkgs <- c(
  "AmesHousing",   # for Ames housing data
  "caret",         # for data splitting (and other ML-related) functions
  "devtools",      # for installing R packages from
  "earth",         # for MARS algorithm
  "gbm",           # for generalized boosted models**
  "ggplot2",       # for nicer graphics
  "kernlab",       # for email spam data
  "pdp",           # for partial dependence plots**
  "ranger",        # for faster random forest
  "randomForest",  # for classic random forest
  "RBitmoji",      # because we can...
  "RColorBrewer",  # for nicer looking color palettes
  "roundhouse",    # for pure awesomeness
  "rpart",         # for binary recursive partioning (i.e., CART)
  "rsample",       # for data splitting functions
  "tibble",        # for nicer data frames
  "xaringan",      # for building this presentation
  "xgboost"        # for eXtreme Gradient Boosting,
  # "vip"            # for variable importance plots**
)

# ** Developed, authored, and/or maintained by 84.51 employees

# Install required (CRAN) packages
for (pkg in pkgs) {
  if (!(pkg %in% installed.packages()[, "Package"])) {
    install.packages(pkg)
  }
}

# Install required GitHub packages
devtools::install_github("koalaverse/vip")  # use dev version for now
devtools::install_github("thomasp85/gganimate")
devtools::install_github("thomasp85/transformr")

# Load required packages
library(ggplot2)
library(gganimate)
library(mlbench)
library(pdp)
library(tidyr)
library(vip)

# Colors
dark2 <- RColorBrewer::brewer.pal(8, name = "Dark2")

# Set up plotting grid
par(mfrow = c(1, 2))

# Boston housing data
plot(
  formula = cmedv ~ lstat, 
  data = pdp::boston, 
  # asp = 1,
  cex = 1.2,
  pch = 19,
  col = adjustcolor(dark2[1], alpha.f = 0.1),
  main = "Regression",
  xlab = "Lower status of the population (%)",
  ylab = "Median home value"
)
mars <- earth::earth(cmedv ~ lstat, data = pdp::boston)
x <- seq(from = 1.73, max = 37.97, length = 100)
y <- predict(mars, newdata = data.frame(lstat = x))
lines(x, y)

# Simulate data
set.seed(805)
norm2d <- as.data.frame(mlbench::mlbench.2dnormals(
  n = 100,
  cl = 2,
  r = 4,
  sd = 1
))
names(norm2d) <- c("x1", "x2", "y")  # rename columns

# Scatterplot
plot(
  formula = x2 ~ x1, 
  data = norm2d, 
  asp = 1,
  cex = 1.2,
  pch = c(17, 19)[norm2d$y],
  col = adjustcolor(dark2[norm2d$y], alpha.f = 0.5),
  main = "Classification",
  xlab = "Income (standardized)",
  ylab = "Lot size (standardized)"
)
abline(a = 0, b = -1)
legend("topleft", pch = c(17, 19), col = dark2[1:2], cex = 0.5, bty = "n",
       inset = 0.01, title = "Owns a riding mower", legend = c("Yes", "No"))

# Simulate some data 
n <- 100
set.seed(8451)
df <- tibble::tibble(
  x = runif(n, min = -2, max = 2),
  y = rnorm(n, mean = 1 + 2*x + x^2, sd = 1)
)
p <- ggplot(df, aes(x, y)) + 
  geom_point(alpha = 0.5) + 
  theme_light()
p1 <- p + 
  geom_smooth(method = "lm", formula = y ~ x, se = FALSE) +
  ggtitle("Under fitting")
p2 <- p + 
  geom_smooth(method = "lm", formula = y ~ x + I(x^2), se = FALSE) +
  ggtitle("Just right?")
p3 <- p + 
  geom_smooth(method = "loess", span = 0.1, se = FALSE) +
  ggtitle("Over fitting")
gridExtra::grid.arrange(p1, p2, p3, nrow = 1)

# Simulate data
set.seed(8451)
trn <- as.data.frame(mlbench.friedman1(1000, sd = 1))#[, c(1:5, 11)]
tst <- as.data.frame(mlbench.friedman1(1000, sd = 1))#[, c(1:5, 11)]

# Run K-nn simulation
k <- seq(from = 2, to = 30, by = 1)
res <- matrix(NA, nrow = length(k), ncol = 3, 
              dimnames = list(NULL, c("k", "Train", "Test")))
for (i in seq_along(k)) {
  fit <- caret::knnreg(y ~ ., k = k[i], data = trn)
  res[i, "k"] <- k[i]
  res[i, "Train"] <- caret::RMSE(predict(fit, newdata = trn), trn$y)
  res[i, "Test"] <- caret::RMSE(predict(fit, newdata = tst), tst$y)
}
res <- res %>%
  as.data.frame() %>%
  gather(key = "sample", value = "rmse", -k)

# Plot results
best <- min(res[res[, "sample"] == "Test", "rmse"])
ggplot(res, aes(x = 1/k, y = 1/rmse, color = sample)) +
  geom_line(size = 1) +
  geom_hline(yintercept = 1/best, linetype = "dashed") +
  scale_x_log10() +
  scale_y_log10() +
  xlab("Model complexity") +
  ylab("Model performance") +
  theme_light() +
  ggtitle("Bias-variance tradeoff") +
  annotate("text", x = 0.042, y = 0.44, label = "High Bias\nLow Variance", 
           col = "grey30", size = 5) +
  annotate("text", x = 0.39, y = 0.44, label = "Low Bias\nHigh Variance",
           col = "grey30", size = 5) +
  theme(legend.title = element_blank(),
        legend.position = c(0.1, 0.85),
        legend.text = element_text(size = 10),
        axis.text.x = element_blank(),
        axis.ticks.x = element_blank(),
        axis.text.y = element_blank(),
        axis.ticks.y = element_blank())

library(rsample)  # for data splitting

# Make sure that you get the same random numbers
set.seed(123)
data_split <- initial_split(
  pdp::boston, prop = 0.7, strata = "cmedv"
)

boston_train <- training(data_split)  # train
boston_test  <- testing(data_split)  #test

nrow(boston_train)/nrow(pdp::boston)

library(ggplot2)  # for awesome graphics

# Do the response distributions seem to differ?
ggplot(boston_train, aes(x = cmedv)) +
  geom_density(trim = TRUE) +
  geom_density(data = boston_test, trim = TRUE, col = "red")

# Trasformations
p1 <- ggplot(ames, aes(x = Sale_Price, y = ..density..)) +
  geom_histogram(bins = 20, fill = "dodgerblue2", color = "white") +
  # geom_density() +
  scale_x_continuous(name = "Sale price ($K)",
                     labels = function(x) paste0(floor(x/1000))) +
  theme_light() +
  ggtitle("Skew right")
p2 <- ggplot(ames, aes(x = log(Sale_Price), y = ..density..)) +
  geom_histogram(bins = 20, fill = "dodgerblue2", color = "white") +
  # geom_density() +
  theme_light() +
  ggtitle("More symmetric")
grid.arrange(p1, p2, ncol = 1)

# Load the mushroom edibility data
url <- "https://bgreenwell.github.io/MLDay18/data/mushroom.csv"
mushroom <- read.csv(url)  # load the data from GitHub
mushroom$veil.type <- NULL  # only takes on a single value  

# Load required packages
library(caret)  # for data splitting function
library(rpart)  # for binary recursive partitioning

# Partition the data into train/test sets
set.seed(101)
trn_id <- createDataPartition(
  y = mushroom$Edibility, p = 0.5, list = FALSE
)
trn <- mushroom[trn_id, ]   # training data
tst <- mushroom[-trn_id, ]  # test data

# Function to calculate accuracy
accuracy <- function(pred, obs) {
  sum(diag(table(pred, obs))) / length(obs)
}
# Decision stump (test error = 1.53%):
cart1 <- rpart(
  Edibility ~ ., data = trn,
  control = rpart.control(maxdepth = 1) 
)

# Get test set predictions
pred1 <- predict(
  cart1, newdata = tst, 
  type = "class"
)

# Compute test set accuracy
accuracy(
  pred = pred1, 
  obs = tst$Edibility
)
# Optimal tree (test error = 0%):
cart2 <- rpart(
  Edibility ~ ., data = trn, 
  control = list(cp = 0, minbucket = 1, minsplit = 1) 
)

# Get test set predictions
pred2 <- predict(
  cart2, newdata = tst, 
  type = "class"
)

# Compute test set accuracy
accuracy(
  pred = pred2, 
  obs = tst$Edibility
)
# Test set confusion matrices
confusionMatrix(pred1, tst$Edibility)
confusionMatrix(pred2, tst$Edibility)

# Load required packages
library(rpart.plot)

# Tree diagram (deep tree)
prp(cart1,
    type = 4,
    clip.right.labs = FALSE, 
    branch.lwd = 2,
    extra = 1, 
    under = TRUE,
    under.cex = 1.5,
    split.cex = 1.5,
    box.col = c("palegreen3", "pink")[cart1$frame$yval])

# Tree diagram (shallow tree)
prp(cart2,
    type = 4,
    clip.right.labs = FALSE, 
    branch.lwd = 2,
    extra = 1, 
    under = TRUE,
    under.cex = 1.5,
    split.cex = 1.5,
    box.col = c("palegreen3", "pink")[cart2$frame$yval])

# Load the data
data(spam, package = "kernlab") 

# Partition the data into train/test sets
set.seed(101)  # for reproducibility
trn_id <- createDataPartition(spam$type, p = 0.7, list = FALSE)
trn <- spam[trn_id, ]                # training data
tst <- spam[-trn_id, ]               # test data
xtrn <- subset(trn, select = -type)  # training data features
xtst <- subset(tst, select = -type)  # test data features
ytrn <- trn$type                     # training data response

# Fit a classification tree (cp found using k-fold CV)
spam_tree <- rpart(type ~ ., data = trn, cp = 0.001) 
pred <- predict(spam_tree, newdata = xtst, type = "class")

# Compute test set accuracy
(spam_tree_acc <- accuracy(pred = pred, obs = tst$type))

# Tree diagram
prp(spam_tree,
    type = 4,
    clip.right.labs = FALSE, 
    branch.lwd = 2,
    extra = 1, 
    under = TRUE,
    under.cex = 1.5,
    split.cex = 1.5,
    box.col = c("palegreen3", "pink")[spam_tree$frame$yval])

# Variable importance scores
vip::vi(spam_tree) %>%  
  as.data.frame() %>% 
  head(10)

# Variable importance plot
vip::vip(spam_tree, num_features = 10)  

# Simulate some sine wave data
set.seed(1112)  # for reproducibility
x <- seq(from = 0, to = 2 * pi, length = 100)
y <- sin(x) + rnorm(length(x), sd = 0.5)
plot(x, y)
lines(x, sin(x))
legend("topright", legend = "True function", lty = 1, inset = 0.01,
       box.col = "transparent")

# Fit a single regression tree
fit <- rpart(y ~ x, cp = 0)
pred <- predict(fit)
plot(x, y)
lines(x, sin(x))
cols <- RColorBrewer::brewer.pal(9, "Set1")
lines(x, pred, col = cols[1L], lwd = 2)
lgnd <- c("True function", "Single tree")
legend("topright", legend = lgnd, col = c("black", cols[1L]), 
       lty = 1, inset = 0.01, box.col = "transparent")

# Fit many regression trees to bootstrap samples
plot(x, y)
nsim <- 1000
pred_mat <- matrix(nrow = length(x), ncol = nsim)
set.seed(1145)  # for reproducibility
id <- replicate(nsim, sort(sample(length(x), replace = TRUE)))
for (i in 1:nsim) {
  fit <- rpart(y[id[, i]] ~ x[id[, i]], cp = 0)
  pred_mat[, i] <- predict(fit)
  lines(x[id[, i]], pred_mat[, i], 
        col = adjustcolor(cols[2L], alpha.f = 0.05))
}
lines(x, sin(x))
lines(x, pred, col = cols[1L], lwd = 2)
lgnd <- c("True function", "Single tree", "Bootstrapped tree")
legend("topright", legend = lgnd, col = c("black", cols[1L:2L]), 
       lty = 1, inset = 0.01, box.col = "transparent")

# Fit many regression trees to bootstrap samples
plot(x, y)
for (i in 1:nsim) {
  lines(x[id[, i]], pred_mat[, i], 
        col = adjustcolor(cols[2L], alpha.f = 0.05))
}
lines(x, sin(x))
lines(x, pred, col = cols[1L], lwd = 2)
lines(x, apply(pred_mat, MARGIN = 1, FUN = mean), col = cols[6L], lwd = 2)
lgnd <- c("True function", "Single tree", "Bootstrapped tree", "Averaged trees")
legend("topright", legend = lgnd, col = c("black", cols[c(1, 2, 6)]), lty = 1, 
       inset = 0.01, box.col = "transparent")

# Load required packages
library(randomForest)  

# Fit a bagger model
xtst <- subset(tst, select = -type)  
set.seed(1633)  # reproducibility  
spam_bag <- randomForest(
  type ~ ., 
  data = trn, 
  ntree = 250,
  mtry = ncol(xtrn),  
  xtest = xtst,
  ytest = tst$type,
  keep.forest = TRUE
)

# Compute test error
pred <- predict(spam_bag, newdata = xtst, type = "class")
spam_bag_acc <- accuracy(pred = pred, obs = tst$type)

# Tidy test error data
df <- spam_bag$test$err.rate %>%
  as.data.frame() %>%
  tibble::rowid_to_column("num_trees") %>%
  tidyr::gather(type, error, -num_trees) %>%
  dplyr::mutate(
    type = dplyr::case_when(
      type == "nonspam" ~ "Non-spam",
      type == "spam" ~ "Spam",
      TRUE ~ "Overall",
    )
  )

# Animate test error progression
p <- ggplot(df, aes(x = num_trees, y = error, color = type)) +
  geom_path() +
  scale_colour_brewer(palette = "Dark2") +
  geom_hline(yintercept = 1 - spam_tree_acc, linetype = 2, color = "black") +
  geom_hline(yintercept = 1 - spam_bag_acc, linetype = 2, color = dark2[2L]) +
  theme_light() +
  theme(legend.title = element_blank(),
        legend.position = c(0.9, 0.9)) +
  xlab("Number of trees") +
  ylab("Test error") +
  ylim(0.04, 0.11) +
  transition_reveal(id = type, along = num_trees)
animate(p, renderer = gifski_renderer(), device = "png")

# Load required packages
library(randomForest)

# Fit a random forest
xtst <- subset(tst, select = -type)
set.seed(1633)  # reproducibility
spam_rf <- randomForest(
  type ~ ., 
  data = trn, 
  ntree = 250,
  mtry = 7,  # floor(sqrt(p))  
  xtest = xtst,
  ytest = tst$type,
  keep.forest = TRUE
)

# Compute test error
pred <- predict(spam_rf, newdata = xtst, type = "class")
spam_rf_acc <- accuracy(pred = pred, obs = tst$type)

# Tidy test error data
df1 <- spam_bag$test$err.rate %>%
  as.data.frame() %>%
  dplyr::select(error = Test) %>%
  tibble::rowid_to_column("num_trees") %>%
  dplyr::mutate(
    type = "Bagging"
  )
df2 <- spam_rf$test$err.rate %>%
  as.data.frame() %>%
  dplyr::select(error = Test) %>%
  tibble::rowid_to_column("num_trees") %>%
  dplyr::mutate(
    type = "Random forest"
  )
df <- dplyr::bind_rows(df1, df2)

# Animate test error progression
p <- ggplot(df, aes(x = num_trees, y = error, color = type)) +
  geom_path() +
  scale_colour_brewer(palette = "Dark2") +
  geom_hline(yintercept = 1 - spam_tree_acc, linetype = 2, color = "black") +
  geom_hline(yintercept = 1 - spam_bag_acc, linetype = 2, color = dark2[1L]) +
  geom_hline(yintercept = 1 - spam_rf_acc, linetype = 2, color = dark2[2L]) +
  theme_light() +
  theme(legend.title = element_blank(),
        legend.position = c(0.9, 0.9)) +
  xlab("Number of trees") +
  ylab("Test error") +
  ylim(0.04, 0.11) +
  transition_reveal(id = type, along = num_trees)
animate(p, renderer = gifski_renderer(), device = "png")
N <- 100000
set.seed(1537)  # for reproducibility
x <- rnorm(N)
mean(x %in% sample(x, replace = TRUE))  # non-OOB proportion

# Compute test error
pred <- predict(spam_rf, newdata = xtst, type = "class")
spam_rf_acc <- accuracy(pred = pred, obs = tst$type)

# Plot test error
par(mar = c(4, 4, 0.1, 0.1))
plot(seq_len(spam_rf$ntree), spam_rf$test$err.rate[, "Test"], type = "l", 
     col = dark2[4L], ylim = c(0.04, 0.11),
     ylab = "Error estimate", xlab = "Number of trees")
lines(seq_len(spam_rf$ntree), spam_rf$err.rate[, "OOB"], type = "l", 
     col = dark2[1L])
abline(h = spam_rf$err.rate[spam_rf$ntree, "OOB"], lty = 2, col = dark2[1L])
abline(h = 1 - spam_rf_acc, lty = 2, col = dark2[4L])
legend("topright", c("Random forest (OOB)", "Random forest (test)"),
       col = c(dark2[c(1, 4)]), lty = c(1, 1))

# Load required packages
library(ranger)  # a much faster implementation of random forest  

# Load the (corrected) Boston housing data
data(boston, package = "pdp")

# Using the randomForest package
set.seed(2007)  # reproducibility
system.time(
  boston_rf <- randomForest(
    cmedv ~ ., data = boston, 
    ntree = 5000,  
    mtry = 5,  
    importance = FALSE  
  )
)
boston_rf$rsq[boston_rf$ntree]

# Using the ranger package
set.seed(1652)  # reproducibility
system.time(
  boston_ranger <- ranger(
    cmedv ~ ., data = boston, 
    num.trees = 5000,  
    mtry = 5,  # :/  
    importance = "impurity"  
  )
)
boston_ranger$r.squared

# Refit models with less trees
set.seed(1453)  # for reproducibility
boston_rf <- randomForest(cmedv ~ ., data = boston, ntree = 500,
                          importance = TRUE, proximity = TRUE)
boston_ranger <- ranger(cmedv ~ ., data = boston, num.trees = 500,
                        importance = "impurity")

# Construct variable importance plots (the old way)
par(mfrow = c(1, 2))  # side-by-side plots
varImpPlot(boston_rf, main = "")  # randomForest::varImpPlot()  

# Load required packages
library(vip)  # for better (and consistent) variable importance plots

# Construct variable importance plots
p1 <- vip(boston_rf, type = 1) + ggtitle("randomForest")
p2 <- vip(boston_rf, type = 2) + ggtitle("randomForest")
p3 <- vip(boston_ranger) + ggtitle("ranger")
grid.arrange(p1, p2, p3, ncol = 3)  # side-by-side plots 

# Load required packages
library(pdp)

# PDPs for the top two predictors
p1 <- partial(boston_ranger, pred.var = "lstat", plot = TRUE)
p2 <- partial(boston_ranger, pred.var = "rm", plot = TRUE)
p3 <- partial(boston_ranger, pred.var = c("lstat", "rm"),  
              chull = TRUE, plot = TRUE)                   
grid.arrange(p1, p2, p3, ncol = 3)

# 3-D plots  
pd <- attr(p3, "partial.data")  # no need to recalculate 
p1 <- plotPartial(pd, 
  levelplot = FALSE, drape = TRUE, colorkey = FALSE,
  screen = list(z = -20, x = -60)
)

# Using ggplot2  
library(ggplot2)
p2 <- autoplot(pd, palette = "magma")

# ICE and c-ICE curves  
p3 <- boston_ranger %>%  # %>% is automatically imported!
  partial(pred.var = "rm", ice = TRUE, center = TRUE) %>%
  autoplot(alpha = 0.1)

# Display all three plots side by side
grid.arrange(p1, p2, p3, ncol = 3)

# Tree diagram (shallow tree)
prp(cart2,
    type = 4,
    clip.right.labs = FALSE, 
    branch.lwd = 2,
    extra = 1, 
    under = TRUE,
    under.cex = 1.5,
    split.cex = 1.5,
    box.col = c("palegreen3", "pink")[cart2$frame$yval])

# Load the data
data(banknote, package = "alr3")

# Fit a random forest
set.seed(1701)  # for reproducibility
banknote_rf <- randomForest(
  as.factor(Y) ~ ., 
  data = banknote, 
  proximity = TRUE  
)

# Print the OOB confusion matrix
banknote_rf$confusion

# Heatmap of proximity-based distance matrix
heatmap(1 - banknote_rf$proximity, col = viridis::plasma(256))

# Dot chart of proximity-based outlier scores
outlyingness <- tibble::tibble(
  "out" = outlier(banknote_rf),  
  "obs" = seq_along(out),
  "class" = as.factor(banknote$Y)
)
ggplot(outlyingness, aes(x = obs, y = out)) +
  geom_point(aes(color = class, size = out), alpha = 0.5) +
  geom_hline(yintercept = 10, linetype = 2) +  
  labs(x = "Observation", y = "Outlyingness") +
  theme_light() +
  theme(legend.position = "none")

# Multi-dimensional scaling plot of proximity matrix
MDSplot(banknote_rf, fac = as.factor(banknote$Y), k = 2, cex = 1.5)

# Load required packages
library(gbm)

# Fit a GBM to the Boston housing data
set.seed(1053)  # for reproducibility
boston_gbm <- gbm(
  cmedv ~ ., 
  data = boston, 
  var.monotone = NULL,        
  distribution = "gaussian",  # "benoulli", "coxph", etc. 
  n.trees = 10000,            
  interaction.depth = 5,      
  n.minobsinnode = 10,        
  shrinkage = 0.005,          
  bag.fraction = 1,           
  train.fraction = 1,         
  cv.folds = 10  # k-fold CV often gives the best results
)

best_iter <- gbm.perf(  # how many trees?
  boston_gbm, 
  method = "cv"  # or "OOB" or "test" 
)

system.time(
  pd1 <- partial(
    boston_gbm, 
    pred.var = c("lon", "nox"),
    recursive = FALSE,  
    chull = TRUE, 
    n.trees = best_iter  
  )
)

system.time(
  pd2 <- partial(
    boston_gbm, 
    pred.var = c("lon", "nox"),
    recursive = TRUE,  
    chull = TRUE, 
    n.trees = best_iter  
  )
)

# Display plots side by side
grid.arrange(autoplot(pd1), autoplot(pd2), ncol = 2)
ggplot(AmesHousing::make_ames(), aes(x = Sale_Price, y = Overall_Qual)) + 
  ggridges::geom_density_ridges(aes(fill = Overall_Qual)) +  
  scale_x_continuous(labels = scales::dollar) +
  labs(x = "Sale price", y = "Overall quality") +
  theme_light() + theme(legend.position = "none")

# Load required packages
library(xgboost)

# Construct data set
ames <- AmesHousing::make_ames()

# Feature matrix  # or xgb.DMatrix or sparse matrix  
X <- data.matrix(subset(ames, select = -Sale_Price))

# Fit an XGBoost model
set.seed(203)  # for reproducibility
ames_xgb <- xgboost(         # tune using `xgb.cv()`
  data = X, 
  label = ames$Sale_Price, 
  objective = "reg:linear",  # loss function 
  nrounds = 2771,            # number of trees  
  max_depth = 5,             # interaction depth  
  eta = 0.01,                # learning rate  
  subsample = 1,             
  colsample = 1,             
  num_parallel_tree = 1,     
  eval_metric = "rmse",      
  verbose = 0,
  save_period = NULL         
)

# Use k-fold cross-validation to find the "optimal" number of trees
set.seed(1214)  # for reproducibility
ames_xgb_cv <- xgb.cv(
  data = X, 
  label = ames$Sale_Price, 
  objective = "reg:linear",
  nrounds = 10000, 
  max_depth = 5, 
  eta = 0.01, 
  subsample = 1,          
  colsample = 1,          
  num_parallel_tree = 1,  
  eval_metric = "rmse",   
  early_stopping_rounds = 50,
  verbose = 0,
  nfold = 5
)

# Plot cross-validation results
plot(test_rmse_mean ~ iter, data = ames_xgb_cv$evaluation_log, type = "l", 
     ylim = c(0, 200000), xlab = "Number of trees", ylab = "RMSE",
     main = "Results from using xgb.cv()")
lines(train_rmse_mean ~ iter, data = ames_xgb_cv$evaluation_log, col = "red2")
abline(v = ames_xgb_cv$best_iteration, lty = 2)
legend("topright", legend = c("Train", "CV"), lty = 1, col = c("red2", 1),
       inset = 0.15)

# Variable importance plots
p1 <- vip(ames_xgb, feature_names = colnames(X), type = "Gain")
p2 <- vip(ames_xgb, feature_names = colnames(X), type = "Cover")
p3 <- vip(ames_xgb, feature_names = colnames(X), type = "Frequency")
grid.arrange(p1, p2, p3, ncol = 3)

# top 10 features
vip(
  object = ames_xgb, 
  feature_names = colnames(X), 
  type = "Gain",
  num_features = 20, #nrow(X), 
  bar = FALSE
)

# Partial dependence plots
oq_ice <- partial(ames_xgb, pred.var = "Overall_Qual", ice = TRUE, 
                  center = TRUE, train = X)
p4 <- autoplot(partial(ames_xgb, pred.var = "Gr_Liv_Area", train = X))
p5 <- autoplot(partial(ames_xgb, pred.var = "Garage_Cars", train = X))
p6 <- autoplot(oq_ice, alpha = 0.1)
grid.arrange(p4, p5, p6, ncol = 3)

# Partial dependence plots for the top/bottom three features
ames_vi <- vi(ames_xgb, feature_names = colnames(X))
feats <- c(ames_vi$Variable[1:3], ames_vi$Variable[21:23])
pds <- lapply(feats, FUN = function(x) {
  pd <- cbind(x, partial(ames_xgb, pred.var = x, train = X))
  names(pd) <- c("xvar", "xval", "yhat")
  pd
})
pds <- do.call(rbind, pds)
ggplot(pds, aes(x = xval, y = yhat)) +
  geom_line(size = 1.5) +
  geom_hline(yintercept = mean(ames$Sale_Price), linetype = 2, col = "red2") +
  facet_wrap( ~ xvar, scales = "free_x") +
  labs(x = "", y = "Partial dependence") +
  theme_light()
