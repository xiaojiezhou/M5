########### Set up ########
tmp = ls()
tmp = setdiff(tmp, c("")); tmp
rm(list=c(tmp, 'tmp'))

path = "/Users/xiaojiezhou/Documents/DataScience/Kaggle/M5/"
setwd(paste(path))

source("/Users/xiaojiezhou/Documents/OftenUsed/RCode/FreqUsedRFunc.r")


########### Packages, constants, data, functions¶ ############
# Packages ----
suppressMessages({
  library(tidyverse)
  library(data.table)
  library(RcppRoll)
  if (packageVersion("catboost") < "0.22") { # At the time of writing, only version 0.12 is on Kaggle which e.g. does not support early_stopping_rounds
    devtools::install_url('https://github.com/catboost/catboost/releases/download/v0.23/catboost-R-Darwin-0.23.tgz', INSTALL_opts = c("--no-multiarch"))  
  }
  library(catboost)
})



# Functions ----

free <- function() invisible(gc()) 

d2int <- function(X) {
  X %>% extract(d, into = "d", "([0-9]+)", convert = TRUE)
}

demand_features <- function(X) {
  X %>% 
    group_by(id) %>% 
    mutate(lag_7 = dplyr::lag(demand, 7),
           lag_28 = dplyr::lag(demand, 28),
           roll_lag7_w7 = roll_meanr(lag_7, 7),
           roll_lag7_w28 = roll_meanr(lag_7, 28),
           roll_lag28_w7 = roll_meanr(lag_28, 7),
           roll_lag28_w28 = roll_meanr(lag_28, 28)) %>% 
    ungroup() 
}

# Constants ----
FIRST <- 1914 # Start to predict from this "d"
LENGTH <- 28  # Predict so many days
nrows=Inf

# Data ----
main <- "rawdata"
calendar <- fread(file.path(main, "calendar.csv"), stringsAsFactors = TRUE, 
                  drop = c("date", "weekday", "event_type_1", "event_type_2"))
train <- fread(file.path(main, "sales_train_validation.csv"), stringsAsFactors = TRUE,
               drop = paste0("d_", 1:1000), nrows =nrows)
prices <- fread(file.path(main, "sell_prices.csv"), stringsAsFactors = TRUE)

########### Prepare data¶ ###########
train[, paste0("d_", FIRST:(FIRST + 2 * LENGTH - 1))] <- NA
train <- train %>% 
  mutate(id = gsub("_validation", "", id)) %>% 
  gather("d", "demand", -id, -item_id, -dept_id, -cat_id, -store_id, -state_id) %>% 
  d2int() %>% 
  left_join(calendar %>% d2int(), by = "d") %>% 
  left_join(prices, by = c("store_id", "item_id", "wm_yr_wk")) %>% 
  select(-wm_yr_wk) %>% 
  mutate(demand = as.numeric(demand)) %>% 
  mutate_if(is.factor, as.integer) %>% 
  demand_features() %>% 
  filter(d >= FIRST | !is.na(roll_lag28_w28))



## fwrite(train, file='adata/train_m1.gz')
# fwrite(train, file='adata/train_m1_drop_1000.gz')


########### split the data into test, validation and training ########
## train = fread('adata/train_m1.gz')
# train = fread('adata/train_m1_drop_1000.gz')

# Response and features
y <- "demand"
x <- setdiff(colnames(train), c(y, "d", "id"))

#  test <-   filter(train, d >= FIRST)
test <-   filter(train, d >= FIRST - 56)  #Why?? is this for rolling forecast???


train <- filter(train, d < FIRST)

set.seed(3134)
idx <- sample(nrow(train), trunc(0.1 * nrow(train)))
valid <- catboost.load_pool(train[idx, x], label = train[[y]][idx])
train <- catboost.load_pool(train[-idx, x], label = train[[y]][-idx])
rm(prices, idx, calendar)
free()



########### CatBoost! ###########
# Parameters
params <- list(iterations = 2000,
               metric_period = 5,
               #       task_type = "GPU",
               loss_function = "RMSE",
               eval_metric = "RMSE",
               random_strength = 0.5,
               depth = 7,
               # early_stopping_rounds = 400,
               learning_rate = 0.2,
               l2_leaf_reg = 0.1,
               random_seed = 93)

# Fit
fit <- catboost.train(train, valid, params = params)

# Variable importance plot
catboost.get_feature_importance(fit, valid) %>% 
  as.data.frame() %>% 
  setNames("Importance") %>% 
  rownames_to_column("Feature") %>% 
  ggplot(aes(x = reorder(Feature, Importance), y = Importance)) +
  geom_bar(stat = "identity") +
  coord_flip()

########### Submission ###########
# train = fread('adata/train_m1_drop_1000.gz')
# test <-   filter(train, d >= FIRST - 56)  #Why?? is this for rolling forecast???

#--- rolling forecast ----
for (day in FIRST:(FIRST + LENGTH - 1)) {
  cat(".")
  test[test$d == day, y] <- test %>% 
    filter(between(d, day - 56, day)) %>% 
    demand_features() %>% 
    filter(d == day) %>% 
    select_at(x) %>% 
    catboost.load_pool() %>% 
    catboost.predict(fit, .) * (1.025 + 0.01 * (day > 1928)) # https://www.kaggle.com/kyakovlev/m5-dark-magic
#   catboost.predict(fit, .) 
}
  
# Reshape to submission structure
submission <- test %>% 
  mutate(id = paste(id, ifelse(d < FIRST + LENGTH, "validation", "evaluation"), sep = "_"),
         F = paste0("F", d - FIRST + 1 - LENGTH * (d >= FIRST + LENGTH)),
         demand = ifelse(demand<0, 0, demand)) %>% 
  select(id, F, demand) %>% 
  spread(F, demand, fill = 1) %>% 
  select_at(c("id", paste0("F", 1:LENGTH)))

my_write_csv(submission)

########### ! ###########



############ Delete forecast ################
test_pool =  test %>%   select_at(x) %>% 
  catboost.load_pool()

test$pred = catboost.predict(fit, test_pool)*1.03

#--- reshape to submission structure
rslt = test %>% 
  mutate(id = paste(id, ifelse(d < FIRST + LENGTH, "validation", "evaluation"), sep = "_"),
         F = paste0("F", d - FIRST + 1 - LENGTH * (d >= FIRST + LENGTH)),
         demand=ifelse(d <= FIRST+LENGTH-1, pred, NA),
         demand = ifelse(demand<0, 0, demand)) %>%       
  select(id, F, demand) %>% 
  spread(F, demand, fill = 1) %>% 
  select_at(c("id", paste0("F", 1:LENGTH)))

fname =  paste0(path, 'rawdata/sample_submission.csv'); fname

submission = fread(fname) %>% select(id) %>%
  left_join( rslt, by='id')

my_write_csv(submission)
#my_write_csv_gz(submission)

########### ! ###########

########### ! ###########

########### ! ###########

########### ! ###########
