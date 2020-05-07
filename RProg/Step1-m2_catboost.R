##########################
# Purpose:  catboost with revised data using data.table
##########################

########### Set up ########
tmp = ls()
tmp = setdiff(tmp, c("")); tmp
rm(list=c(tmp, 'tmp'))

path = "/Users/x644435/Documents/Private/kaggle/M5/"
setwd(paste(path))

source("/Users/x644435/Desktop/OftenUsed/RCode/FreqUsedRFunc.r")

#---- constants -----

library(data.table)
library(catboost)
library(ggplot2)

set.seed(0)

h <- 28 # forecast horizon
max_lags <- 420 # number of observations to shift by
tr_last <- 1913 # last training day
fday <- as.IDate("2016-04-25") # first day to forecast
nrows <- Inf


#---- functions -----
free <- function() invisible(gc()) 

create_dt <- function(nrows = Inf) {
  
    dt <- fread("/Users/x644435/Documents/Private/kaggle/M5/rawdata/sales_train_validation.csv", nrows = nrows)
    cols <- dt[, names(.SD), .SDcols = patterns("^d_")]
    #dt[, (cols) := transpose(lapply(transpose(.SD),
    #                                function(x) {
    #                                  i <- min(which(x > 0))
    #                                  x[1:i-1] <- NA
    #                                  x})), .SDcols = cols] # remove leading 0s
    
    free()

    dt[, paste0("d_", (tr_last+1):((tr_last+1) + 2 * h - 1))] <- NA  #adding validation & test period
    
    dt <- (melt(dt,
                     measure.vars = patterns("^d_"),
                     variable.name = "d", 
                     na.rm=FALSE,
                     value.name = "sales"))
  
  
  cal <- fread("/Users/x644435/Documents/Private/kaggle/M5/rawdata/calendar.csv")
  dt <- dt[cal, `:=`(date = as.IDate(i.date, format="%Y-%m-%d"), # merge tables by reference
                     wm_yr_wk = i.wm_yr_wk,
                     event_name_1 = i.event_name_1,
                     snap_CA = i.snap_CA,
                     snap_TX = i.snap_TX,
                     snap_WI = i.snap_WI), on = "d"]
  
  prices <- fread("/Users/x644435/Documents/Private/kaggle/M5/rawdata/sell_prices.csv")
  dt[prices, sell_price := i.sell_price, on = c("store_id", "item_id", "wm_yr_wk")] # merge again
}

create_fea <- function(dt) {
  # dt <- create_dt(nrows = 3); dt_view(dt)
  # dt[, `:=`(d = NULL,  wm_yr_wk = NULL)] # remove useless columns
  dt[, `:=`(wm_yr_wk = NULL, d = as.numeric(gsub("d_", "", as.character(d))))] # remove useless columns
  
  cols <- c("item_id", "store_id", "state_id", "dept_id", "cat_id", "event_name_1") 
  dt[, (cols) := lapply(.SD, function(x) as.integer(factor(x))), .SDcols = cols] # convert character columns to integer
  free()
  
  lag <- c(7, 28) 
  lag_cols <- paste0("lag_", lag) # lag columns names
  
  cols = c("sales")
  dt[, (lag_cols) := shift(.SD, lag), by = id, .SDcols = cols] # add lag vectors
  
  # cols = dt[, names(.SD), .SDcols = sapply(dt, is.list)]  
  # dt[, (cols) := lapply(.SD, unlist), .SDcols=cols]  #--- unlist colunms
  
  
  cols = c("sales")
  win <- c(7, 28) # rolling window size
  roll_cols <- paste0("rmean_", t(outer(lag, win, paste, sep="_"))) # rolling features columns names
  dt[, (roll_cols) := frollmean(.SD, win, na.rm = TRUE), by = id, .SDcols = lag_cols] # rolling features on lag_cols
  
  dt[, `:=`(wday = wday(date), # time features
            mday = mday(date),
            week = week(date),
            month = month(date),
            year = year(date))]
}

#---- read in data & add features -----
dt <- create_dt()
free()

create_fea(dt)
free()
fwrite(dt, file='adata/dt_m2.gz')
# dt = fread('adata/dt_m2.gz')


#--- split the data into test, validation and training ----
# Response and features
y <- "sales"
x <- setdiff(colnames(dt), c("sales", "d", "id", "date", "year"))
dt[,max(d)]

#test <- dt[d > tr_last & d<=tr_last+h, ..x]
test <- dt[d > tr_last ]

# training period
sales = dt[d  <= tr_last, sales]
dt <- dt[d <= tr_last,..x]
free()

# sales.test= dt$sales[dt$d > tr_last & dt$d<=tr_last+h]
# 

#train = as.data.frame(train)

set.seed(3134)
cat_features=c(1:9, 17:20)
idx <- sample(nrow(dt), trunc(0.1 * nrow(dt)))
train <- catboost.load_pool(dt[-idx], label = sales[-idx], 
                            cat_features = cat_features - 1)
valid <- catboost.load_pool(dt[idx], label = sales[idx],
                            cat_features = cat_features - 1)

train <- catboost.load_pool(dt[-idx], label = sales[-idx])
valid <- catboost.load_pool(dt[idx], label = sales[idx])
free()

########### CatBoost! ###########
# Parameters
params <- list(iterations = 100,
               metric_period = 50,
               #       task_type = "GPU",
               loss_function = "RMSE",
               eval_metric = "RMSE",
               random_strength = 0.5,
               depth = 7,
               # early_stopping_rounds = 400,
               # learning_rate = 0.1,
               l2_leaf_reg = 0.1,
               random_seed = 93)

# Fit
fit <- catboost.train(train, valid, params = params)
free()

# Variable importance plot
catboost.get_feature_importance(fit, valid) %>% 
  as.data.frame() %>% 
  setNames("Importance") %>% 
  rownames_to_column("Feature") %>% 
  ggplot(aes(x = reorder(Feature, Importance), y = Importance)) +
  geom_bar(stat = "identity") +
  coord_flip()

########### Submission ###########
# Day by day predictions
day = (tr_last+2)
for (day in (tr_last+1):((tr_last+1) + h - 1)){
  cat(".")
  test %>% 
    filter(between(d, day - 56, day)) %>% 
     filter(d == day) %>% 
    select_at(x) %>% 
    catboost.load_pool() %>% 
    catboost.predict(fit, .) * 1.03 %>% View(.) # https://www.kaggle.com/kyakovlev/m5-dark-magic
}

# Reshape to submission structure
submission <- test %>% 
  mutate(id = paste(id, ifelse(d < (tr_last+1) + h, "validation", "evaluation"), sep = "_"),
         F = paste0("F", d - (tr_last+1) + 1 - h * (d >= (tr_last+1) + h))) %>% 
  select(id, F, sales) %>% 
  spread(F, sales, fill = 1) %>% 
  select_at(c("id", paste0("F", 1:h)))

# my_write_csv_gz(submission)


#------ Junk:  Create data for training -----

library(catboost)

sales=as.numeric(tr$sales)
tr = (tr[, -c(1,7,8)])
free()

input_pool = catboost.load_pool(tr,  sales, cat_features = c(1:9, 17:21))
free()

#----- train the model -------
trained_model <- catboost.train(train_pool, params = list(iterations = 200))

