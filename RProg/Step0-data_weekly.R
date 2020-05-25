########################### Step0-data.R ##################### 
# File name:  Step0-data.R
# Purpose:  
# Population:  
##########################
# Purpose:  Create dataset
##########################

########### Set up ########
tmp = ls()
tmp = setdiff(tmp, c("")); tmp
rm(list=c(tmp, 'tmp'))

path = "/Users/xiaojiezhou/Documents/DataScience/Kaggle/M5/"
# path = "/Users/x644435/Documents/Private/kaggle/M5/"
setwd(paste(path))

#source("/Users/x644435/Desktop/OftenUsed/RCode/FreqUsedRFunc.r")
source("/Users/xiaojiezhou/Documents/OftenUsed/RCode/FreqUsedRFunc.r")

source("github.r")
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
  dt[, (cols) := transpose(lapply(transpose(.SD),
                                  function(x) {
                                    i <- min(which(x > 0))
                                    x[1:i-1] <- NA
                                    x})), .SDcols = cols] # remove leading 0s
  
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

daily_agg = function(dat, byvar){
  tmp = dat[,
            .(event_name_1 = median(event_name_1,na.rm=TRUE), 
              snap_CA=max(snap_CA, na.rm=TRUE), snap_TX=max(snap_TX, na.rm=TRUE), snap_WI=max(snap_WI, na.rm=TRUE), 
              sales=sum(sales, na.rm=TRUE), sell_price = mean(sell_price, na.rm=TRUE), 
              lag_7=mean(lag_7, na.rm=TRUE),lag_28=mean(lag_28, na.rm=TRUE),
              rmean_7_7=mean(rmean_7_7, na.rm=TRUE),rmean_7_28=mean(rmean_7_28, na.rm=TRUE),
              rmean_28_7=mean(rmean_28_7, na.rm=TRUE),rmean_28_28=mean(rmean_28_28, na.rm=TRUE),
              N=.N
            ), by=byvar
            ]
  return(tmp)
}


########### Begin: Create dt_m2 ###########
#---- read in data & add features -----
dt <- create_dt()
free()

create_fea(dt)
free()

#---- save dt_m2 -----
fwrite(dt, file='adata/dt_m2_revised.gz')
# dt = fread('adata/dt_m2.gz')
# dt = fread('adata/dt_m2_revised.gz')

########### End: Create dt_m2 ###########

########### Begin: Create aggregated data ############
train = fread('adata/train_m1_drop_1000.gz')
head(dt)
#--- Level 12 (item/store, 30490) daily data ----
dt = as.data.table(dt)
dim(dt)[1]/max(dt$d)  #=> 30490 

#--- Level 11 (item/state, 9147) daily data ----
dt11 = daily_agg(dt,  byvar=c('item_id', 'dept_id', 'cat_id', 'state_id', 
                             'date', 'd', 'wday',  'mday', 'week', 'month', 'year')  )
dim(dt11)[1]/max(dt11$d)  #=> 9147 = 3049 items in 3 states

# fwrite(dt11, file='adata/dt11.gz')

#--- Level 10 (item/all states, 3049) daily data ----
dt10 = daily_agg(dt,  byvar=c('item_id', 'dept_id', 'cat_id', 'date', 
                              'd', 'wday', 'mday', 'week', 'month', 'year')  )
dim(dt10)[1]/(max(dt10$d))  #=> 3049 items


# fwrite(dt10, file='adata/dt10.gz')

#--- Level 9 (dept/store, 70) daily data ----
dt9 = daily_agg(dt,   byvar=c('dept_id', 'cat_id', 'store_id',  'state_id', 
                              'date', 'd', 'wday',  'mday', 'week', 'month', 'year')  )
dim(dt9)[1]/max(dt9$d)  #=> 7 department, 10 stores 


# fwrite(dt9, file='adata/dt9.gz')

#--- Level 8 (category/store, 30) daily data ----
dt8 = daily_agg(dt,   byvar=c('cat_id', 'store_id',  'state_id', 
                              'date', 'd', 'wday',  'mday', 'week', 'month', 'year')  )
dim(dt8)[1]/max(dt8$d)  #=> 3 categories, 10 stores 


# fwrite(dt8, file='adata/dt8.gz')

#--- Level 7 (dept/state, 21) daily data ----
dt7 = daily_agg(dt,   byvar=c('dept_id', 'cat_id', 'state_id', 
                              'date', 'd', 'wday',  'mday', 'week', 'month', 'year')  )
dim(dt7)[1]/max(dt7$d)  #=> 7 dept, 3 states 


# fwrite(dt7, file='adata/dt7.gz')

#--- Level 6 (category/state, 9) daily data ----
dt6 = daily_agg(dt,   byvar=c('cat_id', 'state_id', 
                              'date', 'd', 'wday',  'mday', 'week', 'month', 'year')  )
dim(dt6)[1]/max(dt6$d)  #=> 3 category, 3 states 

# fwrite(dt6, file='adata/dt6.gz')

#--- Level 5 (dept/all states,7) daily data ----
dt5 = daily_agg(dt,   byvar=c('dept_id', 'cat_id',
                              'date', 'd', 'wday',  'mday', 'week', 'month', 'year')  )
dim(dt5)[1]/max(dt5$d)  #=> 7 dept, 1 country

# fwrite(dt5, file='adata/dt5.gz')

#--- Level 4 (cat/all states, 3) daily data ----
dt4 = daily_agg(dt,   byvar=c( 'cat_id',
                               'date', 'd', 'wday',  'mday', 'week', 'month', 'year')  )
dim(dt4)[1]/max(dt4$d)  #=> 3 dept, 1 country

# fwrite(dt4, file='adata/dt4.gz')

#--- Level 3 (all products/store, 10) daily data ----
dt3 = daily_agg(dt,   byvar=c( 'store_id', 'state_id',
                               'date', 'd', 'wday',  'mday', 'week', 'month', 'year')  )
dim(dt3)[1]/max(dt3$d)  #=> 10

# fwrite(dt3, file='adata/dt3.gz')

#--- Level 2 (all products/state, 3) daily data ----
dt2 = daily_agg(dt,   byvar=c( 'state_id',
                               'date', 'd', 'wday',  'mday', 'week', 'month', 'year')  )
dim(dt2)[1]/max(dt2$d)  #=> 3 

# fwrite(dt2, file='adata/dt3.gz')
#--- Level 1 (all products/all states, 1) daily data ----
dt1 = daily_agg(dt,   byvar=c( 
                               'date', 'd', 'wday',  'mday', 'week', 'month', 'year')  )
dim(dt1)[1]/max(dt1$d)  #=> 1 

# fwrite(dt1, file='adata/dt1.gz')




#--- Level weekly: (item/store, 30490) weekly data ----
dt_weekly = dt[, 
               .(event_name_1 = max(event_name_1, na.rm=TRUE), 
                 start_date=min(date, na.rm=TRUE), start_month = min(month, na.rm=TRUE),
                 snap_CA=max(snap_CA, na.rm=TRUE), snap_TX=max(snap_TX, na.rm=TRUE), snap_WI=max(snap_WI, na.rm=TRUE), 
                 sales=mean(sales, na.rm=TRUE)*7, sell_price = mean(sell_price, na.rm=TRUE), 
                 N =.N
               ), by=list(id, item_id, dept_id, cat_id, store_id, state_id, year, week)
               ][,':='(sales_lastwk = lag(sales, 1),
                       sales_1wklastmonth =lag(sales, 4),
                       sales_last4wk = (lag(sales, 1)+lag(sales, 2)+lag(sales, 3)+lag(sales, 4))/4
               )][,avgsales_weekofyr := mean(sales, na.rm=TRUE)
                  , by=list(id, item_id, dept_id, cat_id, store_id, state_id, week)
                  ][,avgsales_monthofyr := mean(sales, na.rm=TRUE)
                    , by=list(id, item_id, dept_id, cat_id, store_id, state_id, start_month)]

dim(dt_weekly)[1]/286  #=> 30490

# fwrite(dt_weekly, file='adata/dt_weekly.gz')

########### End: Create aggregated data ############

   
rm(dt10)
free()
