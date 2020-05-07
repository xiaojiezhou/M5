#---- set up -----

library(data.table)
library(catboost)
library(ggplot2)

set.seed(0)

h <- 28 # forecast horizon
max_lags <- 420 # number of observations to shift by
tr_last <- 1913 # last training day
fday <- as.IDate("2016-04-25") # first day to forecast
nrows <- Inf

nrows <- 2


free <- function() invisible(gc()) 

#---- functions -----
create_dt <- function(is_train = TRUE, nrows = Inf) {
  
  if (is_train) { # create train set
    dt <- fread("/Users/x644435/Documents/Private/kaggle/M5/rawdata/sales_train_validation.csv", nrows = nrows)
    cols <- dt[, names(.SD), .SDcols = patterns("^d_")]
    dt[, (cols) := transpose(lapply(transpose(.SD),
                                    function(x) {
                                      i <- min(which(x > 0))
                                      x[1:i-1] <- NA
                                      x})), .SDcols = cols]
    free()
  } else { # create test set
    dt <- fread("/Users/x644435/Documents/Private/kaggle/M5/rawdata/sales_train_validation.csv", nrows = nrows,
                drop = paste0("d_", 1:(tr_last-max_lags))) # keep only max_lags days from the train set
    dt[, paste0("d_", (tr_last+1):(tr_last+2*h)) := 0] # add empty columns for forecasting
  }
  dt <- na.omit(melt(dt,
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
  dt[, `:=`(d = NULL, # remove useless columns
            wm_yr_wk = NULL)]

  cols <- c("item_id", "store_id", "state_id", "dept_id", "cat_id", "event_name_1") 
  dt[, (cols) := lapply(.SD, function(x) as.integer(factor(x))), .SDcols = cols] # convert character columns to integer
  free()
  
  lag <- c(7, 28) 
  lag_cols <- paste0("lag_", lag) # lag columns names
  
  cols = c("sales")
  dt[, (lag_cols) := shift(.SD, lag), by = id, .SDcols = cols] # add lag vectors
  
  cols = dt[, names(.SD), .SDcols = sapply(dt, is.list)]  
  dt[, (cols) := lapply(.SD, unlist), .SDcols=cols]  #--- unlist colunms
  
  
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
tr <- create_dt(nrows = 3)
free()

tr[, .(sales = unlist(lapply(.SD, sum))), by = "date", .SDcols = "sales"
   ][, ggplot(.SD, aes(x = date, y = sales)) +
       geom_line(size = 0.3, color = "steelblue", alpha = 0.8) + 
       geom_smooth(method='lm', formula= y~x, se = FALSE, linetype = 2, size = 0.5, color = "gray20") + 
       labs(x = "", y = "total sales") +
       theme_minimal() +
       theme(axis.text.x = element_text(angle = 45, hjust = 1), legend.position="none") +
       scale_x_date(labels=scales::date_format ("%b %y"), breaks=scales::date_breaks("3 months"))]
free()

create_fea(tr)
free()

tr <- na.omit(tr) # remove rows with NA to save memory
free()

#rowid = (1:dim(tr)[1])[rowSums(tr)==0]



#------ Create data for training -----



library(catboost)


sales=as.numeric(tr$sales)
tr = (tr[, -c(1,7,8)])
free()

input_pool = catboost.load_pool(tr,  sales, cat_features = c(1:9, 17:21))
free()

#----- train the model -------
trained_model <- catboost.train(train_pool, params = list(iterations = 200))


#------ Forecast --------
te <- create_dt(FALSE, nrows)

for (day in as.list(seq(fday, length.out = 2*h, by = "day"))){
  cat(as.character(day), " ")
  tst <- te[date >= day - max_lags & date <= day]
  create_fea(tst)
  tst <- data.matrix(tst[date == day][, c("id", "sales", "date") := NULL])
  te[date == day, sales := 1.03*predict(trained_model, tst)]
}


te[, .(sales = unlist(lapply(.SD, sum))), by = "date", .SDcols = "sales"
   ][, ggplot(.SD, aes(x = date, y = sales, colour = (date < fday))) +
       geom_line() + 
       geom_smooth(method='lm', formula= y~x, se = FALSE, linetype = 2, size = 0.3, color = "gray20") + 
       labs(x = "", y = "total sales") +
       theme_minimal() +
       theme(axis.text.x = element_text(angle = 45, hjust = 1), legend.position="none") +
       scale_x_date(labels=scales::date_format ("%b %d"), breaks=scales::date_breaks("14 day"))]

te[date >= fday
   ][date >= fday+h, id := sub("validation", "evaluation", id)
     ][, d := paste0("F", 1:28), by = id
       ][, dcast(.SD, id ~ d, value.var = "sales")
         ][, fwrite(.SD, "sub_dt_lgb.csv")]                
