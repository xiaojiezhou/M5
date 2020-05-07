########################### Step0-data.R ##################### 
# File name:  Step0-data.R
# Purpose:  
# Population:  
########### Set up ########
rm(list=ls())
rm(list=c(''))

path = "/Users/x644435/Documents/Private/kaggle/M5"
setwd(paste(path))

source("/Users/x644435/Desktop/OftenUsed/RCode/FreqUsedRFunc.r")


####### Read all csv files from the directory #########
datapath=paste0(path, '/rawdata/')

filenames = dir(datapath)[grepl('.csv$',  dir(datapath))] # ends with csv
rnames = gsub("*.csv$", "", filenames)

print("Following files are to be read into R:")
print(filenames)

for (i in 1:length(filenames)) assign(rnames[i], read.csv(paste0(datapath,filenames[i])))

####### calendar #########
calendar %>% head(.) %>% View(.)
str(calendar)
my_head(calendar)

####### sales_train_validation: 30490x1919: 1919-6 = 1913 days #########
sales_train_validation %>% my_head(., n=5) %>% View(.)
str(sales_train_validation)
my_head(sales_train_validation)

####### sample_submission: 60980x29: 60980=30490*2, F1-F28: 28th day forecasts #########
sample_submission %>% head(.) %>% View(.)
str(sample_submission)
my_head(sample_submission)

####### sell_prices #########
sell_prices %>% head(.) %>% View(.)
str(sell_prices)
my_head(sell_prices)

####### calendar #########
calendar %>% head(.) %>% View(.)
str(calendar)
my_head(calendar)
