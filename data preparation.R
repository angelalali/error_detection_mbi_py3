EnsurePackage <- function(packageName)
{
  x <- as.character(packageName)
  if(!require(x, character.only = TRUE))
  {
    install.packages(pkgs = x, 
                     repos = "http://cran.r-project.org")
    require(x, character.only = TRUE)
  }
} 

# EnsurePackage("xlsx")
# EnsurePackage("openxlsx")
EnsurePackage("data.table")
EnsurePackage("sampling")
EnsurePackage('caret')

######## work done on 11/2/2016 #######
# prepares train and test data; results shown on friday for client meeting in champange
#######################################
path = '/Users/yisilala/Documents/IBM/projects/kraft heinz project/data 3'
setwd(path)

data_all = read.csv('10-20 15k.csv', strip.white = TRUE, check.names = FALSE)
### by setting strip.white = TRUE, it'll strip unecessary white spaces for you
### by setting check.names = FALSE, the system won't add that "." to replace spaces in column names
champangne = subset(data_all, data_all$Plant == 8318)
champangne = subset(data_all, data_all$Plant == 8318 & data_all$`Material Type` != 'Z001')

# replace all NAs w empty space
champangne[is.na(champangne)] = ''

# set.seed(4456) 
set.seed(3496)
# train = champangne[sample(champangne$Material, floor(0.8*nrow(champangne))),]
# rand_index = sample.int(nrow(champangne), size = floor(0.8*nrow(champangne)))

# do stratified sampling
# stratified sampling with caret package
to_train = createDataPartition(champangne$`P-S matl status`, times = 1, p = 0.8, list = F)
train = champangne[to_train,]
test = champangne[-to_train, ]

write.csv(train, file = 'train.csv', row.names = FALSE)
write.csv(test, file = 'test.csv', row.names = FALSE)



########## update 11/4/2016 ########
# updated the train to include only 19 fields after meeting the champagne guys
#######################################
path = '/Users/yisilala/Documents/IBM/projects/kraft heinz project/data 3'
setwd(path)
test = read.csv('test.csv', strip.white = T, check.names = F, stringsAsFactors = F)
train = read.csv('train w description vEC (Post Brian).csv', strip.white = T, 
                 check.names = F, stringsAsFactors = F)

str(train)

path = '/Users/yisilala/Documents/IBM/projects/kraft heinz project/data 4'
setwd(path)
colum_des = read.csv('11-4 columns description.csv', strip.white = T, check.names = T)

train_clean = train[, names(train) %in% colum_des$Column]
test_clean = test[, names(test) %in% colum_des$Column]

train_clean$`Safety time ind.`[is.na(train_clean$`Safety time ind.`)] = 'None'
test_clean$`Safety time ind.`[is.na(test_clean$`Safety time ind.`)] = 'None'

train_clean$BatchManagement = ifelse(train_clean$BatchManagement == 'X', 1, 0)
test_clean$BatchManagement = ifelse(test_clean$BatchManagement == 'X', 1, 0)

summary(train_clean)
summary(test_clean)

path = '/Users/yisilala/Documents/IBM/projects/kraft heinz project/data 4/'
setwd(path)
write.csv(train_clean, file = 'train.csv', row.names = FALSE)
write.csv(test_clean, file = 'test.csv', row.names = FALSE)



########## update 11/6/2016 ########
# MRP Type was "ND" and MRP Controller was "ZZZ" AND P-S matl should be "5", but not all of them is so
# so check why?
#######################################

train_count = data.table(item = character(0), count = integer(0))
# str(train_count)

item_name = 'total train records'
count = nrow(train_clean)
train_count = rbind(train_count, as.list(c(item_name, count)))
# train_count[nrow(train_count)+1,] = c(item_name, count)

#### how many meets the requirement listed above?
# scenario 1
count = nrow(train_clean[(train_clean$`MRP Type` == 'ND') & 
                           (train_clean$`MRP Controller` == 'ZZZ') & 
                           (train_clean$`P-S matl status` == 5), ])
item_name = '[MRP Type = ND] & [MRP Controller = ZZZ] & [P-S matl = 5]'
train_count = rbind(train_count, as.list(c(item_name, count)))

#### how many that do NOT meet the requirement above?
# scenario 2
count = nrow(train_clean[train_clean$`MRP Type` != 'ND' 
                         & train_clean$`MRP Controller` != 'ZZZ' 
                         & train_clean$`P-S matl status` != '5', ])
item_name = '[MRP Type != ND] & [MRP Controller != ZZZ] & [P-S matl != 5]'
train_count = rbind(train_count, as.list(c(item_name, count)))

# scenario 3
count = nrow(train_clean[train_clean$`MRP Type` == 'ND' 
                         & train_clean$`MRP Controller` == 'ZZZ' 
                         & train_clean$`P-S matl status` != '5', ])
item_name = '[MRP Type = ND] & [MRP Controller = ZZZ] & [P-S matl != 5]'
train_count = rbind(train_count, as.list(c(item_name, count)))

# scenario 4
count = nrow(train_clean[train_clean$`MRP Type` == 'ND' 
                         & train_clean$`MRP Controller` == 'ZZZ' 
                         & train_clean$`P-S matl status` != '5', ])
item_name = '[MRP Type = ND] & [MRP Controller = ZZZ] & [P-S matl != 5]'
train_count = rbind(train_count, as.list(c(item_name, count)))

# scenario 5
count = nrow(train_clean[train_clean$`MRP Type` == 'ND' 
                         & train_clean$`MRP Controller` != 'ZZZ' 
                         & train_clean$`P-S matl status` == '5', ])
item_name = '[MRP Type = ND] & [MRP Controller != ZZZ] & [P-S matl = 5]'
train_count = rbind(train_count, as.list(c(item_name, count)))

# scenario 6
count = nrow(train_clean[train_clean$`MRP Type` != 'ND' 
                         & train_clean$`MRP Controller` == 'ZZZ' 
                         & train_clean$`P-S matl status` == '5', ])
item_name = '[MRP Type != ND] & [MRP Controller = ZZZ] & [P-S matl = 5]'
train_count = rbind(train_count, as.list(c(item_name, count)))

# scenario 7
count = nrow(train_clean[train_clean$`MRP Type` != 'ND' 
                         & train_clean$`MRP Controller` != 'ZZZ' 
                         & train_clean$`P-S matl status` == '5', ])
item_name = '[MRP Type != ND] & [MRP Controller != ZZZ] & [P-S matl = 5]'
train_count = rbind(train_count, as.list(c(item_name, count)))

# scenario 8
count = nrow(train_clean[train_clean$`MRP Type` != 'ND' 
                         & train_clean$`MRP Controller` == 'ZZZ' 
                         & train_clean$`P-S matl status` != '5', ])
item_name = '[MRP Type != ND] & [MRP Controller = ZZZ] & [P-S matl != 5]'
train_count = rbind(train_count, as.list(c(item_name, count)))



####### let's now study how the test data is ########
test_count = data.table(item = character(0), count = integer(0))
# str(test_count)

item_name = 'total test records'
count = nrow(test_clean)
test_count = rbind(test_count, as.list(c(item_name, count)))
# test_count[nrow(test_count)+1,] = c(item_name, count)

#### how many meets the requirement listed above?
# scenario 1
count = nrow(test_clean[(test_clean$`MRP Type` == 'ND') & 
                          (test_clean$`MRP Controller` == 'ZZZ') & 
                          (test_clean$`P-S matl status` == 5), ])
item_name = '[MRP Type = ND] & [MRP Controller = ZZZ] & [P-S matl = 5]'
test_count = rbind(test_count, as.list(c(item_name, count)))

#### how many that do NOT meet the requirement above?
# scenario 2
count = nrow(test_clean[test_clean$`MRP Type` != 'ND' 
                        & test_clean$`MRP Controller` != 'ZZZ' 
                        & test_clean$`P-S matl status` != '5', ])
item_name = '[MRP Type != ND] & [MRP Controller != ZZZ] & [P-S matl != 5]'
test_count = rbind(test_count, as.list(c(item_name, count)))

# scenario 3
count = nrow(test_clean[test_clean$`MRP Type` == 'ND' 
                        & test_clean$`MRP Controller` != 'ZZZ' 
                        & test_clean$`P-S matl status` != '5', ])
item_name = '[MRP Type = ND] & [MRP Controller != ZZZ] & [P-S matl != 5]'
test_count = rbind(test_count, as.list(c(item_name, count)))

# scenario 4
count = nrow(test_clean[test_clean$`MRP Type` == 'ND' 
                        & test_clean$`MRP Controller` == 'ZZZ' 
                        & test_clean$`P-S matl status` != '5', ])
item_name = '[MRP Type = ND] & [MRP Controller = ZZZ] & [P-S matl != 5]'
test_count = rbind(test_count, as.list(c(item_name, count)))

# scenario 5
count = nrow(test_clean[test_clean$`MRP Type` == 'ND' 
                        & test_clean$`MRP Controller` != 'ZZZ' 
                        & test_clean$`P-S matl status` == '5', ])
item_name = '[MRP Type = ND] & [MRP Controller != ZZZ] & [P-S matl = 5]'
test_count = rbind(test_count, as.list(c(item_name, count)))

# scenario 6
count = nrow(test_clean[test_clean$`MRP Type` != 'ND' 
                        & test_clean$`MRP Controller` == 'ZZZ' 
                        & test_clean$`P-S matl status` == '5', ])
item_name = '[MRP Type != ND] & [MRP Controller = ZZZ] & [P-S matl = 5]'
test_count = rbind(test_count, as.list(c(item_name, count)))

# scenario 7
count = nrow(test_clean[test_clean$`MRP Type` != 'ND' 
                        & test_clean$`MRP Controller` != 'ZZZ' 
                        & test_clean$`P-S matl status` == '5', ])
item_name = '[MRP Type != ND] & [MRP Controller != ZZZ] & [P-S matl = 5]'
test_count = rbind(test_count, as.list(c(item_name, count)))

# scenario 8
count = nrow(test_clean[test_clean$`MRP Type` != 'ND' 
                        & test_clean$`MRP Controller` == 'ZZZ' 
                        & test_clean$`P-S matl status` != '5', ])
item_name = '[MRP Type != ND] & [MRP Controller = ZZZ] & [P-S matl != 5]'
test_count = rbind(test_count, as.list(c(item_name, count)))



######## update on 11/7/2016 #######
# convert some of the data's p-s matl status to 5, and THEN split into train & test
#######################################
path = '/Users/yisilala/Documents/IBM/projects/kraft heinz project/data 3'
setwd(path)

data_all = read.csv('10-20 15k.csv', strip.white = TRUE, check.names = FALSE, stringsAsFactors = F)
### by setting strip.white = TRUE, it'll strip unecessary white spaces for you
### by setting check.names = FALSE, the system won't add that "." to replace spaces in column names
# champangne = subset(data_all, data_all$Plant == 8318)
champangne = subset(data_all, data_all$Plant == 8318 & data_all$`Material Type` != 'Z001')

# replace all NAs w empty space
champangne[is.na(champangne)] = ''

set.seed(3430)
### so basically convert 80% of P-S matl status that meet the requirement MRP Type = ND and MRP Controller = ZZZ into 05
champangne[champangne$Material %in%
                    sample(champangne[champangne$`MRP Type` == 'ND' & champangne$`MRP Controller` == 'ZZZ', 'Material'],
                      size = floor(0.8*nrow(champangne[champangne$`MRP Type` == 'ND' & champangne$`MRP Controller` == 'ZZZ', ])), 
                      replace = F),"P-S matl status"] = '05'

path = '/Users/yisilala/Documents/IBM/projects/kraft heinz project/data 4'
setwd(path)
write.csv(champangne, file = 'Champagne data.csv', row.names = FALSE)







