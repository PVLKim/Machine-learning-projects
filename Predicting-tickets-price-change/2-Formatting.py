import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
from datetime import datetime
startTime = datetime.now()
current_datetime = str(datetime.now())[:16]
import warnings
warnings.filterwarnings('ignore')

# BEFORE THAT YOU SHOULD RUN 'priceChangeProbability.py' file
filename = 'cleaned_data.csv'
SAMPLE_TRUE = False

# Extracting sample from the file
if SAMPLE_TRUE == True:
    n = sum(1 for line in open(filename)) - 1 # number of records in file (excludes header)
    s = n # desired sample size
    skip = sorted(random.sample(range(1, n+1), n-s)) # the 0-indexed header will not be included in the skip list
    df = pd.read_csv(filename, skiprows=skip, index_col = 0)

else:
    df = pd.read_csv(filename, index_col = 0)


# # 1. Initial cleaning and formatting
# Resetting index, as all indexes are mixed now
df.reset_index(inplace = True)

# ### The first price update is irrelevant, as it can't be an update of the price, but rather the error of the API output.
# ### That's why I am going to replace those prices that are different on the first date of check from the updated prices with the first updated price, which should be the true initial price. 


# Retrieve first price and updated price from the sequences into separate columns
df['first_price'] = df['prices'].apply(lambda x: x.split(',')[0])
df['first_update'] =  df['updated_prices'].apply(lambda x: x.split(',')[0])

# Check how many rows require correction
print("Number of rows requiring correction: {}".format(df.loc[(df.first_price != df.first_update)].shape[0]))

# Replace those elements with updated price
def replace_first_element(row):
    temp = row['prices'].split(',')
    temp[0] = row['first_update']
    return ','.join(temp)
df.loc[(df.first_price != df.first_update), 'prices'] = df.loc[(df.first_price != df.first_update)].apply(replace_first_element, axis = 1) 

# Delete unnecessary rows
df = df.drop(columns = ['first_price', 'first_update'])

# # 2. Formatting

# Extract last date for each row to identify the last update date for the current dataset
df['last_date'] = df['date_checks'].apply(lambda x: str(x.split(',')[-1]))
today = df['last_date'].max()[:10]
today = datetime.strptime(today, '%Y-%m-%d')
df = df.drop(['last_date'], axis = 1)

# Retain only dates for sequences of date checks
df['date_checks'] = df['date_checks'].apply(lambda x: ','.join(list(map(lambda s: s[:-9], x.split(',')))))
df['first_check'] = df['date_checks'].apply(lambda x: x.split(',')[0])
# Retrieve last date of checking the price 
df['last_check'] = df['date_checks'].apply(lambda x: x.split(',')[-1])

# Retrieve the index of the sequence where update occurred
# This helps to remove those dates and values in both 'prices' and 'updated_prices' columns that don't have updates
def update_index(row):
    prices = row['prices'].split(',')
    u_prices = row['updated_prices'].split(',')
    index_list = list()
    for i in range(len(prices)):
        if prices[i] != u_prices[i]:
            index_list.append(i)
    return index_list
df['update_index'] = df.apply(update_index, axis = 1)

# Dropping those values from all three sequence columns
def drop_non_updates(row):
    temp = row['date_checks'].split(',')
    new_list = list()
    for index in row['update_index']:
        new_list.append(temp[index])
    return ','.join(new_list)
df['date_checks'] = df.apply(drop_non_updates, axis = 1)

def drop_non_updates(row):
    temp = row['prices'].split(',')
    new_list = list()
    for index in row['update_index']:
        new_list.append(temp[index])
    return ','.join(new_list)
df['prices'] = df.apply(drop_non_updates, axis = 1)

def drop_non_updates(row):
    temp = row['updated_prices'].split(',')
    new_list = list()
    for index in row['update_index']:
        new_list.append(temp[index])
    return ','.join(new_list)
df['updated_prices'] = df.apply(drop_non_updates, axis = 1)

# Extracting last update to label those that are within the last 'k' days
df['last_update'] = df['date_checks'].apply(lambda x: x.split(',')[-1])

#### CHECKPOINT ####
df.to_csv('formatted_data.csv')
# df = pd.read_csv('before_labelling.csv', index_col = 0)
string = "Time taken to complete: " + str(datetime.now() - startTime)
print(string)