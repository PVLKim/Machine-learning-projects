import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
from datetime import datetime
startTime = datetime.now()
current_datetime = str(datetime.now())[:16]
import warnings
warnings.filterwarnings('ignore')

from sqlalchemy import create_engine
user_password = 'root:jKn8Po71kLP23'
db_name = 'sobus'
engine = create_engine('mysql+pymysql://' + user_password + '@localhost:3306/' + db_name)

# Choose k
# k = int(input("Choose prediction interval size in days: "))
k = 1

# Import data
df = pd.read_csv('formatted_data.csv', index_col = 0)
print('Shape of the data: {}'.format(df.shape))

# Replace null values with empty strings
df['date_checks'] = df['date_checks'].replace(np.nan, '')
df['prices'] = df['prices'].replace(np.nan, '')
df['updated_prices'] = df['updated_prices'].replace(np.nan, '')
# Retrieve first and last dates of checking the price 
# The difference will give the life of the ticket

# Retrieve today's date
df['last_date'] = df['date_checks'].apply(lambda x: x.split(',')[-1])
today = df['last_date'].max()[:10]
today = datetime.strptime(today, '%Y-%m-%d')
today_date = datetime.strftime(today, '%Y-%m-%d')
df = df.drop(['last_date'], axis=1)

# Turn into strings (otherwise, there these columns stay mixed-type
df['prices'] = df['prices'].astype(str)
df['updated_prices'] = df['updated_prices'].astype(str)

##### BEGIN BUILDING PREDICTION SET ##### 
# The following are almost identical operations as with the original dataset
# This is necessary as the feature values for the 'prediction set' (set from current state of 'tickets_normalized') are slightly different from the training set

# Extract tickets normalized table
con = engine.connect()
query = "SELECT * FROM tickets_normalized"
rs = con.execute(query)
tickets = pd.DataFrame(rs.fetchall())
tickets.columns = rs.keys()
con.close()

# Necessary procedures to define the features for tickets present in 'tickets_normalized' table
df = df.rename(columns={'departure_date': 'departure_time', 'arrival_date': 'arrival_time'})
df['departure_time'] = df['departure_time'] + ':00'
df['arrival_time'] = df['arrival_time'] + ':00'
tickets.drop(['departure_date'], axis=1, inplace=True)
tickets[['departure_time', 'arrival_time']] = tickets[['departure_time', 'arrival_time']].astype(str)
cols = ['departure_id', 'arrival_id', 'company_id', 'conditions', 'departure_time', 'arrival_time']
df = pd.merge(df, tickets, how = 'left', on = cols)
df['departure_time'] = df['departure_time'].str[:-3]
df['arrival_time'] = df['arrival_time'].str[:-3]
df = df.rename(columns={'departure_time': 'departure_date', 'arrival_time': 'arrival_date'})
df_train = df[df.price.notnull()]
df_train.drop(['id', 'price'], axis=1, inplace=True)
df.drop(['id', 'price'], axis=1, inplace=True)
df_train = df_train.rename(columns={'departure_time': 'departure_date', 'arrival_time': 'arrival_date'})

del tickets

today_date = datetime.strftime(today, '%Y-%m-%d')
# Calculate the new 'today' date
df_train.loc[(df_train.last_update.isnull()), 'last_update'] = df_train.loc[(df_train.last_update.isnull()), 'last_update'].replace(np.nan, '')
# Recalculate availability
df_train['available'] = df_train['updated_prices'].apply(lambda x: 0 if x.split(',')[-1] == '0' else 1)

# Last date of ticket
df_train.loc[(df_train.available == 0), 'last_seen'] = df_train.loc[(df_train.available == 0), 'last_check']
df_train.loc[(df_train.available != 0), 'last_seen'] = today_date

# Divide the datetime for each departure and arrival to have dates and times in separate columns
df_train['departure_time'] = df_train['departure_date'].str[-5:]
df_train['arrival_time'] = df_train['arrival_date'].str[-5:]
df_train['departure_date'] = df_train['departure_date'].str[:-6]
df_train['arrival_date'] = df_train['arrival_date'].str[:-6]

# Time for expiration if available
df_train['departure_date'] = pd.to_datetime(df_train['departure_date'], format='%Y-%m-%d %H:%M')
current_date = datetime.strptime(today_date, '%Y-%m-%d')
df_train['time_left'] = (df_train['departure_date'] - current_date).dt.days
df_train.loc[(df_train.time_left < 0), 'time_left'] = 0

# Retrieve the first important feature: 'frequency of updates'
df_train.loc[(df_train.prices != ''), 'num_updates'] = df_train.loc[(df_train.prices != ''), 'prices'].apply(lambda x: len(x.split(',')))
df_train.loc[(df_train.prices == ''), 'num_updates'] = 0
df_train['num_updates'] = df_train['num_updates'].astype(int)
print('Share of tickets that never updated: {} %'.format(round(df_train[df_train['num_updates'] == 0].shape[0] / df_train.shape[0] * 100, 2)))

# Retrieve the second important feature: 'lifetime of the ticket'
df_train['last_seen'] = pd.to_datetime(df_train['last_seen'], format='%Y-%m-%d')
df_train['first_check'] = pd.to_datetime(df_train['first_check'], format='%Y-%m-%d')
df_train['life'] = (df_train['last_seen'] - df_train['first_check']).dt.days

# Retrieve third important feature 'number of days without updates'
df_train['recency'] = 0
df_train['last_check'] = pd.to_datetime(df_train['last_check'], format='%Y-%m-%d')
df_train.loc[(df_train.date_checks == ''), 'recency'] = df_train.loc[(df_train.date_checks == ''), 'life']
c_date = datetime.strptime(today_date, '%Y-%m-%d')
df_train.loc[(df_train.date_checks != ''), 'recency'] = (c_date - df_train.loc[(df_train.date_checks != ''), 'last_check']).dt.days

# Correcting number of updates
df_train.loc[(df_train.date_checks.str[0] == ','), 'num_updates'] = df_train.loc[(df_train.date_checks.str[0] == ','), 'num_updates'] - 1

# Average distance between updates
def compute(x):
    temp = x.split(',')
    temp = temp[::-1]
    sum = 0
    if len(temp) == 2:
        dist = (datetime.strptime(temp[0], '%Y-%m-%d') - datetime.strptime(temp[1], '%Y-%m-%d')).days
        return dist        
    else:
        for i in range(len(temp)-1):
            dist = (datetime.strptime(temp[i], '%Y-%m-%d') - datetime.strptime(temp[i+1], '%Y-%m-%d')).days
            sum += dist
        return sum/(len(temp)-1)
df_train.loc[(df_train.num_updates <= 1), 'aver_dist'] = 0
df_train.loc[(df_train.num_updates > 1), 'aver_dist'] = df_train.loc[(df_train.num_updates > 1), 'date_checks'].apply(compute)

# Expired earlier than departure date
df_train['expired_early'] = 0
df_train.loc[(df_train.available == 0) & (df_train.departure_date > today_date), 'expired_early'] = 1
df_train.loc[~((df_train.available == 0) & (df_train.departure_date > today_date)), 'expired_early'] = df_train.loc[~((df_train.available == 0) & (df_train.departure_date > today_date)), 'expired_early'].replace(np.nan, 0)
df_train['expired_early'] = df_train['expired_early'].astype(int)

#### CHECKPOINT ####
filename = 'prediction_set_full.csv'
df_train.to_csv(filename)
del df_train

##### END OF PREDICTION SET #####

# Label those rows that have updates in the indicated dates
def labelling(x):
    i = 0
    label = 0
    global k
    while i < k: 
        if x == datetime.strftime(today - timedelta(i), '%Y-%m-%d'): 
            label = 1
        i += 1
    return label
df['label'] = df['last_update'].apply(labelling)

# Create the list of dates to be removed
dates_to_remove = [datetime.strftime(today - timedelta(i), '%Y-%m-%d') for i in range(k)]

# Find the index starting from which you should delete array elements (in 'sequence'-columns)
def index_of_minus_k(x):
    temp = x.split(',')
    for index, element in enumerate(dates_to_remove):
        if element in temp:
            return temp.index(element)
    return len(temp)
df['index_minus_k'] = df['date_checks'].apply(index_of_minus_k)

# Remove last k-elements from 'sequence' columns
def remove_last_k(row):
    return ','.join(row['prices'].split(',')[:row['index_minus_k']])
df.loc[df.last_update >= dates_to_remove[-1], 'prices'] = df.loc[df.last_update >= dates_to_remove[-1]].apply(remove_last_k, axis = 1)

def remove_last_k(row):
    return ','.join(row['updated_prices'].split(',')[:row['index_minus_k']])
df.loc[df.last_update >= dates_to_remove[-1], 'updated_prices'] = df.loc[df.last_update >= dates_to_remove[-1]].apply(remove_last_k, axis = 1)

def remove_last_k(row):
    return ','.join(row['date_checks'].split(',')[:row['index_minus_k']])
df.loc[df.last_update >= dates_to_remove[-1], 'date_checks'] = df.loc[df.last_update >= dates_to_remove[-1]].apply(remove_last_k, axis = 1)

# Extracting distribusion company id-s
# con = engine.connect()
# query = "SELECT id FROM companies WHERE distribusion_id IS NOT NULL"
# rs = con.execute(query)
# distr = pd.DataFrame(rs.fetchall())
# distr.columns = rs.keys()
# con.close() 
# distr = set(distr['id'].tolist())

# Create features from company_id
# df['flixbus'] = df['company_id'].apply(lambda x: 1 if x == 2 else 0)
# df['ouibus'] = df['company_id'].apply(lambda x: 1 if x == 3 else 0)
# df['distribusion'] = df['company_id'].apply(lambda x: 1 if x in distr else 0)
# top_95_companies = [2, 24, 56, 20, 3, 7, 65, 18, 9, 16, 5, 70, 14, 6, 8, 53, 13, 28, 30, 62, 79, 95, 54]
# df['top95'] = df['company_id'].apply(lambda x: 1 if x in top_95_companies else 0)

# Calculate the new 'today' date
date_at_minus_k = datetime.strftime(datetime.strptime(dates_to_remove[-1], '%Y-%m-%d') - timedelta(1), '%Y-%m-%d')
df.loc[(df.last_update.isnull()), 'last_update'] = df.loc[(df.last_update.isnull()), 'last_update'].replace(np.nan, '')

# Recalculate availability
df['available'] = df['updated_prices'].apply(lambda x: 0 if x.split(',')[-1] == '0' else 1)

# Last date of ticket
df.loc[(df.available == 0), 'last_seen'] = df.loc[(df.available == 0), 'last_check']
df.loc[(df.available != 0), 'last_seen'] = date_at_minus_k

# Divide the datetime for each departure and arrival to have dates and times in separate columns
df['departure_time'] = df['departure_date'].str[-5:]
df['arrival_time'] = df['arrival_date'].str[-5:]
df['departure_date'] = df['departure_date'].str[:-6]
df['arrival_date'] = df['arrival_date'].str[:-6]

# Time for expiration if available
df['departure_date'] = pd.to_datetime(df['departure_date'], format='%Y-%m-%d %H:%M')
current_date = datetime.strptime(date_at_minus_k, '%Y-%m-%d')
df['time_left'] = (df['departure_date'] - current_date).dt.days
df.loc[(df.time_left < 0), 'time_left'] = 0

# Removing those elements that have their first checks during recent dates
df = df[df.first_check < dates_to_remove[-1]]

# Retrieve the first important feature: 'frequency of updates'
df.loc[(df.prices != ''), 'num_updates'] = df.loc[(df.prices != ''), 'prices'].apply(lambda x: len(x.split(',')))
df.loc[(df.prices == ''), 'num_updates'] = 0
df['num_updates'] = df['num_updates'].astype(int)
print('Share of tickets that never updated: {} %'.format(round(df[df['num_updates'] == 0].shape[0] / df.shape[0] * 100, 2)))

# Retrieve the second important feature: 'lifetime of the ticket'
df['last_seen'] = pd.to_datetime(df['last_seen'], format='%Y-%m-%d')
df['first_check'] = pd.to_datetime(df['first_check'], format='%Y-%m-%d')
df['life'] = (df['last_seen'] - df['first_check']).dt.days

# Retrieve third important feature 'number of days without updates'
df['recency'] = 0
df['last_check'] = pd.to_datetime(df['last_check'], format='%Y-%m-%d')
df.loc[(df.date_checks == ''), 'recency'] = df.loc[(df.date_checks == ''), 'life']
c_date = datetime.strptime(date_at_minus_k, '%Y-%m-%d')
df.loc[(df.date_checks != ''), 'recency'] = (c_date - df.loc[(df.date_checks != ''), 'last_check']).dt.days

# Correcting number of updates
df.loc[(df.date_checks.str[0] == ','), 'num_updates'] = df.loc[(df.date_checks.str[0] == ','), 'num_updates'] - 1

# Average distance between updates
def compute(x):
    temp = x.split(',')
    temp = temp[::-1]
    sum = 0
    if len(temp) == 2:
        dist = (datetime.strptime(temp[0], '%Y-%m-%d') - datetime.strptime(temp[1], '%Y-%m-%d')).days
        return dist        
    else:
        for i in range(len(temp)-1):
            dist = (datetime.strptime(temp[i], '%Y-%m-%d') - datetime.strptime(temp[i+1], '%Y-%m-%d')).days
            sum += dist
        return sum/(len(temp)-1)
df.loc[(df.num_updates <= 1), 'aver_dist'] = 0
df.loc[(df.num_updates > 1), 'aver_dist'] = df.loc[(df.num_updates > 1), 'date_checks'].apply(compute)

# Expired earlier than departure date
df['expired_early'] = 0
df.loc[(df.available == 0) & (df.departure_date > today_date), 'expired_early'] = 1
df.loc[~((df.available == 0) & (df.departure_date > today_date)), 'expired_early'] = df.loc[~((df.available == 0) & (df.departure_date > today_date)), 'expired_early'].replace(np.nan, 0)
df['expired_early'] = df['expired_early'].astype(int)

#### CHECKPOINT ####
filename = 'featured_data.csv'
df.to_csv(filename)
# df[df.available == 0].to_csv(current_datetime + '.csv')
# df = pd.read_csv(filename, index_col = 0)
string = "Time taken to complete: " + str(datetime.now() - startTime)
print(string)