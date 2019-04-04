# coding: utf-8
from datetime import datetime, timedelta
startTime = datetime.now()
current_datetime = str(datetime.now())[:16]
current_date = str(datetime.now())[:10]
import warnings
warnings.filterwarnings('ignore')

# If you run on new machine, change to "YES":
NEW_MACHINE = 'NO'

if NEW_MACHINE == 'YES':
    print("Installing dependencies...")
    import os
    def install_function(package_name):
        os.system('sudo pip3 install ' + package_name)
    install_function('pandas')
    install_function('numpy')
    install_function('sqlalchemy')
    install_function('pymysql')
    install_function('sklearn')
    install_function('imblearn')
    install_function('xgboost')

# Import libraries: 
# 'pandas' for manipulating dataframe, 'numpy' for vector operations, 'sqlalchemy' for interacting with SQL-database
import pandas as pd
import numpy as np
from sqlalchemy import create_engine
user_password = 'root:jKn8Po71kLP23'
db_name = 'sobus'
engine = create_engine('mysql+pymysql://' + user_password + '@localhost:3306/' + db_name)

# Extracting data from database
con = engine.connect()
query = "SET SESSION group_concat_max_len = 100000" # setting parameter to use GROUP_CONCAT function with higher limit of characters per each sequence ('prices', 'updated_prices' and 'date_checks' columns)
rs = con.execute(query)
num_days_past = 60
two_months_before = str(startTime - timedelta(num_days_past))[:10] # finding the date which was 60 days before, in order to limit the data imported from (-60 days to today)
query = "SELECT departure_id, arrival_id, company_id, departure_date, arrival_date, conditions, COUNT(*) AS num_checks, \
         GROUP_CONCAT(date_check ORDER BY id) as date_checks, GROUP_CONCAT(IFNULL(price, 0) ORDER BY id) AS prices, \
         GROUP_CONCAT(IFNULL(updated_price, 0) ORDER BY id) as updated_prices, MIN(available) as available \
         FROM (SELECT id, departure_id, arrival_id, company_id, departure_date, arrival_date, conditions, date_check, price, updated_price, available \
               FROM check_availability_crawling \
               WHERE date_check > \'" + two_months_before + "\'\
               AND date_check < \'" + current_date + "\'\
               UNION ALL \
               SELECT id, departure_id, arrival_id, company_id, departure_date, arrival_date, conditions, date_check, FLOOR(price / passengers) AS price, FLOOR(updated_price / passengers) AS updated_price, available \
               FROM check_availability WHERE departure_date > \'" + two_months_before + "\'\
               AND departure_date < \'" + current_date + "\') AS a \
         GROUP BY departure_id, arrival_id, company_id, departure_date, arrival_date, conditions \
         ORDER BY departure_date DESC;"
rs = con.execute(query)
df = pd.DataFrame(rs.fetchall())
df.columns = rs.keys()
con.close()
print('Number of rows: {}'.format(df.shape[0]))
print("Took {} to extract data.".format(str(datetime.now() - startTime)))
tempTime = datetime.now()

# Extracting last date and removing 'T' to make the date format consistent everywhere
df['departure_date'] = df['departure_date'].apply(lambda x: x if 'T' not in x else x.replace('T', ' '))
df['arrival_date'] = df['arrival_date'].apply(lambda x: x if 'T' not in x else x.replace('T', ' '))
df['date_checks'] = df['date_checks'].apply(lambda x: x if 'T' not in x else x.replace('T', ' '))

# Aggregating data (necessary step as well as further sorting to group tickets, as some of the tickets haven't been grouped in SQL-query due to inherent differences in time formats of tables 'check_availability' and 'check_availability_crawling' (e.g. sometimes it's YYYY-MM-DD HH:MM and in other cases it's YYYY-MM-DDTHH:MM)
def aggregate_strings(x):
    return ','.join(x)
df['available'] = df['available'].astype(str)
df = df.groupby(['departure_id', 'arrival_id', 'company_id', 'departure_date', 'arrival_date', 'conditions'], 
                as_index = False).agg(dict(num_checks = 'sum', date_checks = aggregate_strings, prices = aggregate_strings, 
                                           updated_prices = aggregate_strings, available = lambda x: ','.join(x)))
print("Took {} to aggregate data.".format(str(datetime.now() - tempTime)))
tempTime = datetime.now()

# Sorting all 'sequence'-columns in the right order (order of date checks)
def sorting_prices(row):
    sort_by_list = row['date_checks'].split(',')
    target = row['prices'].split(',')
    target = [x for _,x in sorted(zip(sort_by_list, target), key=lambda pair: pair[0])]
    return ','.join(target)
def sorting_updated_prices(row):
    sort_by_list = row['date_checks'].split(',')
    target = row['updated_prices'].split(',')
    target = [x for _,x in sorted(zip(sort_by_list,target), key=lambda pair: pair[0])]
    return ','.join(target)
def sorting_dates(x):
    temp = x.split(',')
    return ','.join(sorted(temp))
df.loc[(df['available'] != '1') & (df['available'] != '0'), 'prices'] = df.loc[(df['available'] != '1') & (df['available'] != '0')].apply(sorting_prices, axis = 1)
df.loc[(df['available'] != '1') & (df['available'] != '0'), 'updated_prices'] = df.loc[(df['available'] != '1') & (df['available'] != '0')].apply(sorting_updated_prices, axis = 1)
df.loc[(df['available'] != '1') & (df['available'] != '0'), 'date_checks'] = df.loc[(df['available'] != '1') & (df['available'] != '0'), 'date_checks'].apply(sorting_dates)

# Correct available tickets
df['available'] = df['updated_prices'].apply(lambda x: 1 if x.split(',')[-1] != '0' else 0) # Better way to signify available tickets, as the query takes only the minimum, which doesn't always produce correct result

# 2. DATA CLEANING PART
# 2.1. CLEANSING MISSING VALUES FROM 'PRICES' COLUMN 

# Counting zeros in column 'price'
df['count_zeros'] = df['prices'].apply(lambda x: len([i for i in x.split(',') if i == '0']))
df['count_zeros_update'] = df['updated_prices'].apply(lambda x: len([i for i in x.split(',') if i == '0']))

# Label those that have zero at the end for the next function
df['last_zero'] = df['prices'].apply(lambda x: 1 if x.split(',')[-1] == '0' else 0)

# To indicate those that have zeros in `update_prices`, but no zeros in `prices`, you have to use df['last_zero'] column too
def two_or_more_zeros_update_prices(x):
    temp = x.split(',')
    try:
        if temp[-1] == '0' and temp[-2] == '0':
            return 1
        else: 
            return 0
    except IndexError:
        return 0
df['two_or_more_zeros_end_update_prices'] = df['updated_prices'].apply(two_or_more_zeros_update_prices)

# Find boundary 
def strip_zeros(row):
    a = row['updated_prices'].split(',')
    boundary = -1
    while a[boundary] == '0' and abs(boundary) <= len(a):
        boundary -= 1
    return boundary
df.loc[(df.two_or_more_zeros_end_update_prices == 1) & (df.count_zeros_update != df.num_checks), 'boundary'] = df.loc[(df.two_or_more_zeros_end_update_prices == 1) & (df.count_zeros_update != df.num_checks)].apply(strip_zeros, axis = 1)

# Eliminating those results that are likely to be non-existent
non_existent_yet = df.loc[((((df.count_zeros == df.num_checks) | (df.count_zeros_update == df.num_checks)) & (df.num_checks > 1)) | ((df.num_checks == 1) & (df.count_zeros_update == 1)))]
print("Number of rows of non-existent tickets: {}".format(non_existent_yet.shape[0]))
df = df.loc[~((((df.count_zeros == df.num_checks) | (df.count_zeros_update == df.num_checks)) & (df.num_checks > 1)) | ((df.num_checks == 1) & (df.count_zeros_update == 1)))]

# Remove zeros and correct all columns for rows that were affected
df.loc[(df.boundary < -2), 'prices'] = df.loc[df.boundary < -2].apply(lambda row: ','.join(row['prices'].split(',')[:int(row['boundary']+2)]), axis = 1)
df.loc[(df.boundary < -2), 'date_checks'] = df.loc[df.boundary < -2].apply(lambda row: ','.join(row['date_checks'].split(',')[:int(row['boundary']+2)]), axis = 1)
df.loc[(df.boundary < -2), 'num_checks'] = df.loc[(df.boundary < -2), 'num_checks'] + 2 + df.loc[(df.boundary < 0), 'boundary']
df.loc[(df.boundary < -2), 'count_zeros_update'] = df.loc[(df.boundary < -2), 'count_zeros_update'] + 2 + df.loc[(df.boundary < 0), 'boundary']

# For updated prices put zero at the end if the ticket is no longer available
def updated_prices_correction(row):
    temp = row['updated_prices'].split(',')[:int(row['boundary']+2)]
    if temp[-1] != '0' and row['available'] == 0:
        temp[-1] = '0'
    return ','.join(temp)
df.loc[(df.boundary < 0), 'updated_prices'] = df.loc[df.boundary < 0].apply(updated_prices_correction, axis = 1)

# Convert into correct datatype
df['num_checks'] = df['num_checks'].astype(int)
df['count_zeros'] = df['count_zeros'].astype(int)
# df.to_csv('temp2.csv')
# df = pd.read_csv('temp2.csv', index_col = 0)

# Recount zeros 
df.loc[(df.boundary < 0), 'count_zeros'] = df.loc[(df.boundary < 0), 'prices'].apply(lambda x: len([i for i in x.split(',') if i == '0']))
df.loc[(df.boundary < 0), 'count_zeros_update'] = df.loc[(df.boundary < 0), 'updated_prices'].apply(lambda x: len([i for i in x.split(',') if i == '0']))

# Replacing missing values with one check, where updated price exists (if not, then we can't do anything)
df.loc[(df['count_zeros'] == 1) & (df['num_checks'] == 1), 'prices'] = df.loc[(df['count_zeros'] == 1) & (df['num_checks'] == 1), 'updated_prices']

# Replacing missing value that is present somewhere in the price sequence
def replace_zero(row):
    temp = row['prices'].split(',')
    if temp[0] == '0':
        temp.insert(0, temp[1])
        temp.remove('0')
        return ','.join(temp)
    else:
        i = temp.index('0')
        temp.insert(i, temp[i-1])
        temp.remove('0')
        return ','.join(temp)
df.loc[(df['count_zeros'] == 1) & (df['num_checks'] > 1), 'prices'] = df.loc[(df['count_zeros'] == 1) & (df['num_checks'] > 1)].apply(replace_zero, axis = 1)

# Replace missing values for the rest of 'prices'
def replace_zero(row):
    init_list = row['prices'].split(',')
    temp = [i for i, n in enumerate(init_list) if n == '0']
    if 0 in temp:
        temp.remove(0)
    for i in temp:
        init_list.insert(i, init_list[i-1])
        del init_list[i+1]

    init_list = init_list[::-1]
    temp = [i for i, n in enumerate(init_list) if n == '0']
    if 0 in temp:
        temp.remove(0)
    if temp == []:
        init_list = init_list[::-1]
        return ','.join(init_list)
    else:
        for i in temp:
            init_list.insert(i, init_list[i-1])
            del init_list[i+1]
        init_list = init_list[::-1]
        return ','.join(init_list)
df.loc[df.count_zeros > 1, 'prices'] = df.loc[df.count_zeros > 1].apply(replace_zero, axis = 1)

# Recount zeros
df.loc[(df.count_zeros > 0), 'count_zeros2'] = df.loc[(df.count_zeros > 0), 'prices'].apply(lambda x: len([i for i in x.split(',') if i == '0']))
df = df.drop(columns = 'count_zeros')
df = df.rename(columns={'count_zeros2': 'count_zeros'})

# 2.2. CLEANSING MISSING VALUES FROM 'UPDATED_PRICES' COLUMN

df.loc[(df.count_zeros_update > 0), 'count_zeros_update'] = df.loc[(df.count_zeros_update > 0), 'updated_prices'].apply(lambda x: len([i for i in x.split(',') if i == '0']))

# Fill missing values for the updated prices by substituting the values from prices 
def fill_zeros(row):
    updated_prices = row['updated_prices'].split(',')
    prices = row['prices'].split(',')
    temp = [i for i, n in enumerate(updated_prices) if n == '0']
    for i in temp:
        if i == len(updated_prices) - 1:
            pass
        else:
            updated_prices.insert(i, prices[i])
            del updated_prices[i+1]
    return ','.join(updated_prices)
df.loc[(df.count_zeros_update > 0), 'updated_prices'] = df.loc[(df.count_zeros_update > 0)].apply(fill_zeros, axis = 1)

# Recount zeros
df.loc[(df.count_zeros_update > 0), 'count_zeros_update2'] = df.loc[(df.count_zeros_update > 0), 'updated_prices'].apply(lambda x: len([i for i in x.split(',') if i == '0']))
df = df.drop(columns = 'count_zeros_update')
df = df.rename(columns={'count_zeros_update2': 'count_zeros_update'})

# ## 2.3. CORRECTING THE PRICES THAT DON'T UPDATE

def price_does_not_update(row):
    prices = row['prices'].split(',')
    updated_prices = row['updated_prices'].split(',')
    for i in range(len(prices)-1):
        if prices[i+1] != updated_prices[i]:
            return 1
    return 0
df['price_does_not_update'] = df.apply(price_does_not_update, axis = 1)

def correct_updates(row):
    prices = row['prices'].split(',')
    updated_prices = row['updated_prices'].split(',')
    for i in range(len(prices)-1):
        if prices[i+1] != updated_prices[i]:
            prices.insert(i+1, updated_prices[i])
            del prices[i+2]
    return ','.join(prices)
df.loc[(df.price_does_not_update == 1), 'prices'] = df.loc[(df.price_does_not_update == 1)].apply(correct_updates, axis = 1)
df = df.drop(['last_zero',
       'two_or_more_zeros_end_update_prices', 'boundary', 'count_zeros',
       'count_zeros_update', 'price_does_not_update'], axis=1)

df['last_date'] = df['date_checks'].apply(lambda x: x.split(',')[-1])
today = df['last_date'].max()[:10]

def join(row):
    temp = row['date_checks'].split(',')
    temp.append(row['departure_date'] + ':00')
    return ','.join(temp)    
df.loc[(df.available == 1) & (df.departure_date < today), 'date_checks'] = df.loc[(df.available == 1) & (df.departure_date < today)].apply(join, axis=1)

def join(row):
    temp = row['prices'].split(',')
    temp.append(row['updated_prices'].split(',')[-1])
    return ','.join(temp)
df.loc[(df.available == 1) & (df.departure_date < today), 'prices'] = df.loc[(df.available == 1) & (df.departure_date < today)].apply(join, axis=1)

def join(x):
    temp = x.split(',')
    temp.append('0')
    return ','.join(temp)
df.loc[(df.available == 1) & (df.departure_date < today), 'updated_prices'] = df.loc[(df.available == 1) & (df.departure_date < today), 'updated_prices'].apply(join)
# Delete unnecessary column
df = df.drop(['last_date'], axis = 1)

df.to_csv('cleaned_data.csv')
print("Cleaned data saved!")
string = "Time taken to complete: " + str(datetime.now() - startTime)
print(string)