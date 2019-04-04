# coding: utf-8
from datetime import datetime, timedelta
startTime = datetime.now()
current_datetime = str(datetime.now())[:16]
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# Import cities table to join the departure id and arrival id with cities and countries
# These will be used further as features
from sqlalchemy import create_engine
user_password = 'root:jKn8Po71kLP23'
db_name = 'sobus'
engine = create_engine('mysql+pymysql://' + user_password + '@localhost:3306/' + db_name)
con = engine.connect()
query = "SELECT main.id AS departure_id, city_id AS dep_city_id, a.name_fr AS dep_city, a.country AS dep_country \
         FROM station_multi AS main \
         LEFT JOIN (SELECT id, name_fr, country FROM city_multi) AS a \
         ON a.id = city_id"
rs = con.execute(query)
cities = pd.DataFrame(rs.fetchall())
cities.columns = rs.keys()
con.close()

filename = 'featured_data.csv'
df = pd.read_csv(filename, index_col = 0)

df['departure_date'] = df['departure_date'] + ' ' + df['departure_time']
df['arrival_date'] = df['arrival_date'] + ' ' + df['arrival_time']
df['departure_date'] = pd.to_datetime(df['departure_date'], format='%Y-%m-%d %H:%M')
df['arrival_date'] = pd.to_datetime(df['arrival_date'], format='%Y-%m-%d %H:%M')

# Duration of the trip in hours and weekday 
df['trip_duration'] = df['arrival_date'] - df['departure_date']
df['trip_duration'] = df['trip_duration'] / np.timedelta64(1, 'h')
df['weekday'] = df['departure_date'].apply(lambda x: x.weekday())

# Company feature
x = df.groupby('weekday')[['num_updates']].sum()
x.loc[(x.num_updates == 0), 'num_updates'] = 1
dic = np.log(x.num_updates).to_dict()
df['company'] = df['company_id'].apply(lambda x: dic[x] if x in dic.keys() else 0)

# Left join the city names
df = pd.merge(df, cities[['departure_id', 'dep_city_id', 'dep_city', 'dep_country']], how = 'left', on = 'departure_id')
cities = cities.rename(columns={'departure_id': 'arrival_id', 'dep_city_id': 'arr_city_id', 'dep_city': 'arr_city', 'dep_country': 'arr_country'})
df = pd.merge(df, cities[['arrival_id', 'arr_city_id', 'arr_city', 'arr_country']], how = 'left', on = 'arrival_id')

# Two cleaning steps
df.loc[(df.arr_country.isnull()), 'arr_country'] = 'IT'
df.loc[(df.dep_country.isnull()), 'dep_country'] = 'IT'
df = df.drop(list(df[df.dep_city.isnull()].index), axis=0)

df = df[['dep_city', 'arr_city', 'conditions', 'company_id', 'weekday', 'first_check', 'last_update', 'num_updates', 'life', 'recency', 'available', 'time_left', 'expired_early', 'trip_duration', 'company', 'aver_dist', 'label']]

# Conditions encoded
x = df.groupby('conditions')[['num_updates']].sum()
x.loc[(x.num_updates == 0), 'num_updates'] = 1
dic = np.log(x.num_updates).to_dict()
df['conditions'] = df['conditions'].apply(lambda x: dic[x] if x in dic.keys() else 0)

# Weekday encoded
x = df.groupby('weekday')[['num_updates']].sum()
x.loc[(x.num_updates == 0), 'num_updates'] = 1
dic = np.log(x.num_updates).to_dict()
df['weekday'] = df['weekday'].apply(lambda x: dic[x] if x in dic.keys() else 0)

# Departure city encoded
x = df.groupby('dep_city')[['num_updates']].sum()
x.loc[(x.num_updates == 0), 'num_updates'] = 1
dic = np.log(x.num_updates).to_dict()
df['departure'] = df['dep_city'].apply(lambda x: dic[x] if x in dic.keys() else 0)

# Arrival city encoded
x = df.groupby('arr_city')[['num_updates']].sum()
x.loc[(x.num_updates == 0), 'num_updates'] = 1
dic = np.log(x.num_updates).to_dict()
df['destination'] = df['arr_city'].apply(lambda x: dic[x] if x in dic.keys() else 0)
df[['life', 'recency']] = df[['life', 'recency']].replace(np.nan, 0)
df.loc[(df.recency == -1), 'recency'] = 0

#### CHECKPOINT ####
df.to_csv('final_data.csv')

#### PREDICTION SET ####
filename = 'prediction_set_full.csv'
df_train = pd.read_csv(filename, index_col = 0)

con = engine.connect()
query = "SELECT main.id AS departure_id, city_id AS dep_city_id, a.name_fr AS dep_city, a.country AS dep_country \
         FROM station_multi AS main \
         LEFT JOIN (SELECT id, name_fr, country FROM city_multi) AS a \
         ON a.id = city_id"
rs = con.execute(query)
cities = pd.DataFrame(rs.fetchall())
cities.columns = rs.keys()
con.close()

df_train['departure_date'] = df_train['departure_date'] + ' ' + df_train['departure_time']
df_train['arrival_date'] = df_train['arrival_date'] + ' ' + df_train['arrival_time']
df_train['departure_date'] = pd.to_datetime(df_train['departure_date'], format='%Y-%m-%d %H:%M')
df_train['arrival_date'] = pd.to_datetime(df_train['arrival_date'], format='%Y-%m-%d %H:%M')

# Duration of the trip in hours and weekday 
df_train['trip_duration'] = df_train['arrival_date'] - df_train['departure_date']
df_train['trip_duration'] = df_train['trip_duration'] / np.timedelta64(1, 'h')
df_train['weekday'] = df_train['departure_date'].apply(lambda x: x.weekday())

# Company feature
x = df.groupby('weekday')[['num_updates']].sum()
x.loc[(x.num_updates == 0), 'num_updates'] = 1
dic = np.log(x.num_updates).to_dict()
df_train['company'] = df_train['company_id'].apply(lambda x: dic[x] if x in dic.keys() else 0)

# Left join the city names
df_train = pd.merge(df_train, cities[['departure_id', 'dep_city_id', 'dep_city', 'dep_country']], how = 'left', on = 'departure_id')
cities = cities.rename(columns={'departure_id': 'arrival_id', 'dep_city_id': 'arr_city_id', 'dep_city': 'arr_city', 'dep_country': 'arr_country'})
df_train = pd.merge(df_train, cities[['arrival_id', 'arr_city_id', 'arr_city', 'arr_country']], how = 'left', on = 'arrival_id')

# Two cleaning steps
df_train.loc[(df_train.arr_country.isnull()), 'arr_country'] = 'IT'
df_train.loc[(df_train.dep_country.isnull()), 'dep_country'] = 'IT'
df_train = df_train.drop(list(df_train[df_train.dep_city.isnull()].index), axis=0)

df_train = df_train[['dep_city', 'arr_city', 'conditions', 'company_id', 'weekday', 'first_check', 'last_update', 'num_updates', 'life', 'recency', 'available', 'time_left', 'expired_early', 'trip_duration', 'company', 'aver_dist']]

# Conditions encoded
x = df_train.groupby('conditions')[['num_updates']].sum()
x.loc[(x.num_updates == 0), 'num_updates'] = 1
dic = np.log(x.num_updates).to_dict()
df_train['conditions'] = df_train['conditions'].apply(lambda x: dic[x] if x in dic.keys() else 0)

# Weekday encoded
x = df_train.groupby('weekday')[['num_updates']].sum()
x.loc[(x.num_updates == 0), 'num_updates'] = 1
dic = np.log(x.num_updates).to_dict()
df_train['weekday'] = df_train['weekday'].apply(lambda x: dic[x] if x in dic.keys() else 0)

# Departure city encoded
x = df_train.groupby('dep_city')[['num_updates']].sum()
x.loc[(x.num_updates == 0), 'num_updates'] = 1
dic = np.log(x.num_updates).to_dict()
df_train['departure'] = df_train['dep_city'].apply(lambda x: dic[x] if x in dic.keys() else 0)

# Arrival city encoded
x = df_train.groupby('arr_city')[['num_updates']].sum()
x.loc[(x.num_updates == 0), 'num_updates'] = 1
dic = np.log(x.num_updates).to_dict()
df_train['destination'] = df_train['arr_city'].apply(lambda x: dic[x] if x in dic.keys() else 0)
df_train[['life', 'recency']] = df_train[['life', 'recency']].replace(np.nan, 0)
df_train.loc[(df_train.recency == -1), 'recency'] = 0

#### CHECKPOINT ####
df_train.to_csv('final_prediction_set.csv')
#### END OF PREDICTION SET ####

## MODELLING ##
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegressionCV
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils import class_weight
from sklearn.metrics import confusion_matrix, f1_score, recall_score
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import make_pipeline
import pickle

df = pd.read_csv('final_data.csv', index_col = 0)
# xt = pd.read_csv('xt.csv', index_col=0)
# yt = pd.read_csv('yt.csv', index_col=0)

df[['last_update', 'first_check', 'life', 'recency']] = df[['last_update', 'first_check', 'life', 'recency']].replace(np.nan, '0')
df.drop(['dep_city', 'arr_city', 'company_id', 'first_check', 'last_update'], axis = 1, inplace=True)
features = list(df.columns)
features.remove('label')
X = df[features]
y = df[['label']]

# x_train, x_test, y_train, y_test = train_test_split(X, y , test_size=0.2, random_state=0, stratify=y)
# pipe = LogisticRegressionCV(class_weight='balanced', penalty='l2', cv=3)
# pipe.fit(x_train, y_train)
# y_pred = pipe.predict(x_test)
# print('Logistic CV class:')
# print(confusion_matrix(y_test, y_pred))
# print(f1_score(y_test, y_pred), recall_score(y_test, y_pred))
# pickle.dump(pipe, open("MODEL1.pickle.dat", "wb"))

# y_pred = pipe.predict(xt)
# print('Test set results: ')
# print(confusion_matrix(yt, y_pred))
# print(f1_score(yt, y_pred), recall_score(yt, y_pred))
# print()

x_train, x_test, y_train, y_test = train_test_split(X, y , test_size=0.2, random_state=0, stratify=y)
pipe = XGBClassifier(learning_rate=0.1, n_estimators=100, 
                      scale_pos_weight=y_train.shape[0] / y_train[y_train.label==1].shape[0],
                     max_depth=5, min_child_weight=1, gamma=0, subsample=0.8, colsample_bytree=0.8,
                     )
pipe.fit(x_train, y_train)
y_pred = pipe.predict(x_test)
print('XGBoost class:')
print(confusion_matrix(y_test, y_pred))
print(f1_score(y_test, y_pred), recall_score(y_test, y_pred))
pickle.dump(pipe, open("MODEL.pickle.dat", "wb"))

# Prediction
df = pd.read_csv('final_prediction_set.csv', index_col=0)
df[['last_update', 'first_check', 'life', 'recency']] = df[['last_update', 'first_check', 'life', 'recency']].replace(np.nan, '0')
df.drop(['dep_city', 'arr_city', 'company_id', 'first_check', 'last_update'], axis = 1, inplace=True)  
pipe = pickle.load(open('MODEL.pickle.dat', 'rb'))

features = list(df.columns)
X = df[features]
y_pred = pipe.predict(X)
df['to_update'] = y_pred
print(' Number of predicted values: ', df[df.to_update == 1].shape[0])

cols = ['departure_id', 'arrival_id', 'company_id', 'departure_time', 'arrival_time', 'conditions']
d = pd.read_csv('prediction_set_full.csv', usecols=['departure_id', 'arrival_id', 'company_id', 'departure_time', 'arrival_time', 'conditions', 'departure_date', 'arrival_date'])
d['departure_time'] = d['departure_date'] + ' ' + d['departure_time'] + ':00'
d['arrival_time'] = d['arrival_date'] + ' ' + d['arrival_time'] + ':00'
df[cols] = d[cols]
df[df.to_update == 1][cols].to_sql('tickets_to_update', con=engine, if_exists='replace', index = False)
print("SUCCESS: {} rows were updated in the table".format(df[df.to_update == 1].shape[0]))
string = "Time taken to complete: " + str(datetime.now() - startTime)
print(string)