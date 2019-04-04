import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.linear_model import LogisticRegressionCV
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils import class_weight
from sklearn.metrics import confusion_matrix, f1_score, recall_score
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV

filename = 'data_x_2018-12-10 10:29_1.csv'
df = pd.read_csv(filename, index_col = 0)
                 
df[['last_update', 'first_check', 'life', 'recency']] = df[['last_update', 'first_check', 'life', 'recency']].replace(np.nan, '0')
df = df[['departure_id', 'arrival_id', 'departure_time', 'arrival_time', 'conditions', 'flixbus',
       'ouibus', 'distribusion', 'top95', 'num_updates', 'recency', 'available', 'time_left', 'expired_early', 'label']]
df.drop(['departure_id', 'arrival_id', 'departure_time', 'arrival_time'], axis = 1, inplace=True)

features = list(df.columns)
features.remove('label')
X = df[features]
X.head()
x_train, x_test, y_train, y_test = train_test_split(X, df[['label']], test_size=0.25, random_state=0, stratify=df[['label']])

from imblearn.over_sampling import SMOTE
from imblearn.pipeline import make_pipeline

# pipe = make_pipeline(
#     SMOTE(),
#     RandomForestClassifier(n_estimators = 100, min_samples_leaf=4, min_samples_split=10))
# pipe.fit(x_train, y_train)
# y_pred = pipe.predict(x_test)
# print(confusion_matrix(y_test, y_pred))

# # save the smote rf model
# import pickle
# pickle.dump(pipe, open("rf_smote.pickle.dat", "wb"))

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) 

pipe = make_pipeline(
    SMOTE(),
    XGBClassifier()
)

weights = np.linspace(0.005, 0.05, 10)
# Best parameters {'smote__ratio': 0.005}
gsc = GridSearchCV(
    estimator=pipe,
    param_grid={
        #'smote__ratio': [{0: int(num_neg), 1: int(num_neg * w) } for w in weights]
        'smote__ratio': weights
    },
    scoring='f1',
    cv=3
)
grid_result = gsc.fit(x_train, y_train)

print("Best parameters : %s" % grid_result.best_params_)