# -*- coding: utf-8 -*-
"""
@author: Pushkar Vengurlekar
"""

# Load necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, KFold
from sklearn import tree, neighbors
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.metrics import classification_report
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import MultinomialNB

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.metrics import confusion_matrix, roc_curve, auc, classification_report
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_predict
from sklearn import preprocessing
from sklearn.preprocessing import Normalizer
import math
from sklearn.svm import SVR
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestRegressor


df0 = pd.read_csv("train_data_class/final_final/train0.csv")
df1 = pd.read_csv("train_data_class/final_final/train1.csv")
df2 = pd.read_csv("train_data_class/final_final/train2.csv")
df3 = pd.read_csv("train_data_class/final_final/train3.csv")
df4 = pd.read_csv("train_data_class/final_final/train4.csv")
df5 = pd.read_csv("train_data_class/final_final/train5.csv")
df6 = pd.read_csv("train_data_class/final_final/train6.csv")
df7 = pd.read_csv("train_data_class/final_final/train7.csv")
df8 = pd.read_csv("train_data_class/final_final/train8.csv")
df9 = pd.read_csv("train_data_class/final_final/train9.csv")
df10 =pd.read_csv("train_data_class/final_final/train10.csv")
df11 =pd.read_csv("train_data_class/final_final/train11.csv")
df12 =pd.read_csv("train_data_class/final_final/train12.csv")
test =pd.read_csv("train_data_class/test.csv")

frames = [df0,df1, df2, df3, df4, df5, df6, df7, df8, df9, df10, df11, df12]
data = pd.concat(frames)

X_train_old, X_test_old, y_train_both, y_test_both = train_test_split(X, Y, test_size=0.2, random_state =42)


import xgboost as xgb
from sklearn.model_selection import train_test_split, GridSearchCV, KFold, cross_val_score, RandomizedSearchCV

clf =xgb.XGBClassifier(subsample= 0.8, reg_alpha= 0.1, n_estimators= 175, min_child_weight= 1, max_depth= 13, learning_rate= 0.1, gamma= 0.4, colsample_bytree=0.8)
clf.fit(X_train, y_train)

max_depth_range = range(1,30, 3)
min_samples_leaf_range = range(10,100,5)
max_features_list = ["auto", "sqrt"]
param_grid = dict(max_depth=max_depth_range,
                  max_features= max_features_list,
                  min_samples_leaf= min_samples_leaf_range)
rf = RandomForestRegressor()
grid_forest = GridSearchCV(rf,
                           param_grid,
                           cv=KFold(n_splits = 3,
                                    shuffle= True,
                                    random_state = 456),
                                    scoring = 'neg_mean_squared_error',
                                    n_jobs =-1)
grid_forest.fit(X_train, Y_train)


rf = RandomForestRegressor(max_depth=10, max_features= 'auto', min_samples_leaf= 4)
rf.fit(X_train, Y_train)


from sklearn.ensemble import AdaBoostRegressor
para_adaboost = {'n_estimators':[1, 2, 4, 8, 16, 32, 64, 100, 200],
                 'loss':['linear','square','exponential'],
                 'learning_rate':[0.5,1.0,2.0,3.0,5.0,10.0]}
ada = AdaBoostRegressor()

grid_adaboost = GridSearchCV(ada, para_adaboost, cv=KFold(n_splits = 3, shuffle= True, random_state = 456), scoring = 'neg_mean_squared_error', n_jobs = -1)
grid_adaboost.fit(X_train2,Y_train2)

print("Best Score:",-grid_adaboost.best_score_)
print("Best parameters:",grid_adaboost.best_params_)

from sklearn.metrics import mean_squared_error
best_ada = grid_adaboost.best_estimator_
print(best_ada,'\n')

prediction_best_ada = best_ada.predict(X_test2)
print("RMSE: {0:.2}".format(np.sqrt(mean_squared_error(Y_test2, prediction_best_ada))))

xg_cv  = xgb.XGBRegressor()
param_grid = {
        'silent': [False],
        'max_depth': [6, 10, 15, 20],
        'learning_rate': [0.001, 0.01, 0.1, 0.2, 0,3],
        'subsample': [0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
        'colsample_bytree': [0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
        'colsample_bylevel': [0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
        'min_child_weight': [0.5, 1.0, 3.0, 5.0, 7.0, 10.0],
        'gamma': [0, 0.25, 0.5, 1.0],
        'reg_lambda': [0.1, 1.0, 5.0, 10.0, 50.0, 100.0],
        'n_estimators': [100]}

xgb_for_grid = RandomizedSearchCV(estimator= xg_cv,
                                     param_distributions= param_grid,
                                     n_iter = 100, cv = 5, verbose=2,
                                     random_state=42,n_jobs = -1)

xgfinal = xgb.XGBRegressor(subsample= 0.9, silent= False, reg_lambda= 100.0, n_estimators= 100, min_child_weight= 10.0, max_depth= 6, learning_rate= 0.2, gamma= 0.25, colsample_bytree= 0.5, colsample_bylevel= 0.6)

prediction_test = xgfinal.predict(X_test)

from sklearn.metrics import mean_squared_error
print("MSE: {0:.2}".format(mean_squared_error(Y_test, prediction_test)))