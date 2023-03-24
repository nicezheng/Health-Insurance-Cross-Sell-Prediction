"""
@File : model_xgboost.py
@Author : 计科18-1 181002105 蒋政
@Date : 2021/7/6
@Task : 
"""
import pickle

import xgboost as xgb
import catboost as cb
import lightgbm as lgb
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import roc_auc_score, classification_report, accuracy_score, roc_curve, auc
from sklearn.linear_model import LogisticRegression
from matplotlib.pyplot import plot, title, xlabel, ylabel
from sklearn.model_selection import RandomizedSearchCV
from sklearn.neighbors import KNeighborsClassifier
import datetime

import numpy as np
def get_model_params(clf_name=None, hyper_params={}):
    if clf_name == "lgb":
        params_dict = {
            'n_estimators': range(100, 200, 1),
            'max_depth': range(2, 10, 1),
            'learning_rate': np.array([0.1,0.01,0.001,0.0001]),
            'colsample_bytree': np.linspace(0.1, 0.98, 5),
            'min_child_weight': range(1, 10, 1),
        }
        clf = lgb.LGBMClassifier(**hyper_params)
    elif clf_name == 'xgb':
        params_dict = {
            'n_estimators': range(100, 300, 4),
            'max_depth': range(3, 18, 1),
            'colsample_bytree': np.linspace(0.5, 1, 2),
            'min_child_weight': range(1, 10, 1),
            'reg_alpha': range(40,180,1),
            'reg_lambda': np.linspace(0.5, 1, 2),
            'gamma': range(1, 9, 1)
        }
        clf = xgb.XGBClassifier(**hyper_params)
    elif clf_name == 'cb':
        params_dict = {
            'n_estimators': range(80, 200, 4),
            'max_depth': range(2, 15, 2),
            'learning_rate': np.linspace(0.01, 0.1, 10),
            'l2_leaf_reg': range(1,10,2),
        }
        clf = cb.CatBoostClassifier(**hyper_params)
    elif clf_name == 'rf':
        params_dict = {
            'n_estimators': range(80, 300, 4),
            'max_depth': range(2, 15, 1),
            # 'learning_rate': np.linspace(0.01, 1, 20),
            'min_samples_leaf': [4, 6, 8],
            'min_samples_split': [5, 7, 10],
        }
        clf = RandomForestClassifier(**hyper_params)
    elif clf_name == 'gb':
        params_dict = {
            'n_estimators': range(80, 300, 4),
            'max_depth': range(2, 10, 1),
            'learning_rate': np.linspace(0.01, 2, 20),
            'min_samples_leaf': [3],
            'min_samples_split': [5, 7, 10],
        }
        clf = GradientBoostingClassifier(**hyper_params)
    elif clf_name == 'knn':
        params_dict = {
           'n_neighbors': [3],
            'weights': ['uniform', 'distance']
        }
        clf = KNeighborsClassifier(**hyper_params)
    else:
        clf = None
        params_dist = None
    return clf, params_dict