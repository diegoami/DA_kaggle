from sklearn.ensemble import  RandomForestRegressor, GradientBoostingRegressor
import argparse
import os

from sklearn.linear_model import Lasso
from sklearn.preprocessing import LabelBinarizer

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV

import xgboost as xgb


import sys

from xgboost import XGBRegressor

sys.path.append('..')
from kaggle_solver import KaggleSolver
import pandas as pd
import numpy as np

def preprocess(tdata):
    columns_objects, columns_others = tdata.columns[tdata.dtypes == np.dtype('O')], tdata.columns[tdata.dtypes != np.dtype('O')]
    tdata[columns_objects] = tdata[columns_objects].fillna('Unknown')
    tdata[columns_others] = tdata[columns_others].fillna(0)
    return tdata

def postprocess_categorical(tdata):
    columns_objects, columns_others = tdata.columns[tdata.dtypes == np.dtype('O')], tdata.columns[tdata.dtypes != np.dtype('O')]

    for column_object in columns_objects:
        tdata[column_object] = tdata[column_object].astype('category')
        tdata[column_object] = tdata[column_object].cat.codes
    scaler = StandardScaler()
    tdata[columns_others] = scaler.fit_transform(tdata[columns_others] )
    return tdata



def postprocess_hard(tdata):
    columns_objects, columns_others = tdata.columns[tdata.dtypes == np.dtype('O')], tdata.columns[tdata.dtypes != np.dtype('O')]


    for column_object in columns_objects:
        dummy_columns = pd.get_dummies(tdata[column_object], prefix=column_object, sparse=True)
        tdata = pd.concat([tdata,dummy_columns])
        tdata = tdata.drop(column_object)
        return tdata

    for column_object in columns_others:
        pass



if __name__ == "__main__":
    argparser = argparse.ArgumentParser()

    argparser.add_argument('--percentage')
    args = argparser.parse_args()

    percentage = 1
    perc_100 = str(int(percentage*100))

    if args.percentage :
        print("Training on only {}% of samples".format(perc_100))
        percentage = float(args.percentage)
        perc_100 = str(int(percentage * 100))

    kaggleSolver = KaggleSolver(id_column='Id',result_column='SalePrice', train_filename='data/train.csv',
                                test_filename='data/test.csv', preprocess=preprocess, postprocess=postprocess_categorical, percentage=percentage )
    rfparams = {
        "max_features" : range(16,24,2),
    }
    xgbparams = {
        "learning_rate" : (0.001, 0.01, 0.1),
        "n_estimators"  : (5000,6000,7000)

    }

    classifier_params = [
       {"classifier": RandomForestRegressor(max_features=20, n_estimators=1500,min_samples_leaf=1),"output_file": "out/rfc_maf_20_nes_1000_msl_1_" + perc_100 + ".csv"},
        {"classifier": XGBRegressor(max_depth=5, min_child_weight=3, gamma=0, colsample_bytree=0.9, subsample=0.7, alpha=1e-3,learning_rate=0.01, n_estimators=5000)                                   ,"output_file":"out/xgbregressor_4_mad_5_mcw_3_gamma_0_cbt_09_ssa_07_a_1e3_lr_001_nes=5000" + perc_100 + ".csv"},

        {"classifier": GridSearchCV(RandomForestRegressor(n_estimators=1500,min_samples_leaf=1), rfparams),"output_file" : "out/grid_rfc_" + perc_100 + ".csv"},
        {"classifier":
            GridSearchCV    (XGBRegressor
                                (max_depth=5, min_child_weight=3, gamma=0, colsample_bytree=0.9, subsample=0.7, alpha=1e-3), xgbparams
                            ),
          "output_file": "out/grid_xgbregressor_4_mad_5_mcw_3_gamma_0_cbt_09_ssa_07_a_1e3_" + perc_100 + ".csv"
         }

    ]
    print(classifier_params)
    for _ in classifier_params:
        if not os.path.isfile(_["output_file"]):
            kaggleSolver.solve(classifier=_["classifier"], output_file=_["output_file"] )