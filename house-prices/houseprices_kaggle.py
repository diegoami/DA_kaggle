from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, RandomForestRegressor, \
    GradientBoostingRegressor
import argparse
from sklearn.preprocessing import LabelBinarizer
from sklearn.preprocessing import StandardScaler

import sys
sys.path.append('..')
from kaggle_solver import KaggleSolver
import pandas as pd
import numpy as np

def preprocess(tdata):
    columns_objects, columns_others = tdata.columns[tdata.dtypes == np.dtype('O')], tdata.columns[tdata.dtypes != np.dtype('O')]
    tdata[columns_objects] = tdata[columns_objects].fillna('Unknown')
    tdata[columns_others] = tdata[columns_others].fillna(0)
    return tdata

def postprocess(tdata):
    columns_objects, columns_others = tdata.columns[tdata.dtypes == np.dtype('O')], tdata.columns[tdata.dtypes != np.dtype('O')]
#    for column_object in columns_objects:
#        dummy_columns = pd.get_dummies(tdata[column_object], prefix=column_object)
#        tdata = pd.concat([tdata,dummy_columns])
#        tdata = tdata.drop(column_object)
#    return tdata
    for column_object in columns_objects:
        tdata[column_object] = tdata[column_object].astype('category')
        tdata[column_object] = tdata[column_object].cat.codes
    scaler = StandardScaler()
    tdata[columns_others] = scaler.fit_transform(tdata[columns_others] )

    return tdata

if __name__ == "__main__":
    argparser = argparse.ArgumentParser()

    argparser.add_argument('--percentage')
    args = argparser.parse_args()

    percentage = float(args.percentage)
    perc_100 = str(percentage *1000)

    if args.percentage:
        print("Training on only {} of samples".format(args.percentage))
        percentage = float(args.percentage)

    kaggleSolver = KaggleSolver(id_column='Id',result_column='SalePrice', train_filename='data/train.csv',
                                test_filename='data/test.csv', preprocess=preprocess, postprocess=postprocess, percentage=percentage )

    classifier_params = [
      #  {"classifier" : SVC(kernel='linear', C=1000, gamma=1000), "output_file" : "out/svcl_C_1000_gam_1000.csv"},
        {"classifier":   GradientBoostingRegressor(n_estimators=1000, max_depth=8), "output_file": "out/gbc_nes_1000_mad_t_8_"+perc_100+".csv"},
        {"classifier": RandomForestRegressor(max_features=20, n_estimators=1000),"output_file": "out/rfc_maf_10_nes_1000_" + perc_100 + ".csv"}
    ]

    for _ in classifier_params:
        kaggleSolver.solve(classifier=_["classifier"], output_file=_["output_file"] )