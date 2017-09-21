from sklearn.ensemble import  RandomForestClassifier
import argparse
import os

from sklearn.linear_model import Lasso
from sklearn.preprocessing import LabelBinarizer

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV

import xgboost as xgb


import sys

from xgboost import XGBRegressor, XGBClassifier

sys.path.append('..')
from kaggle_solver import KaggleSolver

def preprocess(tdata):
    return tdata
def postprocess(tdata):
    return tdata




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

    for p in range(5,105,5):
        percentage = p/100
        perc_100 = str(int(percentage * 100))
        print("Processsing percentage = "+perc_100 +"%")
        kaggleSolver = KaggleSolver(id_column=None,result_column='Label', train_filename='data/train.csv',
                                    test_filename='data/test.csv', preprocess=preprocess, postprocess=postprocess, percentage=percentage, id_column_out='ImageId')

        classifier_params = [
           {"classifier": RandomForestClassifier(n_estimators=50),"output_file": "out/rfc_n20_1_" + perc_100 + ".csv"},
            {"classifier": XGBClassifier(max_depth=5, min_child_weight=3),"output_file": "out/xgb_mad_5_cwe_3_1_" + perc_100 + ".csv"},
        ]
        print(classifier_params)
        for _ in classifier_params:
            if not os.path.isfile(_["output_file"]):
                kaggleSolver.solve(classifier=_["classifier"], output_file=_["output_file"] )