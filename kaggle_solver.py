import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score
from math import floor

class KaggleSolver:


    def __init__(self, train_filename, test_filename, id_column, result_column, preprocess, postprocess, percentage):


        self.train_filename = train_filename
        self.test_filename  = test_filename
        self.id_column = id_column
        self.result_column = result_column
        self.preprocess = preprocess
        self.postprocess = postprocess
        self.percentage = percentage

    def print_score(self, classifier, X_train, y_train):

        scores = cross_val_score(classifier, X_train, y_train, cv=5)
        print(scores)

    def print_sol(self, classifier, X_test, output_file, ids):

        yp = classifier.predict(X_test)
        dsol = pd.DataFrame()
        dsol[self.id_column] = ids
        dsol[self.result_column] = yp

        dsol = dsol.set_index(self.id_column)
        dsol.to_csv(output_file, index_label=self.id_column)


    def solve(self, classifier,  output_file):

        train_df = pd.read_csv(self.train_filename)
        test_df = pd.read_csv(self.test_filename)

        if (self.percentage < 1):
            lperc = floor(len(train_df)*self.percentage)
            train_df = train_df[:lperc]

        if self.preprocess != None:
            train_df = self.preprocess(train_df )
            test_df = self.preprocess(test_df )

        ids = test_df[self.id_column]
        results = train_df[self.result_column]


        if self.postprocess != None:
            train_df = self.postprocess(train_df)
            test_df = self.postprocess(test_df)

        relevant_columns = [x for x in list(train_df.columns) + list(test_df.columns) if x not in [self.id_column, self.result_column]]

        miss_test_columns = [x for x in relevant_columns if x not in test_df.columns]
        miss_train_columns = [x for x in relevant_columns if x not in train_df.columns]

        for ms in miss_test_columns:
            test_df[ms] = 0

        for ms in miss_train_columns:
            train_df[ms] = 0


        X_train = np.array(train_df[relevant_columns])
        y_train = np.array(results)

        X_test = np.array(test_df[relevant_columns])
        classifier.fit(X=X_train, y=y_train)

        self.print_score(classifier=classifier, X_train=X_train, y_train=y_train)
        self.print_sol(classifier=classifier, X_test=X_test, output_file=output_file, ids=ids)