from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.svm import SVC


from kaggle_solver import KaggleSolver
import pandas as pd

def substrings_in_string(big_string, substrings):
    for substring in substrings:
        if big_string.find(substring) != -1:
            return substring
    print(big_string)
    return 0


def get_df(tdata):
    cabin_list = ['A', 'B', 'C', 'D', 'E', 'F', 'T', 'G', 'Unknown']
    title_list = ['Mrs', 'Mr', 'Master', 'Miss', 'Major', 'Rev',
                  'Dr', 'Ms', 'Mlle', 'Col', 'Capt', 'Mme', 'Countess',
                  'Don', 'Jonkheer']

    def convSex(row):
        return 1 if row['Sex'] == 'male' else 0

    def convEmbarked(row):
        return {'S': 0, 'C': 1, 'Q': 2}[row['Embarked']]

    # replacing all titles with mr, mrs, miss, master
    def conv_deck(x):
        deck = x['Deck']
        if deck in cabin_list:
            return cabin_list.index(deck)
        else:
            return 0

    def replace_titles(x):
        title = x['Title']
        if title in ['Don', 'Major', 'Capt', 'Jonkheer', 'Rev', 'Col']:
            return 0
        elif title in ['Countess', 'Mme']:
            return 1
        elif title in ['Mlle', 'Miss']:
            return 2
        elif title == 'Dr':
            if x['Sex'] == 'male':
                return 3
            else:
                return 4
        else:
            return 5


    tdata['Cabin'] =     tdata['Cabin'].fillna("")
    tdata['Deck'] = tdata ['Cabin'].map(lambda x: substrings_in_string(x, cabin_list) if x != "" else 0)
    tdata['Deck'] = tdata.apply(conv_deck, axis=1)

    med_age = tdata.median()['Age']

    tdata['Age'] = tdata['Age'].fillna(med_age)

    tdata['Title'] = tdata['Name'].map(lambda x: substrings_in_string(x, title_list))
    tdata['Title'] = tdata.apply(replace_titles, axis=1)

    med_fare = tdata.median()['Fare']

    tdata['Fare'] = tdata['Fare'].fillna(med_fare)
    tdata['Embarked'] = tdata['Embarked'].fillna('C')
    tdata['SexC'] = tdata.apply(convSex, axis=1)
    tdata['Family_Size'] = tdata['SibSp'] + tdata['Parch']
    tdata['Fare_Per_Person'] = tdata['Fare'] / (tdata['Family_Size'] + 1)
    tdata['EmbarkedC'] = tdata.apply(convEmbarked, axis=1)
    return tdata

def step_df(tdata):
    dummy_sex = pd.get_dummies(tdata['SexC'], prefix='SexC')
    dummy_embarked = pd.get_dummies(tdata['EmbarkedC'], prefix='EmbarkedC')
    dummy_title = pd.get_dummies(tdata['Title'], prefix='Title')
    dummy_pclass = pd.get_dummies(tdata['Pclass'], prefix='Pclass')
    dummy_pclass = pd.get_dummies(tdata['Deck'], prefix='Deck')

    tdata = pd.concat([tdata, dummy_sex, dummy_embarked, dummy_title, dummy_pclass], axis=1)

    if 'PassengerId' in tdata.columns:
        tdata = tdata.drop('PassengerId', axis=1)

    return tdata


kaggleSolver = KaggleSolver(id_column='PassengerId',result_column='Survived', train_filename='data/train.csv',
                            test_filename='data/test.csv', preprocess=get_df, postprocess=step_df)

classifier_params = [
    {"classifier" : SVC(kernel='linear', C=1000, gamma=1000), "output_file" : "out/svcl_C_1000_gam_1000.csv"},
    {"classifier": GradientBoostingClassifier(n_estimators=1000, max_depth=5), "output_file": "out/gbc_nes_1000_mad_t.csv"},
    {"classifier": RandomForestClassifier(max_features=10,n_estimators=1000), "output_file": "out/rfc_maf_10_nes_1000.csv"}
]

rel_columns = [ 'Age', 'SibSp', 'Parch',
       'Fare',  'SexC_0', 'SexC_1', 'EmbarkedC_0',
       'EmbarkedC_1', 'EmbarkedC_2', 'Title_0', 'Title_1', 'Title_2',
       'Title_3', 'Title_4', 'Title_5', 'Deck_0', 'Deck_1', 'Deck_2', 'Deck_3',
       'Deck_4', 'Deck_5', 'Deck_6', 'Deck_7']

for _ in classifier_params:
    kaggleSolver.solve(classifier=_["classifier"], relevant_columns=rel_columns, output_file=_["output_file"] )