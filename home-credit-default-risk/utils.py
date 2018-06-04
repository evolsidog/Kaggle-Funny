# -*- coding: utf-8 -*-

from sklearn.metrics import roc_curve, auc, classification_report, precision_recall_curve
from sklearn import preprocessing
from sklearn.base import TransformerMixin
from sklearn.model_selection import GridSearchCV
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def split_num_str_data(df):
    '''

    :param df: Dataframe that we want to split
    :return: Returns two dataframes. The first is a dataframe which only contains numerical variables. The second
    dataframe only contains string data.
    '''

    df_num = df.select_dtypes(include=['int64', 'float64', 'uint8'])
    df_str = df.select_dtypes(include='object')
    return df_num, df_str


def preprocess_test_set(df_test, df_train):
    '''

    This function prepares the test set for the prediction.
    This preparation includes:
    1 - Keep the same columns as in the training set
    2 - Impute null values
    3 - Encode categorical vlaues

    :param df: Test set dataframe
    :return:
    '''
    print("Adjusting test data to predict over it")
    # Keep the same columns
    df_test = df_test.loc[:, df_test.columns.isin(df_train.columns)]

    df_test = DataFrameImputer(fill_type="mean_mode").fit_transform(X=df_test)

    df_test_num, df_test_str = split_num_str_data(df_test)

    df_str = encode_categorical_variables(df_test_str)
    df_num = pd.concat([df_test_num, df_str], axis=1)

    return df_num

def pr_auc_score(y_true, y_score):
    """
    Generates the Area Under the Curve for precision and recall.
    """
    precision, recall, thresholds = \
        precision_recall_curve(y_true, y_score[:, 1])

    return auc(recall, precision, reorder=True)

def estimate_auc(y_test, y_pred, name=None):
    '''
    This function calculates de area under the roc curve.

    :param y_test: Real labels([1, 0, 1, etc])
    :param name: Name you want to give the file
    :param y_pred: Predicted labels as a probability score([0.123678, 0.967, 0.1231, etc])
    :return:
    '''
    fpr, tpr, thresholds = roc_curve(y_true=y_test, y_score=y_pred, pos_label=1)
    roc_auc = auc(fpr, tpr)
    print("AUC ", roc_auc)

    fig = plt.figure()
    plt.title('Receiver Operating Characteristic')
    plt.plot(fpr, tpr, 'b', label='AUC = %0.5f' % roc_auc)
    plt.legend(loc='lower right')
    plt.plot([0, 1], [0, 1], 'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    fig.savefig('roc_curve_' + name + '_.png')


def plot_grid_search(clf_grid, n_estimators, max_depths):
    print(clf_grid.best_params_)

    fig = plt.figure()
    plt.title('Grid Search')

    scores = [x[1] for x in clf_grid.grid_scores_]
    scores = np.array(scores).reshape(len(n_estimators), len(max_depths))

    for ind, i in enumerate(n_estimators):
        plt.plot(max_depths, scores[ind], label='C: ' + str(i))
    plt.legend()
    plt.xlabel('Max Depth')
    plt.ylabel('Number estimators')
    plt.show()
    fig.savefig('GridResults.png')


def get_important_features(clf, columns):
    '''

    :param clf: Sklearn Classifier optimized with GridSearch
    :param columns: list of columns
    :return:
    '''
    feature_importances = clf.feature_importances_
    f_i = sorted(zip(map(lambda x: round(x, 4), feature_importances), columns), reverse=True)
    print("Most important features are: ", f_i)

    return f_i


def get_most_important_features(f_i):
    r = []
    for item in f_i:
        if item[0] > 0.01:
            r.append(item[1])

    return r


def report_classification(y_test, predicted):
    report = classification_report(y_test, predicted)
    print("Classification report: ", report)


def encode_categorical_variables(df):
    '''


    :param df: Codifies each column of the dataframe with Label Encoder algorithm.

     LabelEncoder acts in two steps:

     1 - Fit:

        'a' -> 1
        'b' -> 2
        'c' -> 3
        'd' -> 4

     2 - Transform:

        ['a', 'a', 'b', 'c', 'a', 'd'] --> [1, 1, 2, 3, 1, 4]


    :return: The encoded Dataframe
    '''

    le = preprocessing.LabelEncoder()
    for col in df.columns:
        le.fit(df[col])
        df.loc[:, col] = le.transform(df[col])
        df.loc[:, col] = df[col] - df[col].median()
    return df


def drop_nan_by_thresh(df, thresh, axis):
    shape = axis - 1 if axis == 1 else axis + 1
    df = df.dropna(axis=axis, thresh=int(thresh * df.shape[shape]))
    return df


def deal_with_nan(df, users_default=None, action="drop_rows"):
    '''

    :param df: Dataframe we want to remove NaN from
    :param action: Type of action. 'drop_rows' if we want to drop rows with at least one NaN.
    Choose "impute" if you want to impute the NaN values
    :return: The Dataframe without NaN's
    '''

    # We want mantein maximum defaults users. We drop targets 0 and impute targets 1.
    # users_pay_df = df[~df["SK_ID_CURR"].isin(users_default)]
    # users_defaults_df = df[df["SK_ID_CURR"].isin(users_default)]
    #
    # print("rows which target are 0", users_pay_df.shape[0])
    # users_pay_df = users_pay_df.dropna(axis=0, thresh=int(0.5 * users_pay_df.shape[1]))
    # users_pay_df = DataFrameImputer(fill_type="mean_mode").fit_transform(X=users_pay_df)
    # print("dropping row which contains nan's ", users_pay_df.shape[0])
    #
    # users_defaults_df = users_defaults_df.dropna(axis=0, thresh=int(0.5 * users_defaults_df.shape[1]))
    # users_defaults_df = DataFrameImputer(fill_type="mean_mode").fit_transform(X=users_defaults_df)
    # print("imputing with mode and mean rows which target are 1")

    if action == "drop_rows":
        df = df.dropna(axis=0)
        print("dropping row which contains nan's ", df.shape[0])

    elif action == "impute":
        df = DataFrameImputer(fill_type="mean_mode").fit_transform(X=df)
        print("imputing with mode and mean")

    elif action == "impute_knn":

        # Split numeric and non numeric data
        df_app_train_num, df_app_train_str = split_num_str_data(df)

        df_app_train_str = DataFrameImputer(fill_type="mean_mode").fit_transform(X=df_app_train_str)


        print("Encoding categorical variables")
        df_app_train_str = encode_categorical_variables(df_app_train_str)
        df_app_train = pd.concat([df_app_train_num, df_app_train_str], axis=1)
        df = KNN(k=2).complete(df_app_train)

    # return pd.concat([users_pay_df, users_defaults_df])
    return df

def get_best_grid_cv_model(clf, param_grid, X_train, y_train):
    '''

    :param clf: Sklearn classifier(such as RandomForestClassifier, GradientBoostingClassifier, etc)
    :param param_grid: Object of type dictionary:

        param_grid = {
        "n_estimators": [200, 300],
        "max_depth": [20, 25],
        "min_samples_leaf": [3, 5],
        'min_samples_split': [5, 10],
        'bootstrap': [True, False]
    }

    :param X_train: Dataframe without target variable
    :param y_train: Target variable

    :return: A classifier instance with the optimal parameters for the current dataset and parameters grid
    '''

    cv_grid_cv = GridSearchCV(estimator=clf, param_grid=param_grid, verbose=10)
    cv_grid_cv.fit(X_train, y_train)

    print(cv_grid_cv.best_estimator_.get_params())

    params = {k: cv_grid_cv.best_estimator_.get_params()[k] for k in param_grid}

    clf = clf(**params)

    return clf


def mini_profiling(df, name=None):
    num_rows = 10
    with pd.option_context('display.max_rows', num_rows, 'display.max_columns', len(df.columns)):
        print("------------ Mini profiling " + name + " ---------------")
        print(name + " shape: ", df.shape)
        print(name + " data types: ", df.dtypes)
        print(name + " describe: ", df.describe())
        print(df.head(num_rows))


def clean_df(df, thresh, action='drop_rows'):
    print("Columns before dropping column nan's", df.shape[1])
    # Removes columns
    df = drop_nan_by_thresh(df=df, thresh=thresh, axis=1)
    print("Columns after dropping column nan's", df.shape[1])

    print("Rows before dealing with nan ", df.shape[0])
    # Remove rows
    df = deal_with_nan(df=df, users_default=None, action=action)
    print("Rows after dealing with nan ", df.shape[0])

    # Split numeric and non numeric data
    df_num, df_str = split_num_str_data(df)

    print("Encoding categorical variables")
    df_str = encode_categorical_variables(df_str)
    df = pd.concat([df_num, df_str], axis=1)

    return df


# Categorical DataFrame imputer
# https://stackoverflow.com/questions/25239958/impute-categorical-missing-values-in-scikit-learn?utm_medium=organic&utm_source=google_rich_qa&utm_campaign=google_rich_qa


class DataFrameImputer(TransformerMixin):
    def __init__(self, fill_type="mean_mode"):
        '''

        :param fill_type: Should be 'mean_mode' to impute numerical columns with mean and categorical columns with mode.
        Should be 'const' if we want to impute with a constant value(usually -1)
        '''
        self.fill_type = fill_type

    def fit(self, X, y=None):
        '''

        :param X: Dataframe we want to impute
        :return: Filled dataframe
        '''

        if self.fill_type == "mean_mode":
            self.fill = pd.Series([X[c].value_counts().index[0]
                                   if X[c].dtype == np.dtype('O') else X[c].mean() for c in X],
                                  index=X.columns)

        elif self.fill_type == "const":
            self.fill = pd.Series(["-1" if X[c].dtype == np.dtype('O') else -1 for c in X],
                                  index=X.columns)

        return self

    def transform(self, X, y=None):
        return X.fillna(self.fill)
