# -*- coding: utf-8 -*-
import pandas as pd
import time
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from scipy.stats import spearmanr
from sklearn import ensemble, model_selection
import lightgbm as lgb

USER = "vic"
BASE_PATH = "/home/" + USER + "/.kaggle/competitions/santander-value-prediction"
desired_columns = 50
desired_width = 280
pd.set_option('display.max_columns', desired_columns)
pd.set_option('display.width', desired_width)


def get_profiling(df):
    print("--- Info---: ")
    print(df.info())
    print("Shape: ", df.shape)
    print("Head: ", df.head(5))
    print("Nulls: ", df.isnull().values.any())
    print("Describe: ", df.describe())


def get_missing_values(df):
    missing_df = df.isnull().sum(axis=0).reset_index()
    missing_df.columns = ['column_name', 'missing_count']
    missing_df = missing_df[missing_df['missing_count'] > 0]
    missing_df = missing_df.sort_values(by='missing_count')

    return missing_df


def plot_histogram(serie, name=None):
    plt.figure(figsize=(12, 5))
    # plt.hist(serie, bins=100)
    sns.distplot(serie, bins=100, kde=False)
    plt.title('Histogram target counts')
    plt.xlabel('Count')
    plt.ylabel('Target')
    plt.savefig(name + '.png')


def plot_log_histogram(serie, name=None):
    plt.figure(figsize=(12, 5))
    # plt.hist(np.log(1+serie), bins=100)
    sns.distplot(np.log1p(serie), bins=100, kde=False)
    plt.title('Histogram target counts')
    plt.xlabel('Count')
    plt.ylabel('Log 1+Target')
    plt.savefig('log_' + name + '.png')


def plot_scatter(x, serie, name=None):
    plt.figure(figsize=(12, 5))
    plt.scatter(range(x), np.sort(serie))
    plt.xlabel('index', fontsize=12)
    plt.ylabel('Target', fontsize=12)
    plt.title(name + " distribution", fontsize=14)
    plt.savefig("Scatter " + name + '.png')


def run_lgb(train_X, train_y, val_X, val_y, test_X):
    params = {
        "objective": "regression",
        "metric": "rmse",
        "num_leaves": 30,
        "learning_rate": 0.01,
        "bagging_fraction": 0.7,
        "feature_fraction": 0.7,
        "bagging_frequency": 5,
        "bagging_seed": 2018,
        "verbosity": -1
    }

    lgtrain = lgb.Dataset(train_X, label=train_y)
    lgval = lgb.Dataset(val_X, label=val_y)
    evals_result = {}
    model = lgb.train(params, lgtrain, 1000, valid_sets=[lgval], early_stopping_rounds=100, verbose_eval=200,
                      evals_result=evals_result)

    pred_test_y = model.predict(test_X, num_iteration=model.best_iteration)
    return pred_test_y, model, evals_result


ini_time = time.time()

train_path = os.path.join(BASE_PATH, 'train.csv')
test_path = os.path.join(BASE_PATH, 'test.csv')

print("----------- Loading train and test sets -----------")
train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')

# EDA from https://www.kaggle.com/tunguz/yaeda-yet-another-eda and
# https://www.kaggle.com/sudalairajkumar/simple-exploration-baseline-santander-value

print("----------------------- EDA -------------------------------")
print("------------- PROFILING TRAIN DATAFRAME ----------------")
get_profiling(train_df)
print("------------- PROFILING TEST DATAFRAME ----------------")
get_profiling(test_df)

# In train set there are 1845 float, 3147 int and 1 object columns, but in the test set all columns are float except one

print("------------- PLOT TARGET ----------------")
plot_histogram(train_df.target.values, name='train_target')
plot_log_histogram(train_df.target.values, name='train_target')

print("--------- Log Target -----------")
train_log_target_df = pd.DataFrame()
train_log_target_df['target'] = np.log(1 + train_df['target'].values)
print(train_log_target_df.describe())

print("------ Scatter target --------")
plot_scatter(train_df.shape[0], train_df.target.values, name='train_target')

print("--------- Get missing values --------")
missing_values_df = get_missing_values(train_df)
print(missing_values_df.head(5))

# There aren't null values

print("--------- Get columns without variance --------")
unique_df = train_df.nunique().reset_index()
unique_df.columns = ['col_name', 'unique_count']
constant_df = unique_df[unique_df['unique_count'] == 1]
print("Number or constants columns: ", constant_df.shape[0])

print("----------- Correlation features with target -----------")
#  it might be a good idea to use Spearman correlation inplace of pearson since spearman is computed on ranks and so
# depicts monotonic relationships while pearson is on true values and depicts linear relationships.
labels = []
values = []
for col in train_df.columns:
    if col not in ["ID", "target"]:
        labels.append(col)
        values.append(spearmanr(train_df[col].values, train_df["target"].values)[0])
corr_df = pd.DataFrame({'col_labels': labels, 'corr_values': values})
corr_df = corr_df.sort_values(by='corr_values')

corr_df = corr_df[(corr_df['corr_values'] > 0.1) | (corr_df['corr_values'] < -0.1)]
ind = np.arange(corr_df.shape[0])
width = 0.9
fig, ax = plt.subplots(figsize=(12, 30))
rects = ax.barh(ind, np.array(corr_df.corr_values.values), color='b')
ax.set_yticks(ind)
ax.set_yticklabels(corr_df.col_labels.values, rotation='horizontal')
ax.set_xlabel("Correlation coefficient")
ax.set_title("Correlation coefficient of the variables")
plt.savefig("Correlation_coeficients.png")

print("---------- Correlation heat map ----------")
# Now let us take these variables whose absolute value of correlation with the target is greater than 0.11
# (just to reduce the number of features fuether) and do a correlation heat map.
cols_to_use = corr_df[(corr_df['corr_values'] > 0.11) | (corr_df['corr_values'] < -0.11)].col_labels.tolist()

temp_df = train_df[cols_to_use]
corrmat = temp_df.corr(method='spearman')
f, ax = plt.subplots(figsize=(20, 20))

# Draw the heatmap using seaborn
sns.heatmap(corrmat, vmax=1., square=True, cmap="YlGnBu", annot=True)
plt.title("Important variables correlation map", fontsize=15)
plt.savefig("Correlation_Heat_Map.png")

# Seems like none of the selected variables have spearman correlation more than 0.7 with each other.
# The generated previous plots helped us in identifying the important individual variables which are correlated with target.

print("------------- Feature importance -------------")
print("------ Extra trees model ---------")
# Our Evaluation metric for the competition is RMSLE. So let us use log of the target variable to build our models.
train_X = train_df.drop(constant_df.col_name.tolist() + ["ID", "target"], axis=1)
test_X = test_df.drop(constant_df.col_name.tolist() + ["ID"], axis=1)
train_y = np.log1p(train_df["target"].values)

model = ensemble.ExtraTreesRegressor(n_estimators=200, max_depth=20, max_features=0.5, n_jobs=-1, random_state=0)
model.fit(train_X, train_y)

# plot the importances
feat_names = train_X.columns.values
importances = model.feature_importances_
std = np.std([tree.feature_importances_ for tree in model.estimators_], axis=0)
indices = np.argsort(importances)[::-1][:20]

plt.figure(figsize=(12, 12))
plt.title("Feature importances")
plt.bar(range(len(indices)), importances[indices], color="r", yerr=std[indices], align="center")
plt.xticks(range(len(indices)), feat_names[indices], rotation='vertical')
plt.xlim([-1, len(indices)])
plt.savefig("Extra Trees Features importances.png")

# 'f190486d6' seems to be the important variable followed by '58e2e02e6'

print("------ Light GBM ------")
kf = model_selection.KFold(n_splits=5, shuffle=True, random_state=2017)
pred_test_full = 0
for dev_index, val_index in kf.split(train_X):
    dev_X, val_X = train_X.loc[dev_index, :], train_X.loc[val_index, :]
    dev_y, val_y = train_y[dev_index], train_y[val_index]
    pred_test, model, evals_result = run_lgb(dev_X, dev_y, val_X, val_y, test_X)
    pred_test_full += pred_test
pred_test_full /= 5.
pred_test_full = np.expm1(pred_test_full)

# Feature Importance
fig, ax = plt.subplots(figsize=(12,18))
lgb.plot_importance(model, max_num_features=50, height=0.8, ax=ax)
ax.grid(False)
plt.title("LightGBM - Feature Importance", fontsize=15)
plt.savefig("LightGBM Features importances.png")

# Making a submission file
sub_df = pd.DataFrame({"ID": test_df["ID"].values})
sub_df["target"] = pred_test_full
sub_df.to_csv("lgb.csv", index=False)

print("Time: ", time.time() - ini_time)
