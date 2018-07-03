# -*- coding: utf-8 -*-
import pandas as pd
import time
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

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
    missing_df = missing_df[missing_df['missing_count']>0]
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
    #plt.hist(np.log(1+serie), bins=100)
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


ini_time = time.time()

train_path = os.path.join(BASE_PATH,'train.csv')
test_path = os.path.join(BASE_PATH,'test.csv')

print("----------- Loading train and test sets -----------")
train_df = pd.read_csv(train_path)
test_df = pd.read_csv(test_path)

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
train_log_target_df['target'] = np.log(1+train_df['target'].values)
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
print("Remove this columns from train and test")

train_df = train_df.drop(constant_df.col_name.tolist(), axis=1)
test_df = test_df.drop(constant_df.col_name.tolist(), axis=1)
print("Train shape now: ", train_df.shape[0])

print("----------- Correlation features with target -----------")
print("Time: ", time.time() - ini_time)


