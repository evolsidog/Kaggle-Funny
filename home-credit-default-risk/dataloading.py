# -*- coding: utf-8 -*-

import pandas as pd
import gc
import os
import time

from utils import mini_profiling, clean_df

ini_time = time.time()

TARGET = "TARGET"
SK_ID_CURR = "SK_ID_CURR"
USER = "vic"
THRESH = 0.8
desired_width = 250
pd.set_option('display.width', desired_width)
gc.enable()

# TODO
# 1. Try with getdummies and concat instead of label encoder

base_path = "/home/" + USER + "/.kaggle/competitions/home-credit-default-risk"
print("--------- Starting data loading -------------")

print("Reading train.csv")
df_app_train = pd.read_csv(os.path.join(base_path, "application_train.csv"))
users_default = df_app_train[df_app_train[TARGET] == 1][SK_ID_CURR]

del df_app_train
gc.collect()

print("-------------- Loading data bureau files ---------------")

print("------ Reading bureau_balance.csv ------")
df_bureau_bal = pd.read_csv(os.path.join(base_path, "bureau_balance.csv"))
mini_profiling(df_bureau_bal, "bureau_balance")
print("Get dummies bureau balance")
bureau_bal_status_dum = pd.get_dummies(df_bureau_bal['STATUS'], prefix='BUREAU_BAL_STATUS')
df_bureau_bal = pd.concat([df_bureau_bal, bureau_bal_status_dum], axis=1).drop('STATUS', axis=1)
del bureau_bal_status_dum
gc.collect()

print("Cleaning bureau balance")
df_bureau_bal = clean_df(df_bureau_bal, THRESH)
print(df_bureau_bal.head(10))

print("Processing bureau balance")
df_bureau_bal = df_bureau_bal.drop(columns=['MONTHS_BALANCE'])
df_bureau_bal = df_bureau_bal.groupby('SK_ID_BUREAU').sum().reset_index('SK_ID_BUREAU')
print(df_bureau_bal.head(10))

print("------ Reading bureau.csv ------")
df_bureau = pd.read_csv(os.path.join(base_path, "bureau.csv"))
mini_profiling(df_bureau, "bureau")
print("Get dummies bureau")
bureau_credit_active_dum = pd.get_dummies(df_bureau['CREDIT_ACTIVE'], prefix='BUREAU_CREDIT_ACTIVE')
bureau_credit_currency_dum = pd.get_dummies(df_bureau['CREDIT_CURRENCY'], prefix='BUREAU_CREDIT_CURRENCY')
bureay_credit_type_dum = pd.get_dummies(df_bureau['CREDIT_TYPE'], prefix='BUREAU_CREDIT_TYPE')
df_bureau = pd.concat([df_bureau, bureau_credit_active_dum, bureau_credit_currency_dum, bureay_credit_type_dum],
                      axis=1).drop(columns=['CREDIT_ACTIVE', 'CREDIT_CURRENCY', 'CREDIT_TYPE'])
del bureau_credit_active_dum, bureau_credit_currency_dum, bureay_credit_type_dum
gc.collect()

print("Cleaning bureau")
df_bureau = clean_df(df_bureau, THRESH)
print(df_bureau.head(10))

print("Processing bureau")
df_bureau = df_bureau.groupby(['SK_ID_CURR', 'SK_ID_BUREAU']).sum().reset_index()

print("Merge bureau and bureau balance")
# TODO After merge, int columns convert to float columns. Bug in pandas
df_bureau_merge = df_bureau.merge(right=df_bureau_bal, how='left', on='SK_ID_BUREAU')
print(df_bureau_merge.head(10))

# Now group by sk_id_curr to prepare merge with train and test files
df_bureau_merge = df_bureau_merge.drop('SK_ID_BUREAU', axis=1)
df_bureau_merge = df_bureau_merge.groupby(['SK_ID_CURR']).sum().reset_index()


del df_bureau_bal, df_bureau
gc.collect()

print("------- Join intermediate tables with train and test --------")
print("Loading data and test files")
df_app_train = pd.read_csv(os.path.join(base_path, "application_train.csv"))
df_app_test = pd.read_csv(os.path.join(base_path, "application_test.csv"))

print("Columns train before merging", df_app_train.shape[1])
df_app_train_merge = df_app_train.merge(right=df_bureau_merge.reset_index(), how='left', on='SK_ID_CURR')
print("Columns train after merging", df_app_train_merge.shape[1])

print("Columns test before merging", df_app_test.shape[1])
df_app_test_merge = df_app_test.merge(right=df_bureau_merge.reset_index(), how='left', on='SK_ID_CURR')
print("Columns train after merging", df_app_test_merge.shape[1])

del df_app_train, df_app_test
gc.collect()

print("------- Generating new train and test files ---------")
df_app_train_merge.to_csv("merge_application_train.csv")
df_app_test_merge.to_csv("merge_application_test.csv")

del df_app_train_merge, df_app_test_merge
gc.collect()

print("Total seconds = ", time.time() - ini_time)
