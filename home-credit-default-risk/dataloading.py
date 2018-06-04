# -*- coding: utf-8 -*-

import pandas as pd
import gc
import os
import time

from utils import mini_profiling, clean_df

ini_time = time.time()

TARGET = "TARGET"
SK_ID_CURR = "SK_ID_CURR"
USER = "ncarvalho"
THRESH = 0.75
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
df_bureau_bal = df_bureau_bal.groupby('SK_ID_BUREAU').agg(['sum', 'mean', 'std', 'min']).reset_index('SK_ID_BUREAU')
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
df_bureau = df_bureau.groupby(['SK_ID_CURR', 'SK_ID_BUREAU']).agg(['sum', 'mean', 'std']).reset_index()

print("Merge bureau and bureau balance")
# TODO After merge, int columns convert to float columns. Bug in pandas
df_bureau_merge = df_bureau.merge(right=df_bureau_bal, how='left', on='SK_ID_BUREAU')
print(df_bureau_merge.head(10))

# Now group by sk_id_curr to prepare merge with train and test files
df_bureau_merge = df_bureau_merge.drop('SK_ID_BUREAU', axis=1)
df_bureau_merge = df_bureau_merge.groupby(['SK_ID_CURR']).agg(['sum', 'mean', 'std', 'min']).reset_index()


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

print("------- Generating new train and test files for Bureau Data ---------")
df_app_train_merge.to_csv("merge_application_train.csv")
df_app_test_merge.to_csv("merge_application_test.csv")

del df_app_train_merge, df_app_test_merge
gc.collect()




print("------ Reading credit_card_balance.csv ------")


df_credit = pd.read_csv(os.path.join(base_path, "credit_card_balance.csv"))
mini_profiling(df_credit, "credit_card_balance")

print("------ Reading installments_payments.csv ------")

df_installments_payments = pd.read_csv(os.path.join(base_path, "installments_payments.csv"))

print("------------------------Group both by SK_ID_CURR ----------------------")

df_installments_payments_grouped = df_installments_payments.groupby([SK_ID_CURR]).agg(['mean', 'sum', 'std']).reset_index()
df_credit_grouped = df_credit.groupby([SK_ID_CURR]).agg(['mean', 'sum', 'std', 'min']).reset_index()

print("------------------------ Merge credit_card_balance and installments_payments ----------------------")

df_merge = df_installments_payments_grouped.merge(df_credit_grouped, how="left", on='SK_ID_CURR').reset_index()

print("------------------------ Reade merge_application_train.csv ----------------------")

df_app_train = pd.read_csv("merge_application_train.csv")

print("------------------------ Reade merge_application_test.csv ----------------------")

df_app_test = pd.read_csv("merge_application_test.csv")

print("Test size before merge", df_app_train.shape)
print("Train size before merge", df_app_test.shape)

df_app_train_merge = df_merge.merge(df_app_train, how="right", on='SK_ID_CURR').reset_index()
df_app_test_merge = df_merge.merge(df_app_test, how="right", on='SK_ID_CURR').reset_index()

print("Test size after merge", df_app_test_merge.shape)
print("Train size after merge", df_app_train_merge.shape)

df_app_train_merge.to_csv("merge_application_train.csv")
df_app_test_merge.to_csv("merge_application_test.csv")

del df_app_train, df_app_test, df_app_test_merge, df_app_train_merge
del df_credit_grouped, df_credit, df_installments_payments, df_installments_payments_grouped
gc.collect()

print(df_merge.head(10))


print("------ Reading POS_CASH_balance.csv ------")


posh_cash_balance = pd.read_csv(os.path.join(base_path, "POS_CASH_balance.csv"))

print("------ Reading previous_application.csv ------")

df_previous_application = pd.read_csv(os.path.join(base_path, "previous_application.csv"))


posh_cash_balance_grouped = posh_cash_balance.groupby([SK_ID_CURR]).agg(['mean', 'sum', 'std', 'min']).reset_index()
df_credit_grouped = df_previous_application.groupby([SK_ID_CURR]).agg(['mean', 'sum', 'std', 'min']).reset_index()


del posh_cash_balance, df_previous_application
gc.collect()



df_merge = df_credit_grouped.merge(posh_cash_balance_grouped, how="left", on='SK_ID_CURR').reset_index()


print("------------------------ Reade merge_application_train.csv ----------------------")

df_app_train = pd.read_csv("merge_application_train.csv")

print("------------------------ Reade merge_application_test.csv ----------------------")

df_app_test = pd.read_csv("merge_application_test.csv")

print("Test size before merge", df_app_train.shape)
print("Train size before merge", df_app_test.shape)

df_app_train_merge = df_merge.merge(df_app_train, how="right", on='SK_ID_CURR').reset_index()
df_app_test_merge = df_merge.merge(df_app_test, how="right", on='SK_ID_CURR').reset_index()


print("Test size after merge", df_app_test_merge.shape)
print("Train size after merge", df_app_train_merge.shape)

df_app_train_merge.to_csv("merge_application_train.csv")
df_app_test_merge.to_csv("merge_application_test.csv")

print("Total seconds = ", time.time() - ini_time)




