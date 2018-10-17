# -*- coding: utf-8 -*-
from sklearn import ensemble
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.model_selection import cross_val_score
import os
import time
import pandas_profiling

SEED = 1234
USER = "victor"
PATH = os.path.join(os.path.sep, "home", USER, ".kaggle", "house-prices-advanced-regression-techniques")
# ------ Load train and test ----- #

ini_time = time.time()
train_path = os.path.join(PATH, 'train.csv')
test_path = os.path.join(PATH, 'test.csv')

train_df = pd.read_csv(train_path, parse_dates=True, encoding='UTF-8')
test_df = pd.read_csv(test_path, parse_dates=True, encoding='UTF-8')

# ------ EDA ------ #
print(train_df.describe())
print(train_df.corr())
print(train_df.shape)

profile = pandas_profiling.ProfileReport(train_df)
profile.to_file(outputfile="profiling_V1.html")
rejected_variables = profile.get_rejected_variables(threshold=0.7)
print(rejected_variables)

# ------ Workaround ----- #
numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
fake_train_df = train_df.select_dtypes(include=numerics)
fake_train_df = fake_train_df.fillna(method='pad')
print(fake_train_df.shape)


Y = fake_train_df['SalePrice']
x = fake_train_df.drop('SalePrice', axis=1)
# x_train, x_test, y_train, y_test = train_test_split(x, Y, test_size=0.3, random_state=SEED)

# ------ Model -------#
model = ensemble.AdaBoostRegressor(n_estimators=100, learning_rate=2, random_state=SEED)
# model.fit(x_train, y_train)



evaluation = cross_val_score(model, x, Y)
print(evaluation)

end_time = time.time()
print("TIME: " + str(end_time - ini_time))

