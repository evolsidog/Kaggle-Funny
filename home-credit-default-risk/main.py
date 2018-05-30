from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from utils import split_num_str_data, drop_nan_by_thresh, estimate_auc, report_classification
from sklearn.model_selection import train_test_split, GridSearchCV
from imblearn.over_sampling import SMOTE, ADASYN
import pandas as pd
import random
import os
import time

ini_time = time.time()

SEED = 1234
TEST_SIZE = 0.2
NJOBS = 4
TARGET = "TARGET"
SK_ID_CURR = "SK_ID_CURR"
USER = "vic"

random.seed(SEED)

# TODO Improves
# Quitar oversampling y probar random forest con balanceo -> hecho, empeora mucho, tanto con "balanced" como metiendo a mano los pesos
# Probar best params y entrenar con columnas relevantes -> hecho, al no coger todas las columnas mejora un poco pero al tener el balanceo anterior empeora
# Train with 80 - 20 -> Hecho
# Check features importance con el oversampling original de Nuno y ver si mejora un poco al quitarle features innecesarias -> Hecho
# isolated forest
# Probar xgboost

base_path = "/home/" + USER + "/.kaggle/competitions/home-credit-default-risk"
print("Loading files")
df_app_train = pd.read_csv(os.path.join(base_path, "application_train.csv"))
df_app_test = pd.read_csv(os.path.join(base_path, "application_test.csv"))

print("------------- Initial preprocessing ---------------")
print("Positive class data:", sum(df_app_train[TARGET]))
print("Total data :", df_app_train[TARGET].count())

pos_weight = sum(df_app_train[TARGET]) / df_app_train[TARGET].count() * 100
print("Positive class data percentage:", pos_weight)

# Split numeric and non numeric data
df_app_train_num, df_app_train_str = split_num_str_data(df_app_train)
print("Columns before dropping column nan's", df_app_train_num.shape[1])

df_app_train_num = drop_nan_by_thresh(df=df_app_train_num, thresh=0.80)
print("Columns after dropping column nan's", df_app_train_num.shape[1])
print("Rows before dropping row nan's ", df_app_train_num.shape[0])

df_app_train_num = df_app_train_num.dropna(axis=0)
print("Rows after dropping row nan's ", df_app_train_num.shape[0])

# Separate predicted variables
X = df_app_train_num.loc[:, ~ df_app_train_num.columns.isin([TARGET, SK_ID_CURR])]
y = df_app_train_num[TARGET]

print("------------- After preprocessing ---------------")
print("Positive class data:", sum(df_app_train_num[TARGET]))
print("Total data :", df_app_train_num[TARGET].count())

pos_weight_after = sum(df_app_train_num[TARGET]) / df_app_train_num[TARGET].count() * 100
print("Positive class data percentage:", pos_weight_after)
# y_weight = int(round(100 / pos_weight_after))
# print("Weight for unbalancing class:", y_weight)

X_resampled, y_resampled = SMOTE(random_state=SEED, kind='regular').fit_sample(X, y)
print("Train Size before resampling", X_resampled.shape[0])
print("Positive class samples percentage after resampling", sum(y_resampled) / len(y_resampled))

X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=TEST_SIZE,
                                                    random_state=SEED, stratify=y_resampled)
print("Train Size after resampling", X_train.shape[0])

clf = RandomForestClassifier(n_jobs=NJOBS,
                             random_state=SEED)

# Try 1: --> n_estimators = 150, min_samples_lead = 10, max_depth=15
# param_grid = {
#     "n_estimators": [50, 150],
#     "max_depth": [5, 15],
#     "min_samples_leaf": [10, 40]}

# Try 2 --> n_estimators = 150, min_samples_lead = 10, max_depth=15
# param_grid = {
#     "n_estimators": [100, 150],
#     "max_depth": [10, 15],
#     "min_samples_leaf": [10, 20]}

# Try 3 --> n_estimators = 200, min_samples_lead = 5, max_depth=20
# param_grid = {
#     "n_estimators": [150, 200],
#     "max_depth": [15, 20],
#     "min_samples_leaf": [5, 10]}

# Try 4 (cv=5) --> n_estimators = 300, min_samples_lead = 3, max_depth=25, bootstrap=False, min_samples_split=5
param_grid = {
    "n_estimators": [200, 300],
    "max_depth": [20, 25],
    "min_samples_leaf": [3, 5],
    'min_samples_split': [5, 10],
    'bootstrap': [True, False]
}

'''
cv_rfc = GridSearchCV(estimator=clf, param_grid=param_grid, cv=5, verbose=10)
cv_rfc.fit(X_train, y_train)
print("Best hiperparameters after grid search: ", cv_rfc.best_estimator_.get_params())
# Best hiperparameters after grid search:  {'bootstrap': True, 'class_weight': None, 'criterion': 'gini', 'max_depth': 20, 'max_features': 'auto', 'max_leaf_nodes': None, 'min_impurity_decrease': 0.0, 'min_impurity_split': None, 'min_samples_leaf': 5, 'min_samples_split': 2, 'min_weight_fraction_leaf': 0.0, 'n_estimators': 200, 'n_jobs': -1, 'oob_score': False, 'random_state': 1234, 'verbose': 0, 'warm_start': False}

columns = list(X.columns)
feature_importances = cv_rfc.best_estimator_.feature_importances_
print("Most important features are: ", sorted(zip(map(lambda x: round(x, 4), feature_importances), columns), reverse=True))
params = {k: cv_rfc.best_estimator_.get_params()[k] for k in param_grid}
'''
'''
[(0.1102, 'FLAG_DOCUMENT_3'), (0.1067, 'FLAG_PHONE'), (0.0991, 'EXT_SOURCE_3'), (0.0835, 'EXT_SOURCE_2'),
(0.0696, 'FLAG_WORK_PHONE'), (0.06, 'REG_CITY_NOT_WORK_CITY'), (0.0514, 'CNT_FAM_MEMBERS'),
(0.0448, 'REGION_POPULATION_RELATIVE'), (0.044, 'REGION_RATING_CLIENT'), (0.0407, 'LIVE_CITY_NOT_WORK_CITY'),
(0.0401, 'DAYS_LAST_PHONE_CHANGE'), (0.0392, 'REGION_RATING_CLIENT_W_CITY'), (0.0378, 'AMT_REQ_CREDIT_BUREAU_MON'),
(0.033, 'AMT_REQ_CREDIT_BUREAU_QRT'), (0.0302, 'AMT_REQ_CREDIT_BUREAU_YEAR'), (0.0256, 'CNT_CHILDREN'),
(0.02, 'FLAG_DOCUMENT_8'), (0.0198, 'REG_CITY_NOT_LIVE_CITY'), (0.0141, 'OBS_30_CNT_SOCIAL_CIRCLE'),
(0.0139, 'OBS_60_CNT_SOCIAL_CIRCLE'), (0.0097, 'DEF_30_CNT_SOCIAL_CIRCLE'), (0.0066, 'DEF_60_CNT_SOCIAL_CIRCLE')]
'''

clf_opt = RandomForestClassifier(n_jobs=NJOBS, random_state=SEED, bootstrap=False, max_depth=25, min_samples_leaf=3,
                                 min_samples_split=5, n_estimators=300)
print("Fitting model ... ")
clf_opt.fit(X_train, y_train)

print("Model fitted. Predicting ...")
y_pred_test = clf_opt.predict(X_test)
y_pred_train = clf_opt.predict(X_train)

'''
clf.fit(X=X_train, y=y_train)
'''

estimate_auc(y_pred=y_pred_train, y_test=y_train)
estimate_auc(y_pred=y_pred_test, y_test=y_test)

report_classification(y_test=y_train, predicted=y_pred_train)
report_classification(y_test=y_test, predicted=y_pred_test)

# Predict on test data
print("Predict on test data")
# Keep the same columns
df_app_test = df_app_test.loc[:, df_app_test.columns.isin(df_app_train_num.columns)]

# Fill with mean
df_app_test = df_app_test.apply(lambda x: x.fillna(x.mean()), axis=0)

y_pred_test = clf_opt.predict_proba(df_app_test.loc[:, df_app_test.columns != SK_ID_CURR])
sk_id_curr = df_app_test[SK_ID_CURR]
y_pred_test_flat = list(map(lambda x: x[1], y_pred_test))

print("Generating kaggle submision")
kaggle_submission = pd.DataFrame({TARGET: y_pred_test_flat, SK_ID_CURR: sk_id_curr})
kaggle_submission.to_csv("submission.csv", index=False)

'''
CV_rfc.fit(X=X_resampled, y=y_resampled)
y_pred = CV_rfc.predict(X_test)
fpr, tpr, thresholds = metrics.roc_curve(y_true=y_test, y_score=y_pred, pos_label=1)
print("AUC ", metrics.auc(x=fpr, y=tpr))
'''
print("Total seconds = ", time.time() - ini_time)
