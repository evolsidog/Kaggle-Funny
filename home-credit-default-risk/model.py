from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from utils import split_num_str_data, drop_nan_by_thresh, estimate_auc, report_classification, preprocess_test_set,\
    deal_with_nan, encode_categorical_variables, get_most_important_features, get_important_features
from sklearn.model_selection import train_test_split, GridSearchCV
from imblearn.over_sampling import SMOTE, ADASYN, RandomOverSampler
from xgboost import XGBClassifier
import pandas as pd
import random
import os
import time

# TODO
# 0 - Mirar algun cruce de tablas por el sk_curr_id
# 1 - Categoricas que son False y True pasarlas a unos y ceros
# 2 - Categoricas que sean cadenas (One hot Encoding?)
# 3 - Una vez mirado lo anterior, quitamos lo nulos(eliminar filas y columnas).
# 4 - Despues imputamos(moda, media, KNN imputer? Revisar con cuidado cuando son cadenas)
# 5 - Usar RFC para sacar la importancia de las features.
# 6 - Quitar las no importantes
# 6.5 - Sacar hiperparametros(si puede ser en amazon)
# 7 - Usar IsolationForest
# 8 - Investigar como obtener el vector de probabilidades(hay un método?¿?)
# 9 - Grid Search
# 10 - Usar XGBoost
# 11 - Quitar variables correladas


ini_time = time.time()

SEED = 1234
TEST_SIZE = 0.2
THRESH = 0.8
NJOBS = -1
TARGET = "TARGET"
SK_ID_CURR = "SK_ID_CURR"
USER = "ncarvalho"

random.seed(SEED)

# TODO Improves
# Quitar oversampling y probar random forest con balanceo -> hecho, empeora mucho, tanto con "balanced" como metiendo a mano los pesos
# Probar best params y entrenar con columnas relevantes -> hecho, al no coger todas las columnas mejora un poco pero al tener el balanceo anterior empeora
# Train with 80 - 20 -> Hecho
# Check features importance con el oversampling original de Nuno y ver si mejora un poco al quitarle features innecesarias -> Hecho
# isolated forest -> no tiene predict_proba y devuelve 1 y -1. Mucho trabajo adaptarlo.
# Probar xgboost -> tampoco mejora mucho y el predict_proba hay que cambiar los argumentos.
# Revisar las metricas tras corregir lo del resampleo del test, en kaggle bien pero en local de pena.

base_path = "/home/" + USER + "/.kaggle/competitions/home-credit-default-risk"
print("Loading files")
df_app_train = pd.read_csv(os.path.join(base_path, "application_train.csv"))
df_app_test = pd.read_csv(os.path.join(base_path, "application_test.csv"))

print("------------- Initial preprocessing ---------------")
print("Positive class data:", sum(df_app_train[TARGET]))
print("Total data :", df_app_train[TARGET].count())

pos_weight = sum(df_app_train[TARGET]) / df_app_train[TARGET].count() * 100
print("Positive class data percentage:", pos_weight)


print("Columns before dropping column nan's", df_app_train.shape[1])

# Removes columns
df_app_train = drop_nan_by_thresh(df=df_app_train, thresh=THRESH, axis=1)

print("Columns after dropping column nan's", df_app_train.shape[1])


print("Rows before dealing with nan ",  df_app_train.shape[0])
df_app_train = deal_with_nan(df=df_app_train, action="drop_rows")

print("Rows after dealing with nan ", df_app_train.shape[0])

# Split numeric and non numeric data
df_app_train_num, df_app_train_str = split_num_str_data(df_app_train)

print("Encoding categorical variables")
df_app_train_str = encode_categorical_variables(df_app_train_str)
df_app_train = pd.concat([df_app_train_num, df_app_train_str], axis=1)

del df_app_train_num, df_app_train_str

print("Removing variables with no variance")
# Remove columns with no variance
df_app_train = df_app_train.loc[:,df_app_train.apply(pd.Series.nunique) != 1]


print("Separating predicter variables from target variable")
# Separate predicter variables and target variable
X = df_app_train.loc[:, ~ df_app_train.columns.isin([TARGET, SK_ID_CURR])]
y = df_app_train[TARGET]


print("Train/Test split")





X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE,
                                                    random_state=SEED, stratify=y)


print("Train size before resampling", X_train.shape[0])


print("SMOTE Oversampling ..") # 0.761
#X_resampled, y_resampled = SMOTE(random_state=SEED, kind='regular').fit_sample(X_train, y_train)
X_resampled, y_resampled = ADASYN(random_state=SEED).fit_sample(X_train, y_train)
pos_weight_after = sum(y_resampled) / y_resampled.size * 100

print("Positive class samples percentage after resampling", pos_weight_after)


print("Train Size after resampling", X_resampled.shape[0])

'''
clf_opt = RandomForestClassifier(n_jobs=NJOBS,
                                 random_state=SEED,
                                 bootstrap=False,
                                 max_depth=6,
                                 min_samples_leaf=3,
                                 min_samples_split=5,
                                 n_estimators=200,
                                 verbose=2)
clf_opt = GradientBoostingClassifier(random_state=SEED,
                                     max_depth=7,
                                     min_samples_leaf=3,
                                     min_samples_split=5,
                                     subsample=0.5,
                                     learning_rate=0.1,
                                     n_estimators=100,
                                     verbose=2)
'''



clf_opt = XGBClassifier(random_state=SEED,
                        max_depth=4,
                        min_samples_leaf=3,
                        min_samples_split=2,
                        n_estimators=100,
                        learning_rate=0.1,
                        subsample=0.5,
                        objective="binary:logistic",
                        silent=False)

print("Fitting model for variable importance... ")
clf_opt.fit(X_train, y_train)

f_i = get_important_features(clf=clf_opt, columns=X_train.columns)

f_m_i = get_most_important_features(f_i)

print("Model fitted. Predicting ...")
X_test = X_test.loc[:, f_m_i]
X_train = X_train.loc[:, f_m_i]


clf_opt = XGBClassifier(random_state=SEED,
                        max_depth=4,
                        min_samples_leaf=3,
                        min_samples_split=2,
                        n_estimators=100,
                        learning_rate=0.1,
                        subsample=0.5,
                        objective="binary:logistic",
                        silent=False)

print("Fitting model ... ")
clf_opt.fit(X_train, y_train)



print("Model fitted. Predicting ...")
y_pred_test = clf_opt.predict_proba(X_test)
y_pred_train = clf_opt.predict_proba(X_train)

# Get the second element of each list. The second element is the probability of label 1.
y_pred_test = list(map(lambda x: x[1], y_pred_test)) # [[0.14, 0.86],[0.23, 0.77],[0.35, 0.65]] -> [0.86, 0.77, 0.65]
y_pred_train = list(map(lambda x: x[1], y_pred_train)) # [[0.14, 0.86],[0.23, 0.77],[0.35, 0.65]] -> [0.86, 0.77, 0.65]

estimate_auc(y_pred=y_pred_train, y_test=y_train, name="Train")
estimate_auc(y_pred=y_pred_test, y_test=y_test, name="Test")

report_classification(y_test=y_train, predicted=clf_opt.predict(X_train))
report_classification(y_test=y_test, predicted=clf_opt.predict(X_test))

# Predict on test data

print("Prepare test data for prediction")


# Get the ID column
sk_id_curr = df_app_test[SK_ID_CURR]

df_app_test = preprocess_test_set(df_train=X_train, df_test=df_app_test)

print("Predict on test data")


# Remove the unique ID(PK) from the test data
df_app_test = df_app_test.loc[:, df_app_test.columns != SK_ID_CURR]

# Predict
y_pred_test = clf_opt.predict_proba(df_app_test[f_m_i])

# This line does this:  [[0.14, 0.86],[0.23, 0.77],[0.35, 0.65]] -> [0.86, 0.77, 0.65]
y_pred_test_prob = list(map(lambda x: x[1], y_pred_test))

print("Generating kaggle submision")
kaggle_submission = pd.DataFrame({TARGET: y_pred_test_prob, SK_ID_CURR: sk_id_curr})
kaggle_submission.to_csv("submission.csv", index=False)


print("Total seconds = ", time.time() - ini_time)