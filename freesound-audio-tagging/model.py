from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
import pandas as pd
import random
import os
import time
import numpy as np
import os
import shutil

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from tqdm import tqdm_notebook
from sklearn.cross_validation import StratifiedKFold
import wave
from scipy.io import wavfile

# TODO
# 0 - Mirar algun cruce de tablas por el sk_curr_id
# 1 - Categoricas que son False y True pasarlas a unos y ceros
# 2 - Categoricas que sean cadenas (One hot Encoding?)
# 3 - Una vez mirado lo anterior, quitamos lo nulos(eliminar filas y columnas).
# 4 - Despues imputamos(moda. media? Revisar con cuidado cuando son cadenas)
# 5 - Usar RFC para sacar la importancia de las features.
# 6 - Quitar las no importantes
# 6.5 - Sacar hiperparametros
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
USER = "crosado"

random.seed(SEED)

# TODO Improves
# Quitar oversampling y probar random forest con balanceo -> hecho, empeora mucho, tanto con "balanced" como metiendo a mano los pesos
# Probar best params y entrenar con columnas relevantes -> hecho, al no coger todas las columnas mejora un poco pero al tener el balanceo anterior empeora
# Train with 80 - 20 -> Hecho
# Check features importance con el oversampling original de Nuno y ver si mejora un poco al quitarle features innecesarias -> Hecho
# isolated forest -> no tiene predict_proba y devuelve 1 y -1. Mucho trabajo adaptarlo.
# Probar xgboost -> tampoco mejora mucho y el predict_proba hay que cambiar los argumentos.
# Revisar las metricas tras corregir lo del resampleo del test, en kaggle bien pero en local de pena.

base_path = "/home/" + USER + "/.kaggle/competitions/freesound-audio-tagging"
print("Loading files")
df_app_train = pd.read_csv(os.path.join(base_path, "train.csv"))
df_app_test = pd.read_csv(os.path.join(base_path, "sample_submission.csv"))

matplotlib.style.use('ggplot')


print("Number of training examples=", df_app_train.shape[0], "  Number of classes=", len(df_app_train.label.unique()))

print('Minimum samples per category = ', min(df_app_train.label.value_counts()))
print('Maximum samples per category = ', max(df_app_train.label.value_counts()))


fname = base_path + '/audio_train/' + '88e792b8.wav'   # Hi-hat
wav = wave.open(fname)
print("Sampling (frame) rate = ", wav.getframerate())
print("Total samples (frames) = ", wav.getnframes())
print("Duration = ", wav.getnframes()/wav.getframerate())



rate, data = wavfile.read(fname)
print("Sampling (frame) rate = ", rate)
print("Total samples (frames) = ", data.shape)
print(data)

plt.plot(data, '-', )

plt.savefig('plot_wave.png')


plt.figure(figsize=(16, 4))
plt.plot(data[:1000], '.')
plt.plot(data[:1000], '-')

plt.savefig('plot_wave_zoom.png')


df_app_train['nframes'] = df_app_train['fname'].apply(lambda f: wave.open(base_path + '/audio_train/' + f).getnframes())
df_app_test['nframes'] = df_app_test['fname'].apply(lambda f: wave.open(base_path + '/audio_test/' + f).getnframes())

_, ax = plt.subplots(figsize=(16, 10))
sns.violinplot(ax=ax, x="label", y="nframes", data=df_app_train)
plt.xticks(rotation=90)
plt.title('Distribution of audio frames, per label', fontsize=16)
plt.savefig('plot_distribution_waves_train.png')

