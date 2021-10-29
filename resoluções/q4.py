from numpy.core.fromnumeric import mean
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn import tree
from sklearn.metrics import mean_absolute_error


MAE_treino, MAE_valid = [], []

data = pd.read_csv('data/reg02.csv')

x = data.iloc[:, :-1].values
y = data.iloc[:, -1].values

# Modelo Árvore de Regressão

clf = tree.DecisionTreeRegressor()

# Cross-Validation k-fold k = 5

kf = KFold(n_splits=5)

for treino_index, valid_index in kf.split(x):

    # Dados de treino e validação

    x_treino, x_valid = x[treino_index], x[valid_index]
    y_treino, y_valid = y[treino_index], y[valid_index]

    # Treino do modelo e predição dos resultados

    clf.fit(x_treino, y_treino)

    y_treino_pred = clf.predict(x_treino)
    y_valid_pred = clf.predict(x_valid)

    # Armazenamento dos resultados para a iteração de k-fold

    MAE_treino.append(mean_absolute_error(y_treino, y_treino_pred))
    MAE_valid.append(mean_absolute_error(y_valid, y_valid_pred))

# Resultados MAE para base de treino e de validação

print("Mean Absolute Error (MAE) para a base de treino: " + str(round(mean(MAE_treino),2)))
print("Mean Absolute Error (MAE) para a base de validação: " + str(round(mean(MAE_valid),2)))