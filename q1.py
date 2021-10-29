import pandas as pd
import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

data = pd.read_csv('class01.csv')

x = data.iloc[:, :-1].values
y = data.iloc[:, -1].values

x_treino, x_valid = x[:350], x[350:]
y_treino, y_valid = y[:350], y[350:]

# Modelo

modelo = GaussianNB().fit(x_treino, y_treino)

# Base de treino

y_treino_pred = modelo.predict(x_treino)

acuracia_treino = accuracy_score(y_treino,y_treino_pred)
print("Acurácia a base de treino: "+ str(round(100* acuracia_treino,1))+ "%")


# Base de validação

y_valid_pred = modelo.predict(x_valid)

acuracia_valid = accuracy_score(y_valid,y_valid_pred)
print("Acurácia a base de validação: "+ str(round(100* acuracia_valid,1))+ "%")