from numpy.core.fromnumeric import mean
import pandas as pd
import numpy as np
from math import dist
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score


data = pd.read_csv('data/class02.csv')

x = data.iloc[:, :-1].values
y = data.iloc[:, -1].values

# Modelo

knn = KNeighborsClassifier(n_neighbors=10)

# Cross-Validation 5 folds

acc = cross_val_score(knn, x, y, cv=5)

# Acurácia de cada fold

for i in range(0,len(acc)):
    print("Acurácia da pasta " + str(i+1) + ": "+ str(round(100*acc[i],1))+ "%")

# Acurácia média

print("Acurácia média para a base de validação: " + str(round(mean(acc*100),1)) + "%")
