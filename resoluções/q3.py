from numpy.core.fromnumeric import mean
import pandas as pd
import numpy as np
from sklearn import linear_model
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import mean_squared_error


RMSE_treino, RMSE_valid= [],[]

data = pd.read_csv('data/reg01.csv')

x = data.iloc[:, :-1].values
y = data.iloc[:, -1].values

# Modelo Lasso alpha = 1

lasso = linear_model.Lasso(alpha=1)

# Cross-Validation Leave One Out (LOO)

loo = LeaveOneOut()

for treino_index, valid_index in loo.split(x):

    # Dados de treino e validação

    x_treino, x_valid = x[treino_index], x[valid_index]
    y_treino, y_valid = y[treino_index], y[valid_index]

    # Treino do modelo e predição dos resultados

    lasso.fit(x_treino, y_treino)

    y_treino_pred = lasso.predict(x_treino)
    y_valid_pred = lasso.predict(x_valid)

    # Armazenamento dos resultados para a iteração de LOO

    RMSE_treino.append(np.sqrt(mean_squared_error(y_treino, y_treino_pred)))
    RMSE_valid.append(np.sqrt(mean_squared_error(y_valid, y_valid_pred)))

# Resultados RMSE para base de treino e de validação

print("Root Mean Squared Error (RMSE) para a base de treino: " + str(round(mean(RMSE_treino),2)))
print("Root Mean Squared Error (RMSE) para a base de validação: " + str(round(mean(RMSE_valid),2)))