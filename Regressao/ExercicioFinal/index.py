import pickle

import matplotlib.pyplot as plt
# import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
# from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split

base = pd.read_csv('parkinson.csv')
base["sex"] = base["sex"].map({"male": 0, "female": 1})

x_base = base.iloc[:,0:19].values
y_base = base.iloc[:,19].values



x_treinamento,x_teste,y_treinamento,y_teste=train_test_split(x_base,y_base,test_size=0.25, random_state=0)

# REGRESSAO POLINOMIAL
# poly= PolynomialFeatures(degree=4)
# x_treinamento_poly= poly.fit_transform(x_treinamento)
# x_teste_poly= poly.transform(x_teste)
# print(x_treinamento_poly.shape, x_teste_poly.shape)

# lrp= LinearRegression()
# lrp.fit(x_treinamento_poly,y_treinamento)

# print("Score com dados de treino", lrp.score(x_treinamento_poly,y_treinamento))
# print("Score com dados de teste", lrp.score(x_teste_poly,y_teste)) #Se der negativo quer dizer que fica pior que realizar a regressao por calculo de média

# predicoes = lrp.predict(x_treinamento_poly)
# print("Erro absoluto medio treinamento:", mean_absolute_error(predicoes,y_treinamento))

# plt.figure(figsize=(20, 10))
# plt.plot(y_treinamento[::10], label="Valores reais")
# plt.plot(predicoes[::10], label="Valores Preditos")
# plt.legend()


# predicoes = lrp.predict(x_teste_poly)
# print("Erro absoluto medio teste:", mean_absolute_error(predicoes,y_teste))

# plt.figure(figsize=(20, 10))
# plt.plot(y_teste[::10], label="Valores reais")
# plt.plot(predicoes[::10], label="Valores Preditos")
# plt.legend()

# REGRESSAO POR ARVORE DE DECISAO
# tree = DecisionTreeRegressor()
# tree.fit(x_treinamento,y_treinamento)
# print("Score com dados de treino:", tree.score(x_treinamento,y_treinamento))
# print("Score com dados de teste:", tree.score(x_teste,y_teste)) 

# predicoes = tree.predict(x_teste)
# print("Erro absoluto medio teste:", mean_absolute_error(predicoes,y_teste))
# plt.figure(figsize=(20, 10))
# plt.plot(y_teste[::10], label="Valores reais")
# plt.plot(predicoes[::10], label="Valores Preditos")
# plt.legend()
# plt.show()

# REGRESSAO POR RANDOM FOREST
# rf = RandomForestRegressor(n_estimators=10)
# rf.fit(x_treinamento,y_treinamento)
# print("Score com dados de treino:", rf.score(x_treinamento,y_treinamento))
# print("Score com dados de teste:", rf.score(x_teste,y_teste)) 

# predicoes = rf.predict(x_teste)
# print("Erro absoluto medio teste:", mean_absolute_error(predicoes,y_teste))
# plt.figure(figsize=(20, 10))
# plt.plot(y_teste[::10], label="Valores reais")
# plt.plot(predicoes[::10], label="Valores Preditos")
# plt.legend()
# plt.show()


#REGRESSAO POR REDES NEURAIS ARTIFICIAIS
mlp = MLPRegressor(max_iter=1000)
scale_x=StandardScaler() #Dados precisam ficar normalizados
scale_y=StandardScaler()
x_treinamento_scaled= scale_x.fit_transform(x_treinamento)
y_treinamento_scaled=scale_y.fit_transform(y_treinamento.reshape(-1,1))
x_teste_scaled= scale_x.transform(x_teste)
y_teste_scaled=scale_y.transform(y_teste.reshape(-1,1))

mlp.fit(x_treinamento_scaled,y_treinamento_scaled.ravel())
print("Score com dados de treino:", mlp.score(x_treinamento_scaled,y_treinamento_scaled.ravel()))
print("Score com dados de teste:", mlp.score(x_teste_scaled,y_teste_scaled.ravel())) 

predicoes_scaled = mlp.predict(x_teste_scaled)
predicoes = scale_y.inverse_transform(
    predicoes_scaled.reshape(-1, 1)
).ravel()

y_real = scale_y.inverse_transform(
    y_teste_scaled
).ravel()

print("Erro absoluto médio (escala real):", mean_absolute_error(y_real, predicoes))
plt.figure(figsize=(20, 10))
plt.plot(y_real[::10],label="Valores reais")
plt.plot(predicoes[::10],label="Valores preditos")
plt.xlabel("Amostras")
plt.ylabel("Valor")
plt.title("MLP - Valores Reais vs Preditos")
plt.legend()
plt.grid(True)
plt.show()