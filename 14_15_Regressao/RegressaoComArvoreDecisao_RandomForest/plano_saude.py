import pandas as pd
import plotly.express as px
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
import numpy as np


base_plano_saude2= pd.read_csv('../plano_saude2.csv')
# print(base_plano_saude2)

x_plano_saude2 = base_plano_saude2.iloc[:,0:1].values
y_plano_saude2 = base_plano_saude2.iloc[:,1].values

# regressor_arvore_saude = DecisionTreeRegressor()
# regressor_arvore_saude.fit(x_plano_saude2,y_plano_saude2)

# previsoes = regressor_arvore_saude.predict(x_plano_saude2)
# print(previsoes)

# grafico = px.scatter (x= x_plano_saude2.ravel(), y=y_plano_saude2) #Cria um grafico com os pontos desejados, geralmente os valores reais
# grafico.add_scatter (x= x_plano_saude2.ravel(), y=previsoes, name= "Regressao") #adiciona uma linha mostrando as previsões geradas
# grafico.show()

x_teste_arvore = np.arange(min(x_plano_saude2.ravel()),max(x_plano_saude2.ravel()),0.1) #Cria registros fake para testar e ter mais informações para analisar o grafico
# print(x_teste_arvore)

x_teste_arvore = x_teste_arvore.reshape(-1,1)


# previsoes = regressor_arvore_saude.predict(x_teste_arvore)
# grafico = px.scatter (x= x_plano_saude2.ravel(), y=y_plano_saude2)
# grafico.add_scatter (x= x_teste_arvore.ravel(), y=previsoes, name= "Regressao")
# grafico.show()

#AQUI COMEÇA O ALGORITMO DO RANDOM FOREST

regressor_random_forest_saude = RandomForestRegressor(n_estimators=10) #Cria 10 arvores de decisao
regressor_random_forest_saude.fit(x_plano_saude2,y_plano_saude2)

print("Score da random Forest", regressor_random_forest_saude.score(x_plano_saude2,y_plano_saude2))

grafico = px.scatter (x= x_plano_saude2.ravel(), y=y_plano_saude2)
grafico.add_scatter (x= x_teste_arvore.ravel(), y=regressor_random_forest_saude.predict(x_teste_arvore), name= "Regressao")
grafico.show()
