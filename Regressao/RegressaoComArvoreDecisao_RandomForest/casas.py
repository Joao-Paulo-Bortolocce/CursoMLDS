import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

base_casas = pd.read_csv('../house_prices.csv')
x_casas = base_casas.iloc[:,3:19].values

y_casas = base_casas.iloc[:,2].values

x_casas_treinamento, x_casas_teste, y_casas_treinamento, y_casas_teste = train_test_split(x_casas, y_casas, test_size=0.3, random_state=0)

# print(x_casas_treinamento.shape)


# regressor_arvore_casas = DecisionTreeRegressor()
# regressor_arvore_casas.fit(x_casas_treinamento,y_casas_treinamento)

# print(regressor_arvore_casas.score(x_casas_teste,y_casas_teste))

# previsoes = regressor_arvore_casas.predict(x_casas_teste)

# print(mean_absolute_error(y_casas_teste,previsoes))

#AQUI COMEÇA O RANDOM FOREST

regressor_random_forest_casas = RandomForestRegressor(n_estimators=100)
regressor_random_forest_casas.fit(x_casas_treinamento,y_casas_treinamento)

# print("Analisando o score na base de dados de treinamento: ", regressor_random_forest_casas.score(x_casas_treinamento,y_casas_treinamento))
# print("Analisando o score na base de dados de teste: ", regressor_random_forest_casas.score(x_casas_teste,y_casas_teste))

previsoes = regressor_random_forest_casas.predict(x_casas_teste)

print(mean_absolute_error(y_casas_teste,previsoes))