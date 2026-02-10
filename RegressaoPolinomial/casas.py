
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error

base_casas = pd.read_csv('house_prices.csv')
x_casas = base_casas.iloc[:,3:19].values

y_casas = base_casas.iloc[:,2].values

x_casas_treinamento, x_casas_teste, y_casas_treinamento, y_casas_teste = train_test_split(x_casas, y_casas, test_size=0.3, random_state=0)

poly = PolynomialFeatures(degree = 2)
x_casas_treinamento_poly = poly.fit_transform(x_casas_treinamento) #fit_transform é apenas no primeiro
x_casas_teste_poly = poly.transform(x_casas_teste)

print(x_casas_treinamento_poly.shape, x_casas_teste_poly.shape)

regressor_casas_poly= LinearRegression()
regressor_casas_poly.fit(x_casas_treinamento_poly,y_casas_treinamento)

# print(regressor_casas_poly.score(x_casas_treinamento_poly,y_casas_treinamento))
# print(regressor_casas_poly.score(x_casas_teste_poly,y_casas_teste))

previsoes = regressor_casas_poly.predict(x_casas_teste_poly)
print("Previsoes\n",previsoes)

print(mean_absolute_error(y_casas_teste,previsoes))