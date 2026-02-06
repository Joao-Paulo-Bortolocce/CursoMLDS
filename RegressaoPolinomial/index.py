import pandas as pd
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
import plotly.express as px


base_plano_saude2= pd.read_csv('plano_saude2.csv')
print(base_plano_saude2)

x_plano_saude2 = base_plano_saude2.iloc[:,0:1].values
y_plano_saude2 = base_plano_saude2.iloc[:,1].values

# print(x_plano_saude2)

poly = PolynomialFeatures(degree= 4)
x_plano_saude2_poly = poly.fit_transform(x_plano_saude2)
# print(x_plano_saude2_poly)

regressor_saude_polinomial = LinearRegression()
regressor_saude_polinomial.fit(x_plano_saude2_poly,y_plano_saude2)

previsoes = regressor_saude_polinomial.predict(x_plano_saude2_poly)
print(previsoes)

grafico = px.scatter (x= x_plano_saude2[:,0], y=y_plano_saude2)
grafico.add_scatter (x= x_plano_saude2[:,0], y=previsoes, name= "Regressao")
grafico.show()