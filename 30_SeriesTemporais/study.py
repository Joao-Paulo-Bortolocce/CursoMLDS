import pandas as pd
import os
import numpy as np
import matplotlib.pylab as plt
from datetime import datetime
from statsmodels.tsa.seasonal import seasonal_decompose
from pmdarima.arima import auto_arima

base_dir = os.path.dirname(__file__)  # Serve para localizar o arquivo .py, para assim poder criar o caminho relativo até os dados em qualquer local
caminho = os.path.join(base_dir, "..", "dados", "AirPassengers.csv")
base = pd.read_csv(caminho)

dataset = pd.read_csv(
    caminho,
    parse_dates=['Month'],
    index_col='Month',
    date_format='%Y-%m'
)


time_series = dataset['#Passengers']

# print(time_series.iloc[1])
# print(time_series['1949-02'])
# print(time_series[datetime(1949,2,1)])
# print(time_series['1950-01-01':'1950-07-31'])
# print(time_series[:'1950-07-31'])
# print(time_series['1950'])
# print(time_series.index.max())
# print(time_series.index.min())
# plt.plot(time_series)
# plt.show()

time_series_ano = time_series.resample('YE').sum() #Soma todos os passageiros de cada ano, tbm é possivel fazer todos os meses
# plt.plot(time_series_ano)
# plt.show()

# decomposicao = seasonal_decompose(time_series)
# tendencia = decomposicao.trend
# sazonal = decomposicao.seasonal
# aleatorio = decomposicao.resid

# plt.plot(tendencia) #Qual a tendencia do número de passageiros de acordo com o passar dos anos
# plt.show()
# plt.plot(sazonal) # Qual o funcionamento do número de passageiros dentro de um periodo de tempo, nesse caso ele deixa claro que em um periodo do ano (Junho - Agosto) tem muito mais passageiros
# plt.show()
# plt.plot(aleatorio) #Indica fenomenos aleatórios, quebra da tendencia ou sazonalidade
# plt.show()

# Parâmetors P, Q e D
model = auto_arima(time_series)

# print(model.order)

predictions = model.predict(n_periods=24) #Preve o valor dos proximos 4 meses

# print(predictions)

train = time_series[:130]

test = time_series[130:]

print(train.shape, test.shape)

model2 = auto_arima(train, suppress_warnings=True)
prediction = pd.DataFrame(model2.predict(n_periods=14), index=test.index)
prediction.columns = ['passengers_predictions']


plt.figure(figsize=(8,5))
plt.plot(train, label = 'Training')
plt.plot(test, label = 'Test')
plt.plot(prediction, label = 'Predictions')
plt.legend();