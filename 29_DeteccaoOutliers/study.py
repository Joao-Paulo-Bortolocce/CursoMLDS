#DETECÇÃO POR BOXPLOT
import os
import pandas as pd
import plotly.express as px
from pyod.models.knn import KNN
import numpy as np

base_dir = os.path.dirname(__file__)  # Serve para localizar o arquivo .py, para assim poder criar o caminho relativo até os dados em qualquer local
caminho = os.path.join(base_dir, "..", "dados", "credit_data.csv")
base_credit = pd.read_csv(caminho)


# print(base_credit.isnull().sum())
base_credit.dropna(inplace=True)
# print(base_credit.isnull().sum())

grafico = px.box(base_credit, y= 'age')
# grafico.show()

outliers_age = base_credit[base_credit['age']<0]

# print(outliers_age)

grafico = px.box(base_credit, y= 'loan')
# grafico.show()

outliers_loan = base_credit[base_credit['loan']> 13300]
# print(outliers_loan)

#DETECÇÃO DE OUTLIERS COM GRAFICO DE DISPERSAO
# grafico = px.scatter(x= base_credit['income'],y=base_credit['age'])
# grafico.show()

# grafico = px.scatter(x= base_credit['income'],y=base_credit['loan'])
# grafico.show()

# grafico = px.scatter(x= base_credit['age'],y=base_credit['loan'])
# grafico.show()

# caminho = os.path.join(base_dir, "..", "dados", "census.csv")
# print(caminho)
# base_census = pd.read_csv(caminho)
# print(base_census)
# grafico = px.scatter(x= base_census['age'],y=base_census['final-weight'])
# grafico.show()

#DETECÇÃO DE OUTLIERS COM BIBLIOTECA PyOD

detector = KNN()
detector.fit(base_credit.iloc[:,1:4])
previsoes = detector.labels_

print(np.unique(previsoes, return_counts=True))

confiaca_previsoes = detector.decision_scores_

outliers = []
for i in range(len(previsoes)):
    if previsoes[i] ==1:
        outliers.append(i)

lista_outliers = base_credit.iloc[outliers,:]