import os
import pandas as pd
import plotly.express as px
from pyod.models.knn import KNN
import numpy as np

base_dir = os.path.dirname(__file__)  # Serve para localizar o arquivo .py, para assim poder criar o caminho relativo até os dados em qualquer local
caminho = os.path.join(base_dir, "..", "dados", "cov_types.csv")
base = pd.read_csv(caminho)

import plotly.express as px

# grafico = px.box(base, y=['Elevation', 'Aspect', 'Slope'])
# grafico.show()


# grafico = px.scatter(x= base['Elevation'],y=base['Aspect'])
# grafico.show()

# grafico = px.scatter(x= base['Elevation'],y=base['Slope'])
# grafico.show()

# grafico = px.scatter(x= base['Aspect'],y=base['Slope'])
# grafico.show()

detector = KNN()
detector.fit(base.iloc[:,:-1])

previsoes = detector.labels_
print(previsoes)

print(np.unique(previsoes, return_counts=True))


confiaca_previsoes = detector.decision_scores_
print(confiaca_previsoes)

outliers = []
for i in range(len(previsoes)):
    if previsoes[i] ==1:
        outliers.append(i)

lista_outliers = base.iloc[outliers,:]

print(lista_outliers)