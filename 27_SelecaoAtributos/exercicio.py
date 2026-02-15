import numpy as np
import pandas as pd
import pickle
import os

from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.feature_selection import VarianceThreshold
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

base_dir = os.path.dirname(__file__)  # Serve para localizar o arquivo .py, para assim poder criar o caminho relativo até os dados em qualquer local
caminho = os.path.join(base_dir, "..", "dados", "cov_types.csv")
base = pd.read_csv(caminho)

columns = base.columns[:-3]



x=base.iloc[:,:-3]
y=base.iloc[:,-1]

scaler = MinMaxScaler()
x = scaler.fit_transform(x)

encoder= LabelEncoder()
y = encoder.fit_transform(y)

# print(x.var(axis=0))

print("Variancia\n")

selector = VarianceThreshold(threshold=0.02)
x_variance = selector.fit_transform(x)

# print(selector.variances_)
# print(x_variance.shape, x.shape)
indices = np.where(selector.variances_ > 0.02)

# print(columns[indices])

x_treinamento,x_teste,y_treinamento,y_teste= train_test_split(x_variance,y, test_size=0.25,random_state=0)
rf = RandomForestClassifier(random_state=0)
rf.fit(x_treinamento,y_treinamento)
previsoes = rf.predict(x_teste)
print(accuracy_score(y_teste,previsoes))
print(classification_report(y_teste,previsoes))

print("\nEXTRA TREE\n")

selecao = ExtraTreesClassifier()
selecao.fit(x,y)

importancias = selecao.feature_importances_ #importancias são colocadas em porcentagem
print(importancias)

indices = []
for i in range(len(importancias)):
    if importancias[i]> 0.07:
        indices.append(i)

x_tree=x[:,indices]
x_treinamento,x_teste,y_treinamento,y_teste= train_test_split(x_tree,y, test_size=0.25,random_state=0)

rf = RandomForestClassifier(criterion="entropy",min_samples_leaf=1, min_samples_split=5, n_estimators=100)
rf.fit(x_treinamento,y_treinamento)
previsoes = rf.predict(x_teste)
print(accuracy_score(y_teste,previsoes))
print(classification_report(y_teste,previsoes))