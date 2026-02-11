import pandas as pd
import os
import numpy as np
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from imblearn.under_sampling import TomekLinks

base_dir = os.path.dirname(__file__)  # Serve para localizar o arquivo .py, para assim poder criar o caminho relativo até os dados em qualquer local
caminho = os.path.join(base_dir, "..", "dados", "census.csv")
base_census = pd.read_csv(caminho)

print(np.unique(base_census['income'], return_counts=True))

print(sns.countplot(x=base_census["income"]))

x_census = base_census.iloc[:,0:14].values
y_census= base_census.iloc[:,14].values

#TRATAMENTO DOS DADOS NÃO FUNCIONAM EM VARIAVEIS CATEGORICAS, processo abaixo transforma em dados numericos

label_encoder_workclass=LabelEncoder()
label_encoder_education=LabelEncoder()
label_encoder_maritial=LabelEncoder()
label_encoder_occupation=LabelEncoder()
label_encoder_relationship=LabelEncoder()
label_encoder_race=LabelEncoder()
label_encoder_sex=LabelEncoder()
label_encoder_country=LabelEncoder()

x_census[:,1]= label_encoder_workclass.fit_transform(x_census[:,1])
x_census[:,3]= label_encoder_education.fit_transform(x_census[:,3])
x_census[:,5]= label_encoder_maritial.fit_transform(x_census[:,5])
x_census[:,6]= label_encoder_occupation.fit_transform(x_census[:,6])
x_census[:,7]= label_encoder_relationship.fit_transform(x_census[:,7])
x_census[:,8]= label_encoder_race.fit_transform(x_census[:,8])
x_census[:,9]= label_encoder_sex.fit_transform(x_census[:,9])
x_census[:,13]= label_encoder_country.fit_transform(x_census[:,13])

#SUBAMOSTRAGEM COM TOMEK LINKS
tl= TomekLinks(sampling_strategy='majority')
x_under,y_under = tl.fit_resample(x_census,y_census)

print(x_under.shape, y_under.shape)

onehotencorder= ColumnTransformer(transformers=[("OneHot",OneHotEncoder(),[1,3,5,6,7,8,9,13])],remainder="passthrough")
x_census=onehotencorder.fit_transform(x_census)
# print(x_census.shape)

x_treinamento,x_teste,y_treinamento,y_teste= train_test_split(x_census,y_census, test_size=0.15,random_state=0)
print(x_treinamento.shape, x_teste.shape)

rf = RandomForestClassifier(criterion="entropy",min_samples_leaf=1, min_samples_split=5, n_estimators=100)
rf.fit(x_treinamento,y_treinamento)
previsoes = rf.predict(x_teste)
print(accuracy_score(y_teste,previsoes))