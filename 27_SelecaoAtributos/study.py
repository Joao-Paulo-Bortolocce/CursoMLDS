import os
import pandas as pd
import numpy as np

from sklearn.preprocessing import LabelEncoder, MinMaxScaler, OneHotEncoder
from sklearn.feature_selection import VarianceThreshold
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.metrics import accuracy_score, classification_report

base_dir = os.path.dirname(__file__)  # Serve para localizar o arquivo .py, para assim poder criar o caminho relativo até os dados em qualquer local
caminho = os.path.join(base_dir, "..", "dados", "census.csv")
base_census = pd.read_csv(caminho)

colunas = base_census.columns[:-1]
# print(colunas)

x_census = base_census.iloc[:, :-1].values
y_census = base_census.iloc[:, 14].values

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

scaler = MinMaxScaler()
x_census_scaler = scaler.fit_transform(x_census)

#SELEÇÃO COM LOW VARIANCE

# for i in range(x_census.shape[1]):
#     print(x_census_scaler[:,i].var())

# valThreshold = 0.05

# selecao= VarianceThreshold(threshold=valThreshold)
# x_census_varancia = selecao.fit_transform(x_census_scaler)

# # print(selecao.variances_) #Faz a mesma coisa que o for acima para exibir as varianças

# indices = np.where(selecao.variances_ > valThreshold) # seleciona os indices dos atributos selecionados no threshold

# indices = [int(i) for i in np.ravel(indices)]


# print(colunas[indices])


# colunasDropar = base_census.columns.difference(
#     base_census.columns[indices]
# )


# # remove a última posição
# colunasDropar=colunasDropar.drop('income')

# base_census_variacia = base_census.drop(columns=colunasDropar)

# x_census_varancia = base_census_variacia.iloc[:,0:5].values
# y_census_varancia = base_census_variacia.iloc[:,5].values

# x_census_varancia[:,0]= label_encoder_workclass.fit_transform(x_census_varancia[:,0])
# x_census_varancia[:,1]= label_encoder_education.fit_transform(x_census_varancia[:,1])
# x_census_varancia[:,2]= label_encoder_maritial.fit_transform(x_census_varancia[:,2])
# x_census_varancia[:,3]= label_encoder_occupation.fit_transform(x_census_varancia[:,3])
# x_census_varancia[:,4]= label_encoder_relationship.fit_transform(x_census_varancia[:,4])

# onehotencorder= ColumnTransformer(transformers=[("OneHot",OneHotEncoder(sparse_output=False),[0,1,2,3,4])],remainder="passthrough")
# x_census_varancia=onehotencorder.fit_transform(x_census_varancia)
# # print(x_census.shape)

# scaler = MinMaxScaler()
# x_census_varancia = scaler.fit_transform(x_census_varancia)


# x_treinamento,x_teste,y_treinamento,y_teste= train_test_split(x_census,y_census_varancia, test_size=0.15,random_state=0)

# rf = RandomForestClassifier(criterion="entropy",min_samples_leaf=1, min_samples_split=5, n_estimators=100)
# rf.fit(x_treinamento,y_treinamento)
# previsoes = rf.predict(x_teste)
# print(accuracy_score(y_teste,previsoes))
# print(classification_report(y_teste,previsoes))


#EXTRA TREE CLASSIFIER

print(x_census_scaler.shape)
selecao = ExtraTreesClassifier()
selecao.fit(x_census_scaler,y_census)

importancias = selecao.feature_importances_ #importancias são colocadas em porcentagem

indices = []
for i in range(len(importancias)):
    if importancias[i]>= 0.029:
        indices.append(i)

x_census_extra = x_census[:, indices]

onehotencorder= ColumnTransformer(transformers=[("OneHot",OneHotEncoder(sparse_output=False),[1,3,5,6,7])],remainder="passthrough")
x_census_extra=onehotencorder.fit_transform(x_census_extra)


x_treinamento,x_teste,y_treinamento,y_teste= train_test_split(x_census_extra,y_census, test_size=0.15,random_state=0)

rf = RandomForestClassifier(criterion="entropy",min_samples_leaf=1, min_samples_split=5, n_estimators=100)
rf.fit(x_treinamento,y_treinamento)
previsoes = rf.predict(x_teste)
print(accuracy_score(y_teste,previsoes))
print(classification_report(y_teste,previsoes))