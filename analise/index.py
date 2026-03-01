import pandas as pd
import os
import numpy as np
import matplotlib.pylab as plt
from sklearn.preprocessing import LabelEncoder
import seaborn as sns


base_dir = os.path.dirname(__file__)  # Serve para localizar o arquivo .py, para assim poder criar o caminho relativo até os dados em qualquer local
caminho = os.path.join(base_dir, "..", "dados", "baseCompletaCorridasSemRedundancia.csv")
base = pd.read_csv(
    caminho,
    sep=',',  # Confirma separador
    quotechar='"',  # Aspas protegem vírgulas internas
    escapechar='\\',  # Escapa aspas internas
    on_bad_lines='skip',
    engine='python',  # Mais tolerante que 'c'
    encoding='utf-8-sig'
)
print(f"Linhas carregadas: {len(base)}")
print(base.head())


print(base)

nome_label= LabelEncoder()
cidade_label=LabelEncoder()
estado_label=LabelEncoder()


base['nome'] = nome_label.fit_transform(base['nome'].astype(str).fillna('MISSING'))
base['cidade'] = cidade_label.fit_transform(base['cidade'].astype(str).fillna('MISSING'))
base['estado'] = estado_label.fit_transform(base['estado'].astype(str).fillna('MISSING'))




figura = plt.figure(figsize=(10,10))
sns.heatmap(base.corr(numeric_only=True),cmap='coolwarm',annot=True)
plt.show()