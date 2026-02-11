import pandas as pd



base_plano_saude2= pd.read_csv('../plano_saude2.csv')


x_plano_saude2 = base_plano_saude2.iloc[:,0:1].values
y_plano_saude2 = base_plano_saude2.iloc[:,1].values