# Import `pyplot` from `matplotlib`
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

#importando a tabela com o pandas
df = pd.read_csv('vazoes_C_60855000.csv', delimiter=',', names = ['Data', 'Maxima', 'Minima', 'Media'])
#print(df)

#Separando as colunas que serão utilizadas pela rede neural...
colunaMedia = df[['Media']]
colunaData = df[['Data']]
#print(colunaMedia)

#Separando os dados para treinamento e dados de testes...
training = colunaMedia.iloc[28:62]
test = colunaMedia.iloc[62:64]
#print(training)
#print(test)

#Embaralhando os dados
#training = training.sample(frac=1)
#test = test.sample(frac=1)
#print(training)
#print(test)

#all rows, all columns except for the last 3 columns
inicio = 0
fim = 12
training_input = training.iloc[0:12]
training_input = np.append(training.iloc[0:12],0)
print(training_input) #print to check
