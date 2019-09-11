# Import `pyplot` from `matplotlib`
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import torch as tc

#importando a tabela com o pandas
df = pd.read_csv('vazoes_C_60855000.csv', delimiter=',', names = ['Data', 'Maxima', 'Minima', 'Media'])
#print(df)

#Separando as colunas que serão utilizadas pela rede neural...
colunaMedia = df[['Media']]
colunaData = df[['Data']]
#print(colunaMedia)

#Separando os dados para treinamento e dados de testes...
training = colunaMedia.iloc[37:63]
test = colunaMedia.iloc[49:63]
#print(training)
#print(test)

#Separando as entradas e saidas do treinamento da rede...
inicio = 0
fim = 12
for i in range(5): #quantos meses vae ser pegos
    training_input = np.append(training.iloc[inicio:fim],[])
    #print("\n", training_input) #print to check
    training_output = np.append(training.iloc[fim],[])
    #print("\n", training_output) #print to check
    inicio = inicio+1
    fim = fim+1

#Separando as entradas e saidas do teste da rede...
inicio = 0
fim = 12
for i in range(2): #quantos meses vae ser pegos
    test_input = np.append(test.iloc[inicio:fim],[])
    #print("\n", test_input) #print to check
    test_output = np.append(test.iloc[fim],[])
    #print("\n", test_output) #print to check
    inicio = inicio+1
    fim = fim+1

