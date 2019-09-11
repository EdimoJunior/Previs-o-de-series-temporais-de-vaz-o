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
training = colunaMedia.iloc[37:63]
teste = colunaMedia.iloc[49:63]
#print(training)
#print(test)

#Separando as entradas e saidas do treianmento da rede e as entradas e saidas do teste
inicio = 0
fim = 12
training_input = np.array([])
for i in range(2): #quantos anos vae ser pegos
    training_input = np.append(training.iloc[inicio:fim],[])
    print("\n", training_input) #print to check
    training_output = np.append(training.iloc[fim],[])
    print("\n", training_output) #print to check
    inicio = fim
    fim = fim+12

test_input = np.append(training.iloc[-13:-1],[])
print("\n", test_input) #print to check

