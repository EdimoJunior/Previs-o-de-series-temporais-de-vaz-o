# Import `pyplot` from `matplotlib`
import matplotlib.pyplot as plt
import pandas as pd

#importando a tabela com o pandas
tabela = pd.read_csv('planinhaTestes.csv', delimiter=',', names = ['Data', 'Maxima', 'Minima', 'Media'])

#vetor p/ armazenar datas
vetDatas = ["" for x in range(13)]
#print(vetDatas)

#vetor p/ armazenar médias
vetMedias = [0] * 13
#print(vetMedias)

aux = 0

# preenchendo o vetor de datas
for i, j in tabela.iterrows():
    if i >= 23 and i <=34:
        aux = aux + 1
        #print("Linha = ",i, "\n",j['Data'])
        vetDatas[aux] = j['Data']
        #print()

aux = 0

# preenchendo o vetor de Medias
for i, j in tabela.iterrows():
    if i >= 23 and i <=34:
        aux = aux + 1
        #print("Linha = ",i, "\n",j['Media'])
        vetMedias[aux] = j['Media']
        #print()

# Initialize the plot
fig = plt.figure(figsize=(20,10))
ax1 = fig.add_subplot(111)

# or replace the three lines of code above by the following line:

# Plot the data
ax1.bar(vetDatas,vetMedias)

# Show the plot
plt.show()