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
mat = [[0 for j in range(12)] for j in range(5)]
for i in range(5): #quantos meses vae ser pegos
    #training_input = np.append([[i,i+1]], [[i,+1]], axis=0)

    #training_input = np.matrix(training_input)
    training_input = np.append(training.iloc[inicio:fim], [])
    print("\n", training_input) #print to check
    mat[i]=training_input
    training_output = np.append(training.iloc[fim],[])
    print("\n", training_output) #print to check
    inicio = inicio+1
    fim = fim+1

print(mat)

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

'''class Net(tc.nn.Module):
    def __init__(self, input_size, hidden_size):
        super(Net, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.fc1 = tc.nn.Linear(self.input_size, self.hidden_size)
        self.relu = tc.nn.ReLU()
        self.fc2 = tc.nn.Linear(self.hidden_size, 1)
        self.sigmoid = tc.nn.Sigmoid()
    def forward(self, x):
        hidden = self.fc1(x)
        relu = self.relu(hidden)
        output = self.fc2(relu)
        output = self.sigmoid(output)
        return output

input_size = training_input.size(1) # number of features selected
hidden_size = 30 # number of nodes/neurons in the hidden layer
model = Net(input_size, hidden_size) # create the model
#criterion = tc.nn.BCELoss() # works for binary classification
criterion = tc.nn.ReLU
# without momentum parameter
optimizer = tc.optim.SGD(model.parameters(), lr = 0.9)
#with momentum parameter
optimizer = tc.optim.SGD(model.parameters(), lr = 0.9, momentum=0.2)

model.eval()
y_pred = model(test_input)
before_train = criterion(y_pred.squeeze(), test_output)
print('Test loss before training' , before_train.item())'''