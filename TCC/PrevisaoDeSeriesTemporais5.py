# Import `pyplot` from `matplotlib`
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import torch

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
training_input1 = [[0 for j in range(12)] for j in range(14)]
training_output1 = []

for i in range(14): #quantos meses vae ser pegos

    training_input = np.append(training.iloc[inicio:fim], [])
    print("\n", training_input) #print to check
    training_input1[i]=training_input

    training_output = np.append(training.iloc[fim],[])
    print("\n", training_output) #print to check
    training_output1.append(training.iloc[fim])

    inicio = inicio+1
    fim = fim+1

print("\n", training_input1)
print("\n", training_output1)

print("\nEntrada --> ", training_input1[0])
print("Saida --> ", training_output1[0].Media)

'''class Net(torch.nn.Module):
    def __init__(self, input_size, hidden_size):
        super(Net, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.fc1 = torch.nn.Linear(self.input_size, self.hidden_size)
        self.relu = torch.nn.ReLU()
        self.fc2 = torch.nn.Linear(self.hidden_size, 1)
        self.sigmoid = torch.nn.Sigmoid()
    def forward(self, x):
        hidden = self.fc1(x)
        relu = self.relu(hidden)
        output = self.fc2(relu)
        output = self.sigmoid(output)
        return output

#convert to tensors
training_input1 = torch.FloatTensor(training_input1)
training_output1 = torch.FloatTensor(training_output1)

input_size = training_input1.__sizeof__()# number of features selected
hidden_size = 30 # number of nodes/neurons in the hidden layer
model = Net(input_size, hidden_size) # create the model
#criterion = tc.nn.BCELoss() # works for binary classification
criterion = torch.nn.ReLU
# without momentum parameter
optimizer = torch.optim.SGD(model.parameters(), lr = 0.9)
#with momentum parameter
optimizer = torch.optim.SGD(model.parameters(), lr = 0.9, momentum=0.2)

model.eval()
y_pred = model(training_input1)
before_train = criterion(y_pred.squeeze(), training_output1)
print('Test loss before training' , before_train.item())

model.train()
epochs = 5000
errors = []
for epoch in range(epochs):
    optimizer.zero_grad()
    # Forward pass
    y_pred = model(training_input1)
    # Compute Loss
    loss = criterion(y_pred.squeeze(), training_output1)
    errors.append(loss.item())
    print('Epoch {}: train loss: {}'.format(epoch, loss.item()))
    # Backward pass
    loss.backward()
    optimizer.step()'''