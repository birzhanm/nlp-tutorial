"""
Question 1: What is an underlying idea behind recurrent neural networks (RNN)?
RNN are meant to make use of past inputs to make sense of the current input.
"""

"""
Question 2: How to build an RNN in Numpy?
"""

import numpy as np

def softmax(x):
  return np.exp(x)/sum(np.exp(x))

# Numpy Implementation
class RNN_V1:
    def __init__(self, input_dim, hidden_dim, output_dim, bptt_truncate):
        # Assign instance variable
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        # Randomly initialize weights
        self.U = np.random.uniform(-np.sqrt(1./input_dim), np.sqrt(1./input_dim), (hidden_dim, input_dim))
        self.W = np.random.uniform(-np.sqrt(1./hidden_dim), np.sqrt(1./hidden_dim), (hidden_dim, hidden_dim))
        self.V = np.random.uniform(-np.sqrt(1./hidden_dim), np.sqrt(1./hidden_dim), (output_dim, hidden_dim))

    def forward(self, x):
        # The total number of time steps
        T = len(x)
        # During forward prop we need to store values of hidden states
        s = np.zeros((T+1, self.hidden_dim))
        # We set initial value of hidden state to be zero
        s[-1] = np.zeros(self.hidden_dim)
        o = np.zeros((T, self.output_dim))
        # Perform a forwars pass
        for t in np.arange(T):
          s[t] = np.tanh(self.U.dot(x[t]) + self.W.dot(s[t-1]))
          o[t] = softmax(self.V.dot(s[t]))
        return [o,s]

    def predict(self, x):
        # perform forward propagation and return the index of highest score
        o, s = self.forward(x)
        return np.argmax(o, axis=1)

    def compute_mean_loss(self, x, y):
        loss = 0
        for i in np.arange(len(y)):
            o, s = self.forward(x)
            prob_score = o[i,y[i]]
            loss -= np.log(prob_score)
        return loss/len(y)








# sanity check
"""
input_size = 5
input_dim = 3
output_dim = 3 #number of classes
hidden_dim = 2
bptt_truncate = 4
X = np.random.rand(input_size, input_dim)
y = np.random.randint(low=0, high=output_dim, size=input_size)
rnn = RNN_V1(input_dim, hidden_dim, output_dim, bptt_truncate)
prediction = rnn.predict(X)
loss = rnn.compute_mean_loss(X,y)
print(loss)
"""

"""
Question 3: How to build an RNN in PyTorch?
"""

# import torch
import torch
import torch.nn as nn

# create a synthetic dataset
input_dim = 5
output_dim = 4
input_size = 8
x = torch.randn(input_dim)
y = torch.randn(output_dim)
X = torch.randn(input_size, input_dim)
Y = torch.randn(input_size, output_dim)


# set hidden_dim and initial hidden state
hidden_dim = 3
hidden = torch.zeros(1,1,3)

# define rnn cell
cell = nn.RNN(input_dim, hidden_dim, batch_first=True)

# run the cell
for element in X:
    element = element.view(1,1,-1)
    out, hidden = cell(element, hidden)
    #print('output: {}'.format(out))

# it is possible to perform the above computation without writing for loop
X_new = X.view(1, 8, -1)
out, hidden = cell(X_new, hidden)
#print(out)

# it is possible to have batches. Suppose that we have 3 batches
X = torch.rand(3, 8, 5)
hidden = torch.zeros(1, 3, 3)
out, hidden = cell(X, hidden)
#print(out)

# let us design a simple RNN model
# for that we might need a new dataset. What we can do is to set x_{t+1} = (x_t + x_{t-1})/2
#X = torch.zeros(100,5)
#X[:2,:] = torch.rand(2,5)
#for i in range(2,100):
#    X[i,:] = (X[i-1]+X[i-2])/2 + 0.01*torch.randn(1,5)
#print(X)

def design_batch(batch_size, input_dim):
    X = torch.zeros(batch_size+1, input_dim)
    X[:2,:] = torch.randn(2, input_dim)
    for i in range(2, batch_size+1):
        X[i,:] = (X[i-1]+X[i-2])/2 + 0.1*torch.randn(1,5)
    input = X[:batch_size,:]
    output = X[1:,:]
    return input, output

# print(design_batch(10,5))

input_dim = 5
hidden_dim = 5
batch_size = 1
sequence_length = 10
num_layers = 1

class RNN(nn.Module):
    def __init__(self):
        super(RNN, self).__init__()
        self.rnn = nn.RNN(input_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, hidden, x):
        x = x.view(batch_size, sequence_length, input_dim)
        out, hidden = self.rnn(x, hidden)
        x = self.fc(out)
        return x

    def init_hidden(self):
        return torch.zeros(num_layers, batch_size, hidden_dim)

rnn = RNN()
#print(rnn)

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(rnn.parameters(), lr=0.01)


for epoch in range(1000):
    inputs, labels = design_batch(10, 5)
    optimizer.zero_grad()
    loss=0
    hidden = rnn.init_hidden()

    prediction = rnn(hidden, inputs)
    loss = criterion(prediction, labels)
    if epoch%10 == 0:
        print('Epoch: {} - Loss: {}'.format(epoch, loss))
    loss.backward()
    optimizer.step()
print("Finished Learning")

print(rnn.fc.weight)
