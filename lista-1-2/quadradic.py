import numpy as np
import math
import random
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
def sigmoid(x):
	return 1.0/(1.0 + np.exp(-x))

def sigmoid_der(x):
	return x*(1.0 - x)

def relu(x):
    return x * (x > 0)

def relu_der(x):
    return 1. * (x > 0)

def sourceFunction(x):
    return (10*math.pow(x,5))+(5*math.pow(x,4))+(2*math.pow(x,3))-(0.5*math.pow(x,2))+(3*x)+2

class NN:
    def __init__(self, inputs):
        self.inputs = inputs
        self.l=len(self.inputs)
        self.li=len(self.inputs[0])
        self.wi=np.random.random((1, 25))
        self.wh=np.random.random((25, 1))

    def think(self, inp):
        output =[]
        for i in range(len(inp)):
            s1=sigmoid(np.dot(np.array([inp[i]]), self.wi))
            s2=sigmoid(np.dot(s1, self.wh))
            output.append(s2[0])
        return np.array(output)

    def train(self, inputs,outputs, it):
        for k in range(it):
            for i in range(len(inputs)):
                l0=np.array([inputs[i]])
                l1=sigmoid(np.dot(l0, self.wi))
                l2=sigmoid(np.dot(l1, self.wh))
                l2_err=np.array(outputs[i]) - l2
                l2_delta=np.multiply(l2_err, sigmoid_der(l2))
                l1_err=np.dot(l2_delta, self.wh.T)
                l1_delta=np.multiply(l1_err, sigmoid_der(l1))
                self.wh+=np.dot(l1.T, l2_delta)
                self.wi+=np.dot(l0.T, l1_delta)

inputs=[]
outputs=[]
inputs = np.linspace(0,20,15000).reshape(-1, 1)
inputsTmp = inputs

j = len(inputsTmp)-1
k=0
h=0
while k<j:
    inputs[h] = inputsTmp[k]
    h+=1
    k+=1
    inputs[h] = inputsTmp[j]
    h+=1
    j-=1
for i in range(len(inputs)):
        outputs.append(sourceFunction(inputs[i]))
scaler = MinMaxScaler()
outputs = np.array(outputs)

scaler.fit(np.array(outputs))
outputs = scaler.transform(outputs)
n=NN(inputs)
n.train(inputs, outputs, 20000)
x = np.linspace(0,20,100)
ypredict = scaler.inverse_transform(n.think(x).reshape(-1, 1))
y = []
for i in range(len(x)):
    y.append(sourceFunction(x[i]))
plt.plot(x,ypredict, 'r', ls = '-')
plt.plot(x,y, 'b', ls = '-')
plt.show()
        