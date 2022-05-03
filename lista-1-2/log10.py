import numpy as np
import math
import random
import matplotlib.pyplot as plt

def sigmoid(x):
	return 1.0/(1.0 + np.exp(-x))

def sigmoid_der(x):
	return x*(1.0 - x)

def relu(x):
    return x * (x > 0)

def relu_der(x):
    return 1. * (x > 0)

class NN:
    def __init__(self, inputs):
        self.inputs = inputs
        self.l=len(self.inputs)
        self.li=len(self.inputs[0])
        self.wi=np.random.random((1, 10))
        self.wh=np.random.random((10, 1))

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
dataSize = 2000
for i in range(dataSize):
        x = random.uniform(1, 10)
        inputs.append([x])
        outputs.append([math.log(x, 10)])
        
trainingSet = np.array(inputs[:math.floor(len(inputs)*0.8)])
trainingLabels = np.array(outputs[:math.floor(len(outputs)*0.8)])
testingSet = np.array(inputs[math.floor(len(inputs)*0.8):])
testingLabels = np.array(outputs[math.floor(len(outputs)*0.8):])
n=NN(trainingSet)
n.train(trainingSet, trainingLabels, 2000)
predictions = n.think(testingSet)
x = np.linspace(0,10,100)
ypredict = n.think(x)
y = np.log10(x)
plt.plot(x,ypredict, 'r', ls = '-')
plt.plot(x,y, 'b', ls = '-')
plt.show()
print(100-(corrects/len(predictions)))
        