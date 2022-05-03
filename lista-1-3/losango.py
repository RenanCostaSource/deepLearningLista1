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
        self.wi=np.zeros((2, 20))
        self.wh=np.zeros((20, 3))

    def think(self, inp):
        output =[]
        for i in range(len(inp)):
            s1=sigmoid(np.dot(np.array([inp[i]]), self.wi))
            s2=sigmoid(np.dot(s1, self.wh))
            output.append(s2)
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

dataSize =1000
prevx = np.linspace(-1,1,dataSize)
prevy = np.linspace(-1,1,dataSize)
x = []
y = []
for i in range(dataSize):
      for j in range(dataSize):
        x.append(prevx[i])  
        y.append(prevy[j])
np.random.shuffle(x)
np.random.shuffle(y)
groupsx = [[],[],[],[],[],[],[],[]]
groupsy = [[],[],[],[],[],[],[],[]]
data = []
dataLabels = []
for i in range(dataSize):
    inserted = True
    if(y[i]<=1 - x[i] and x[i] >= 0 and y[i]>= 0):
        groupsx[0].append(x[i])
        groupsy[0].append( y[i])
        dataLabels.append([0,0,0])
        data.append([x[i], y[i]])
    elif(y[i]<=1 + x[i] and x[i] <= 0 and y[i]>= 0):
        groupsx[1].append(x[i])
        groupsy[1].append( y[i])
        dataLabels.append([0,0,1])
        data.append([x[i], y[i]])
    elif(y[i]<= -x[i] -1 and x[i] <= 0 and y[i]<= 0):
        groupsx[2].append(x[i])
        groupsy[2].append( y[i])
        dataLabels.append([0,1,0])
        data.append([x[i], y[i]])
    elif(y[i]<= x[i] -1 and x[i] >= 0 and y[i]<= 0):
        groupsx[3].append(x[i])
        groupsy[3].append( y[i])
        dataLabels.append([0,1,1])
        data.append([x[i], y[i]])
    elif(y[i]>=1 - x[i] and x[i] >= 0 and y[i]>= 0 and x[i]**2 + y[i]**2 <=1):
        groupsx[4].append(x[i])
        groupsy[4].append( y[i])
        dataLabels.append([1,0,0])
        data.append([x[i], y[i]])
    elif(y[i]>=1 + x[i] and x[i] <= 0 and y[i]>= 0 and x[i]**2 + y[i]**2 <=1):
        groupsx[5].append(x[i])
        groupsy[5].append( y[i])
        dataLabels.append([1,0,1])
        data.append([x[i], y[i]])
    elif(y[i]<=-1 - x[i] and x[i] <= 0 and y[i]<= 0 and x[i]**2 + y[i]**2 <=1):
        groupsx[6].append(x[i])
        groupsy[6].append( y[i])
        dataLabels.append([1,1,0])
        data.append([x[i], y[i]])
    elif(y[i]>=-1 + x[i] and x[i] >= 0 and y[i]<= 0 and x[i]**2 + y[i]**2 <=1):
        groupsx[7].append(x[i])
        groupsy[7].append( y[i])  
        dataLabels.append([1,1,1])
        data.append([x[i], y[i]])
      
scaler = MinMaxScaler()
data = scaler.fit_transform(np.array(data))      
n=NN(data)
n.train(data, np.array(dataLabels), 500)
testx = np.linspace(-1,1,50)
testy = np.linspace(-1,1,50)
testData = []
for i in range(10):
  for j in range(10):
      testData.append([testx[i], testy[j]])  
testData = scaler.transform(np.array(testData)) 
classification = n.think(testData)
print(classification)
groupsTestx = [[],[],[],[],[],[],[],[]]
groupsTesty = [[],[],[],[],[],[],[],[]]
for i in range(len(testData)):
    if(classification[i][0][0]>=0.5):
        if classification[i][0][1] >= 0.5:
            if classification[i][0][2]>=0.5:
                groupsTestx[7].append(testData[i][0])
                groupsTesty[7].append(testData[i][1])
            else:
                groupsTestx[6].append(testData[i][0])
                groupsTesty[6].append(testData[i][1])
        else:
            if classification[i][0][2]>=0.5:
                groupsTestx[5].append(testData[i][0])
                groupsTesty[5].append(testData[i][1])
            else:
                groupsTestx[4].append(testData[i][0])
                groupsTesty[4].append(testData[i][1])
    else:
        if classification[i][0][1] >= 0.5:
            if classification[i][0][2]>=0.5:
                groupsTestx[3].append(testData[i][0])
                groupsTesty[3].append(testData[i][1])
            else:
                groupsTestx[2].append(testData[i][0])
                groupsTesty[2].append(testData[i][1])
        else:
            if classification[i][0][2]>=0.5:
                groupsTestx[1].append(testData[i][0])
                groupsTesty[1].append(testData[i][1])
            else:
                groupsTestx[0].append(testData[i][0])
                groupsTesty[0].append(testData[i][1])
                
        
plt.scatter(groupsTestx[0],groupsTesty[0], c='b')
plt.scatter(groupsTestx[1],groupsTesty[1], c='g')
plt.scatter(groupsTestx[2],groupsTesty[2], c='r')
plt.scatter(groupsTestx[3],groupsTesty[3], c='c')
plt.scatter(groupsTestx[4],groupsTesty[4], c='m')
plt.scatter(groupsTestx[5],groupsTesty[5], c='y')
plt.scatter(groupsTestx[6],groupsTesty[6], c='k')
plt.scatter(groupsTestx[7],groupsTesty[7], c='w')
plt.show()
        