```python
import numpy as np
import random
from sklearn.preprocessing import normalize
from sklearn.preprocessing import MinMaxScaler
import joblib
from joblib import Parallel, delayed  
import multiprocessing
import time
import matplotlib.pyplot as plt
from mlxtend.plotting import plot_decision_regions
from mpl_toolkits.mplot3d import Axes3D
scaler_filename = "scaler.save"
```


```python
dataSize = 10000
correctData = []
wrongData = []

def wrongNoise():
    return random.choice([-1,1])*(random.uniform(0.11, 1))
def acceptableNoise():
    return random.uniform(-0.1, 0.1)
for i in range(dataSize):
    if i<(dataSize/8):
        correctData.append([(0+acceptableNoise()),(0+acceptableNoise()),(0+acceptableNoise())])
        wrongData.append([(0+wrongNoise()),(0+wrongNoise()),(0+wrongNoise())])
    elif i<(dataSize/8*2):
        correctData.append([(0+acceptableNoise()),(0+acceptableNoise()),(1+acceptableNoise())])
        wrongData.append([(0+wrongNoise()),(0+wrongNoise()),(1+wrongNoise())])
    elif i<(dataSize/8*3):
        correctData.append([(0+acceptableNoise()),(1+acceptableNoise()),(0+acceptableNoise())])
        wrongData.append([(0+wrongNoise()),(1+wrongNoise()),(0+wrongNoise())])
    elif i<(dataSize/8*4):
        correctData.append([(0+acceptableNoise()),(1+acceptableNoise()),(1+acceptableNoise())])
        wrongData.append([(0+wrongNoise()),(1+wrongNoise()),(1+wrongNoise())])
    elif i<((dataSize/8)*5):
        correctData.append([(1+acceptableNoise()),(0+acceptableNoise()),(0+acceptableNoise())])
        wrongData.append([(1+wrongNoise()),(0+wrongNoise()),(0+wrongNoise())])
    elif i<((dataSize/8)*6):
        correctData.append([(1+acceptableNoise()),(0+acceptableNoise()),(1+acceptableNoise())])
        wrongData.append([(1+wrongNoise()),(0+wrongNoise()),(1+wrongNoise())])
    elif i<((dataSize/8)*7):
        correctData.append([(1+acceptableNoise()),(1+acceptableNoise()),(0+acceptableNoise())])
        wrongData.append([(1+wrongNoise()),(1+wrongNoise()),(0+wrongNoise())])
    else:
        correctData.append([(1+acceptableNoise()),(1+acceptableNoise()),(1+acceptableNoise())])
        wrongData.append([(1+wrongNoise()),(1+wrongNoise()),1+wrongNoise()])
random.shuffle(correctData)
random.shuffle(wrongData)
with open('correct.npy', 'wb') as f:
    np.save(f, np.array(correctData))
with open('wrong.npy', 'wb') as f:
    np.save(f, np.array(wrongData))
```


```python
class RBPerceptron:

  def __init__(self, number_of_epochs = 100, learning_rate = 0.1, thread =0):
    self.number_of_epochs = number_of_epochs
    self.learning_rate = learning_rate
    self.thread = thread
  def binaryCrossEntropy(y_true, y_pred):
    m = y_true.shape[1]
    y_pred = np.clip(y_pred, 1e-7, 1 - 1e-7)
    # Calculating loss
    loss = -1/m * (np.dot(y_true.T, np.log(y_pred)) + np.dot((1 - y_true).T, np.log(1 - y_pred)))

    return loss
  def train(self, X, D):
    num_features = X.shape[1]
    self.w = np.zeros(num_features + 1)
    for i in range(self.number_of_epochs):
      if(i%100 ==0):
        print(self.thread,'-',i)
      for sample, desired_outcome in zip(X, D):
        prediction    = self.predict(sample)
        difference    = (desired_outcome - prediction)
        weight_update = self.learning_rate * difference
        self.w[1:]    += weight_update * sample
        self.w[0]     += weight_update
    return self

  def predict(self, sample):
    outcome = np.dot(sample, self.w[1:]) + self.w[0]
    return np.where(outcome > 0, 1, 0)
```


```python
with open('correct.npy', 'rb') as f:
    correctData = np.load(f)
with open('wrong.npy', 'rb') as f:
    wrongData = np.load(f)
```


```python
# data pre processing
trainingData1 = []
trainingLabels1 = []
trainingData2 = []
trainingLabels2 = []
trainingData3 = []
trainingLabels3 = []
trainingData4 = []
trainingLabels4 = []
trainingData5 = []
trainingLabels5 = []
trainingData6 = []
trainingLabels6 = []
trainingData7 = []
trainingLabels7 = []
trainingData8 = []
trainingLabels8 = []
trainingData9 = []
trainingLabels9 = []
trainingData10 = []
trainingLabels10 = []
trainingData11 = []
trainingLabels11 = []
trainingData12 = []
trainingLabels12 = [] 
npArrayOfData = np.array(correctData+wrongData)
scaler = MinMaxScaler()
scaler.fit(npArrayOfData)
joblib.dump(scaler, scaler_filename) 
correctData = scaler.transform(np.array(correctData))
correctTestingData = correctData[round(len(correctData)*0.8):]
correctData = correctData[:round(len(correctData)*0.8)]
correctDataIndex = len(correctData)-1
wrongData = scaler.transform(np.array(wrongData))
wrongTestingData = wrongData[round(len(wrongData)*0.8):]
wrongData = wrongData[:round(len(wrongData)*0.8)]
wrongDataIndex = len(wrongData)-1
valuesScaled = scaler.transform([[0,0.5,1]])
def decideWichDatasetToPut(data, value):
    if data[0]>valuesScaled[0][2]:
        trainingData1.append(data)
        trainingLabels1.append(value)
    elif data[0]>valuesScaled[0][1]:
        trainingData2.append(data)
        trainingLabels2.append(value)
    elif data[0]>valuesScaled[0][0]:
        trainingData3.append(data)
        trainingLabels3.append(value)
    else:
        trainingData4.append(data)
        trainingLabels4.append(value) 
    if data[1]>valuesScaled[0][2]:
        trainingData5.append(data)
        trainingLabels5.append(value)
    elif data[1]>valuesScaled[0][1]:
        trainingData6.append(data)
        trainingLabels6.append(value)
    elif data[1]>valuesScaled[0][0]:
        trainingData7.append(data)
        trainingLabels7.append(value)
    else:
        trainingData8.append(data)
        trainingLabels8.append(value) 
    if data[2]>valuesScaled[0][2]:
        trainingData9.append(data)
        trainingLabels9.append(value)
    elif data[2]>valuesScaled[0][1]:
        trainingData10.append(data)
        trainingLabels10.append(value)
    elif data[2]>valuesScaled[0][0]:
        trainingData11.append(data)
        trainingLabels11.append(value)
    else:
        trainingData12.append(data)
        trainingLabels12.append(value)
for i in range(len(correctData)+len(wrongData)):
    if random.choice([-1,1])<0:
        if wrongDataIndex>0:
            decideWichDatasetToPut(wrongData[wrongDataIndex], 0)
            wrongDataIndex-=1
        elif correctDataIndex>0:
            decideWichDatasetToPut(correctData[correctDataIndex], 1)
            correctDataIndex-=1
    else:
        if correctDataIndex>0:
            decideWichDatasetToPut(correctData[correctDataIndex], 1)
            correctDataIndex-=1
        elif wrongDataIndex>0:
            decideWichDatasetToPut(wrongData[wrongDataIndex], 0)
            wrongDataIndex-= 1
# end data pre processing
```


```python
start_time = time.time()
epochs = 2000
rate = 0.005
trainingData = [
    trainingData1, 
    trainingData2, 
    trainingData3, 
    trainingData4, 
    trainingData5, 
    trainingData6, 
    trainingData7, 
    trainingData8, 
    trainingData9, 
    trainingData10,
    trainingData11,
    trainingData12
]
trainingLabels = [
    trainingLabels1,
    trainingLabels2,
    trainingLabels3,
    trainingLabels4,
    trainingLabels5,
    trainingLabels6,
    trainingLabels7,
    trainingLabels8,
    trainingLabels9,
    trainingLabels10,
    trainingLabels11,
    trainingLabels12
]
models = [
    RBPerceptron(epochs, rate, 1), 
    RBPerceptron(epochs, rate, 2), 
    RBPerceptron(epochs, rate, 3),
    RBPerceptron(epochs, rate, 4),
    RBPerceptron(epochs, rate, 5),
    RBPerceptron(epochs, rate, 6),
    RBPerceptron(epochs, rate, 7),
    RBPerceptron(epochs, rate, 8),
    RBPerceptron(epochs, rate, 9),
    RBPerceptron(epochs, rate, 10),
    RBPerceptron(epochs, rate, 11),
    RBPerceptron(epochs, rate, 12)
    ]
```


```python
for i in range(len(models)):
    models[i].train(np.array(trainingData[i]), np.array(trainingLabels[i]))
```

    1 - 0
    1 - 100
    1 - 200
    1 - 300
    1 - 400
    1 - 500
    1 - 600
    1 - 700
    1 - 800
    1 - 900
    1 - 1000
    1 - 1100
    1 - 1200
    1 - 1300
    1 - 1400
    1 - 1500
    1 - 1600
    1 - 1700
    1 - 1800
    1 - 1900
    2 - 0
    2 - 100
    2 - 200
    2 - 300
    2 - 400
    2 - 500
    2 - 600
    2 - 700
    2 - 800
    2 - 900
    2 - 1000
    2 - 1100
    2 - 1200
    2 - 1300
    2 - 1400
    2 - 1500
    2 - 1600
    2 - 1700
    2 - 1800
    2 - 1900
    3 - 0
    3 - 100
    3 - 200
    3 - 300
    3 - 400
    3 - 500
    3 - 600
    3 - 700
    3 - 800
    3 - 900
    3 - 1000
    3 - 1100
    3 - 1200
    3 - 1300
    3 - 1400
    3 - 1500
    3 - 1600
    3 - 1700
    3 - 1800
    3 - 1900
    4 - 0
    4 - 100
    4 - 200
    4 - 300
    4 - 400
    4 - 500
    4 - 600
    4 - 700
    4 - 800
    4 - 900
    4 - 1000
    4 - 1100
    4 - 1200
    4 - 1300
    4 - 1400
    4 - 1500
    4 - 1600
    4 - 1700
    4 - 1800
    4 - 1900
    5 - 0
    5 - 100
    5 - 200
    5 - 300
    5 - 400
    5 - 500
    5 - 600
    5 - 700
    5 - 800
    5 - 900
    5 - 1000
    5 - 1100
    5 - 1200
    5 - 1300
    5 - 1400
    5 - 1500
    5 - 1600
    5 - 1700
    5 - 1800
    5 - 1900
    6 - 0
    6 - 100
    6 - 200
    6 - 300
    6 - 400
    6 - 500
    6 - 600
    6 - 700
    6 - 800
    6 - 900
    6 - 1000
    6 - 1100
    6 - 1200
    6 - 1300
    6 - 1400
    6 - 1500
    6 - 1600
    6 - 1700
    6 - 1800
    6 - 1900
    7 - 0
    7 - 100
    7 - 200
    7 - 300
    7 - 400
    7 - 500
    7 - 600
    7 - 700
    7 - 800
    7 - 900
    7 - 1000
    7 - 1100
    7 - 1200
    7 - 1300
    7 - 1400
    7 - 1500
    7 - 1600
    7 - 1700
    7 - 1800
    7 - 1900
    8 - 0
    8 - 100
    8 - 200
    8 - 300
    8 - 400
    8 - 500
    8 - 600
    8 - 700
    8 - 800
    8 - 900
    8 - 1000
    8 - 1100
    8 - 1200
    8 - 1300
    8 - 1400
    8 - 1500
    8 - 1600
    8 - 1700
    8 - 1800
    8 - 1900
    9 - 0
    9 - 100
    9 - 200
    9 - 300
    9 - 400
    9 - 500
    9 - 600
    9 - 700
    9 - 800
    9 - 900
    9 - 1000
    9 - 1100
    9 - 1200
    9 - 1300
    9 - 1400
    9 - 1500
    9 - 1600
    9 - 1700
    9 - 1800
    9 - 1900
    10 - 0
    10 - 100
    10 - 200
    10 - 300
    10 - 400
    10 - 500
    10 - 600
    10 - 700
    10 - 800
    10 - 900
    10 - 1000
    10 - 1100
    10 - 1200
    10 - 1300
    10 - 1400
    10 - 1500
    10 - 1600
    10 - 1700
    10 - 1800
    10 - 1900
    11 - 0
    11 - 100
    11 - 200
    11 - 300
    11 - 400
    11 - 500
    11 - 600
    11 - 700
    11 - 800
    11 - 900
    11 - 1000
    11 - 1100
    11 - 1200
    11 - 1300
    11 - 1400
    11 - 1500
    11 - 1600
    11 - 1700
    11 - 1800
    11 - 1900
    12 - 0
    12 - 100
    12 - 200
    12 - 300
    12 - 400
    12 - 500
    12 - 600
    12 - 700
    12 - 800
    12 - 900
    12 - 1000
    12 - 1100
    12 - 1200
    12 - 1300
    12 - 1400
    12 - 1500
    12 - 1600
    12 - 1700
    12 - 1800
    12 - 1900
    


```python
def predict(data):
    if data[0]>valuesScaled[0][1]:
       highXPredict = models[0].predict(data) 
       lowerXPredict = models[1].predict(data) 
    else:
        highXPredict =models[2].predict(data) 
        lowerXPredict = models[3].predict(data) 
    if data[1]>valuesScaled[0][1]:
        highYPredict =models[4].predict(data) 
        lowerYPredict = models[5].predict(data)   
    else:
        highYPredict = models[6].predict(data) 
        lowerYPredict = models[7].predict(data)
    if data[2]>valuesScaled[0][1]:
        highZPredict = models[8].predict(data) 
        lowerZPredict = models[9].predict(data)
    else:
        highZPredict = models[10].predict(data) 
        lowerZPredict = models[11].predict(data)
    if highXPredict>0.5 and lowerXPredict>0.5 and highYPredict>0.5 and lowerYPredict>0.5 and highZPredict>0.5 and lowerZPredict>0:
        return 1
    else:
        return -1  
```


```python
truePositive = []
falsePositive = []
trueNegative = []
falseNegative = []
prediction = 0 
for i in range(len(correctTestingData)):
    sample = correctTestingData[i]
    correctPredictions = predict(sample)
    if correctPredictions >=0:
        truePositive.append(sample)
        prediction+=1
    else:
        falseNegative.append(sample)
for i in range(len(wrongTestingData)):
    sample = wrongTestingData[i]
    correctPredictions = predict(sample)
    if correctPredictions >=0:
        trueNegative.append(sample)
    else:
        falsePositive.append(sample)
((prediction/len(correctTestingData))*100)
```




    33.35




```python
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.set_xlabel('$X$', fontsize=20, rotation=150)
ax.set_ylabel('$Y$', fontsize=30, rotation=60)
ax.set_zlabel('$Z$', fontsize=30, rotation=60)
ax.yaxis._axinfo['label']['space_factor'] = 3.0
plt.rcParams['figure.figsize'] = [15, 15]
for i in range (round(len(truePositive)/5)):
    ax.scatter(truePositive[i][0], truePositive[i][1], truePositive[i][2], c='g', marker='o')
for i in range (round(len(trueNegative)/5)):
    ax.scatter(trueNegative[i][0], trueNegative[i][1], trueNegative[i][2], c='b', marker='^')
    ax.scatter(falsePositive[i][0], falsePositive[i][1], falsePositive[i][2], c='r', marker='o')
for i in range (round(len(falseNegative)/5)):
    ax.scatter(falseNegative[i][0], falseNegative[i][1], falseNegative[i][2], c='r', marker='^')
plt.show()
```


    
![png](output_9_0.png)
    



```python

```
