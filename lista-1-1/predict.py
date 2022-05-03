from perceptron import RBPerceptron
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import joblib
import random
scaler_filename = "scaler.save"
dataSize = 10
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

models = [
    RBPerceptron(), 
    RBPerceptron(), 
    RBPerceptron(),
    RBPerceptron(),
    RBPerceptron(),
    RBPerceptron(),
    RBPerceptron(),
    RBPerceptron(),
    RBPerceptron(),
    RBPerceptron(),
    RBPerceptron(),
    RBPerceptron()
    ]
with open('weights.npy', 'rb') as f:
    for i in range(12):
        models[i].w = np.load(f)

scaler = joblib.load(scaler_filename)
correctData = scaler.transform(np.array(correctData[:80]))
wrongData = scaler.transform(np.array(wrongData[:80]))
valuesScaled = scaler.transform([[0,0.5,1]])
def predict(data):
    if data[0]>valuesScaled[0][1]:
       highXPredict =models[0].predict(data) 
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
    print("X")
    print(highXPredict)
    print(lowerXPredict)
    print("Y")
    print(highYPredict)
    print(lowerYPredict)
    print("Z")
    print(highZPredict)
    print(lowerZPredict)
    if highXPredict>0 and lowerXPredict>0 and highYPredict>0 and lowerYPredict>0 and highZPredict>0 and lowerZPredict>0:
        return 1
    else:
        return -1   

truePositive = []
falsePositive = []
trueNegative = []
falseNegative = []
print('should all be =>0')
prediction = 0 
for i in range(len(correctData)):
    sample = correctData[i]
    correctPredictions = predict(sample)
    if correctPredictions >=0:
        truePositive.append(sample)
        prediction+=1
    else:
        falseNegative.append(sample)

print('should all be <0')
for i in range(len(wrongData)):
    sample = wrongData[i]
    correctPredictions = predict(sample)
    if correctPredictions >=0:
        trueNegative.append(sample)
    else:
        falsePositive.append(sample)
print(prediction/len(correctData)*100)