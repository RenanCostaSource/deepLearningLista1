import random
from sklearn.preprocessing import normalize
from sklearn.preprocessing import MinMaxScaler
import joblib
from joblib import Parallel, delayed  
import multiprocessing
import time
import numpy as np
import matplotlib.pyplot as plt
from mlxtend.plotting import plot_decision_regions
from mpl_toolkits.mplot3d import Axes3D
from perceptron import RBPerceptron
scaler_filename = "scaler.save"
with open('correct.npy', 'rb') as f:
    correctData = np.load(f)
with open('wrong.npy', 'rb') as f:
    wrongData = np.load(f)
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
Parallel(n_jobs=4)(delayed(models[i].train(np.array(trainingData[i]), np.array(trainingLabels[i]))) for i in range(12))
"""model1 = RBPerceptron(epochs, rate, 1)
model2 = RBPerceptron(epochs, rate, 2)
model3 = RBPerceptron(epochs, rate, 3)
model4 = RBPerceptron(epochs, rate, 4)
model5 = RBPerceptron(epochs, rate, 5)
model6 = RBPerceptron(epochs, rate, 6)
model7 = RBPerceptron(epochs, rate, 7)
model8 = RBPerceptron(epochs, rate, 8)
model9 = RBPerceptron(epochs, rate, 9)
model10 = RBPerceptron(epochs, rate, 10)
model11 = RBPerceptron(epochs, rate, 11)
model12 = RBPerceptron(epochs, rate, 12)
model1.train(np.array(trainingData1), trainingLabels1)
model2.train(np.array(trainingData2), trainingLabels2)
model3.train(np.array(trainingData3), trainingLabels3)
model4.train(np.array(trainingData4), trainingLabels4)
model5.train(np.array(trainingData5), trainingLabels5)
model6.train(np.array(trainingData6), trainingLabels6)
model7.train(np.array(trainingData7), trainingLabels7)
model8.train(np.array(trainingData8), trainingLabels8)
model9.train(np.array(trainingData9), trainingLabels9)
model10.train(np.array(trainingData10), trainingLabels10)
model11.train(np.array(trainingData11), trainingLabels11)
model12.train(np.array(trainingData12), trainingLabels12)
def predict(data):
    if data[0]>valuesScaled[0][1]:
       highXPredict =model1.predict(data) 
       lowerXPredict = model2.predict(data) 
    else:
        highXPredict =model3.predict(data) 
        lowerXPredict = model4.predict(data) 
    if data[1]>valuesScaled[0][1]:
        highYPredict =model5.predict(data) 
        lowerYPredict = model6.predict(data)   
    else:
        highYPredict = model7.predict(data) 
        lowerYPredict = model8.predict(data)
    if data[2]>valuesScaled[0][1]:
        highZPredict = model9.predict(data) 
        lowerZPredict = model10.predict(data)
    else:
        highZPredict = model11.predict(data) 
        lowerZPredict = model12.predict(data)
    if highXPredict>0 and lowerXPredict>0 and highYPredict>0 and lowerYPredict>0 and highZPredict>0 and lowerZPredict>0:
        return 1
    else:
        return -1
 """      
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
    if highXPredict>0 and lowerXPredict>0 and highYPredict>0 and lowerYPredict>0 and highZPredict>0 and lowerZPredict>0:
        return 1
    else:
        return -1    
print("--- %s seconds ---" % (time.time() - start_time))
truePositive = []
falsePositive = []
trueNegative = []
falseNegative = []
print('should all be =>0')
prediction = 0 
for i in range(len(correctTestingData)):
    sample = correctTestingData[i]
    correctPredictions = predict(sample)
    print(correctPredictions)
    if correctPredictions >=0:
        truePositive.append(sample)
        prediction+=1
    else:
        falseNegative.append(sample)

print('should all be <0')
for i in range(len(wrongTestingData)):
    sample = wrongTestingData[i]
    correctPredictions = predict(sample)
    if correctPredictions >=0:
        trueNegative.append(sample)
    else:
        falsePositive.append(sample)
valuePredicted = (prediction/len(correctTestingData)*100)
if valuePredicted > 90:
    with open('weights.npy', 'wb') as f:
        np.save(f, model1.w)
        np.save(f, model2.w)
        np.save(f, model3.w)
        np.save(f, model4.w)
        np.save(f, model5.w)
        np.save(f, model6.w)
        np.save(f, model7.w)
        np.save(f, model8.w)
        np.save(f, model9.w)
        np.save(f, model10.w)
        np.save(f, model11.w)
        np.save(f, model12.w)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.set_xlabel('$X$', fontsize=20, rotation=150)
ax.set_ylabel('$Y$', fontsize=30, rotation=60)
ax.set_zlabel('$Z$', fontsize=30, rotation=60)
ax.yaxis._axinfo['label']['space_factor'] = 3.0
for i in range (round(len(truePositive)/5)):
    ax.scatter(truePositive[i][0], truePositive[i][1], truePositive[i][2], c='g', marker='o')
for i in range (round(len(trueNegative)/5)):
    ax.scatter(trueNegative[i][0], trueNegative[i][1], trueNegative[i][2], c='g', marker='^')
for i in range (round(len(falsePositive)/5)):
    print(falsePositive[i])
    ax.scatter(falsePositive[i][0], falsePositive[i][1], falsePositive[i][2], c='r', marker='o')
for i in range (round(len(falseNegative)/5)):
    ax.scatter(falseNegative[i][0], falseNegative[i][1], falseNegative[i][2], c='r', marker='^')
plt.show()