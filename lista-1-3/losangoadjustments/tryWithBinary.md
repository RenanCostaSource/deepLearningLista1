```python
from tensorflow.keras.optimizers import SGD
import matplotlib.pyplot as plt
import math
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import tensorflow
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import to_categorical
import sklearn.metrics as metrics
import seaborn as sns
import pandas as pd
```


```python
with open('data.npy', 'rb') as f:
    data = np.load(f)
    dataLabels = np.load(f)
testData = data[math.floor(len(data)*0.8):]
data = data[:math.floor(len(data)*0.8)]
testDataLabels = dataLabels[math.floor(len(dataLabels)*0.8):]
dataLabels = dataLabels[:math.floor(len(dataLabels)*0.8)]
dataset = tensorflow.data.Dataset.from_tensor_slices((data, dataLabels))
len(data)
```




    200000




```python
model = Sequential()
model.add(Dense(12, input_shape=(2,), kernel_initializer='normal', activation='relu'))
model.add(Dense(16, activation='sigmoid', kernel_initializer='normal'))
model.add(Dense(4, activation='sigmoid', kernel_initializer='normal'))
model.summary()
```

    Model: "sequential"
    _________________________________________________________________
     Layer (type)                Output Shape              Param #   
    =================================================================
     dense (Dense)               (None, 12)                36        
                                                                     
     dense_1 (Dense)             (None, 16)                208       
                                                                     
     dense_2 (Dense)             (None, 4)                 68        
                                                                     
    =================================================================
    Total params: 312
    Trainable params: 312
    Non-trainable params: 0
    _________________________________________________________________
    


```python
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['binary_accuracy']) 
history = model.fit(data, dataLabels, epochs=30, batch_size=50, verbose=1,validation_split = 0.2)
```

    Epoch 1/30
    3200/3200 [==============================] - 3s 861us/step - loss: 0.3155 - binary_accuracy: 0.8468 - val_loss: 0.1800 - val_binary_accuracy: 0.9093
    Epoch 2/30
    3200/3200 [==============================] - 3s 847us/step - loss: 0.1460 - binary_accuracy: 0.9316 - val_loss: 0.1060 - val_binary_accuracy: 0.9596
    Epoch 3/30
    3200/3200 [==============================] - 3s 831us/step - loss: 0.0772 - binary_accuracy: 0.9740 - val_loss: 0.0551 - val_binary_accuracy: 0.9834
    Epoch 4/30
    3200/3200 [==============================] - 3s 789us/step - loss: 0.0448 - binary_accuracy: 0.9860 - val_loss: 0.0393 - val_binary_accuracy: 0.9870
    Epoch 5/30
    3200/3200 [==============================] - 3s 822us/step - loss: 0.0353 - binary_accuracy: 0.9882 - val_loss: 0.0331 - val_binary_accuracy: 0.9888
    Epoch 6/30
    3200/3200 [==============================] - 3s 816us/step - loss: 0.0301 - binary_accuracy: 0.9899 - val_loss: 0.0289 - val_binary_accuracy: 0.9905
    Epoch 7/30
    3200/3200 [==============================] - 3s 824us/step - loss: 0.0266 - binary_accuracy: 0.9912 - val_loss: 0.0258 - val_binary_accuracy: 0.9920
    Epoch 8/30
    3200/3200 [==============================] - 3s 795us/step - loss: 0.0238 - binary_accuracy: 0.9922 - val_loss: 0.0234 - val_binary_accuracy: 0.9923
    Epoch 9/30
    3200/3200 [==============================] - 3s 793us/step - loss: 0.0216 - binary_accuracy: 0.9930 - val_loss: 0.0212 - val_binary_accuracy: 0.9934
    Epoch 10/30
    3200/3200 [==============================] - 2s 773us/step - loss: 0.0198 - binary_accuracy: 0.9938 - val_loss: 0.0195 - val_binary_accuracy: 0.9938
    Epoch 11/30
    3200/3200 [==============================] - 2s 768us/step - loss: 0.0183 - binary_accuracy: 0.9944 - val_loss: 0.0184 - val_binary_accuracy: 0.9942
    Epoch 12/30
    3200/3200 [==============================] - 2s 774us/step - loss: 0.0170 - binary_accuracy: 0.9948 - val_loss: 0.0174 - val_binary_accuracy: 0.9948
    Epoch 13/30
    3200/3200 [==============================] - 2s 771us/step - loss: 0.0160 - binary_accuracy: 0.9953 - val_loss: 0.0161 - val_binary_accuracy: 0.9949
    Epoch 14/30
    3200/3200 [==============================] - 2s 772us/step - loss: 0.0151 - binary_accuracy: 0.9954 - val_loss: 0.0154 - val_binary_accuracy: 0.9952
    Epoch 15/30
    3200/3200 [==============================] - 2s 777us/step - loss: 0.0144 - binary_accuracy: 0.9956 - val_loss: 0.0145 - val_binary_accuracy: 0.9957
    Epoch 16/30
    3200/3200 [==============================] - 3s 807us/step - loss: 0.0137 - binary_accuracy: 0.9958 - val_loss: 0.0142 - val_binary_accuracy: 0.9957
    Epoch 17/30
    3200/3200 [==============================] - 3s 782us/step - loss: 0.0132 - binary_accuracy: 0.9960 - val_loss: 0.0140 - val_binary_accuracy: 0.9954
    Epoch 18/30
    3200/3200 [==============================] - 3s 782us/step - loss: 0.0126 - binary_accuracy: 0.9962 - val_loss: 0.0130 - val_binary_accuracy: 0.9957
    Epoch 19/30
    3200/3200 [==============================] - 3s 798us/step - loss: 0.0122 - binary_accuracy: 0.9963 - val_loss: 0.0125 - val_binary_accuracy: 0.9961
    Epoch 20/30
    3200/3200 [==============================] - 3s 812us/step - loss: 0.0119 - binary_accuracy: 0.9963 - val_loss: 0.0124 - val_binary_accuracy: 0.9961
    Epoch 21/30
    3200/3200 [==============================] - 3s 803us/step - loss: 0.0115 - binary_accuracy: 0.9965 - val_loss: 0.0120 - val_binary_accuracy: 0.9959
    Epoch 22/30
    3200/3200 [==============================] - 3s 873us/step - loss: 0.0112 - binary_accuracy: 0.9965 - val_loss: 0.0121 - val_binary_accuracy: 0.9957
    Epoch 23/30
    3200/3200 [==============================] - 3s 789us/step - loss: 0.0109 - binary_accuracy: 0.9965 - val_loss: 0.0113 - val_binary_accuracy: 0.9962
    Epoch 24/30
    3200/3200 [==============================] - 3s 787us/step - loss: 0.0106 - binary_accuracy: 0.9966 - val_loss: 0.0109 - val_binary_accuracy: 0.9964
    Epoch 25/30
    3200/3200 [==============================] - 3s 792us/step - loss: 0.0104 - binary_accuracy: 0.9966 - val_loss: 0.0107 - val_binary_accuracy: 0.9964
    Epoch 26/30
    3200/3200 [==============================] - 3s 785us/step - loss: 0.0102 - binary_accuracy: 0.9966 - val_loss: 0.0106 - val_binary_accuracy: 0.9966
    Epoch 27/30
    3200/3200 [==============================] - 2s 771us/step - loss: 0.0099 - binary_accuracy: 0.9967 - val_loss: 0.0102 - val_binary_accuracy: 0.9965
    Epoch 28/30
    3200/3200 [==============================] - 3s 875us/step - loss: 0.0097 - binary_accuracy: 0.9968 - val_loss: 0.0103 - val_binary_accuracy: 0.9964
    Epoch 29/30
    3200/3200 [==============================] - 3s 787us/step - loss: 0.0096 - binary_accuracy: 0.9968 - val_loss: 0.0101 - val_binary_accuracy: 0.9964
    Epoch 30/30
    3200/3200 [==============================] - 2s 775us/step - loss: 0.0094 - binary_accuracy: 0.9968 - val_loss: 0.0100 - val_binary_accuracy: 0.9965
    


```python
plt.plot(history.history['binary_accuracy'])
plt.plot(history.history['val_binary_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
```


    
![png](output_4_0.png)
    



```python
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
```


    
![png](output_5_0.png)
    



```python
test_results = model.evaluate(testData, testDataLabels, verbose=1)
print(f'Test results - Loss: {test_results[0]} - Accuracy: {test_results[1]}%')
```

    1563/1563 [==============================] - 1s 894us/step - loss: 0.0095 - binary_accuracy: 0.9968
    Test results - Loss: 0.00946503970772028 - Accuracy: 0.9967650175094604%
    

### Test results - Loss: 0.006966711021959782 - Accuracy: 0.9974750280380249%


```python
groupsTestx = [[],[],[],[],[],[],[],[],[]]
groupsTesty = [[],[],[],[],[],[],[],[],[]]
testData = data
testDataLabels = dataLabels
for i in range(len(testData)):
    if(testDataLabels[i][3]>=0.5):
        groupsTestx[8].append(testData[i][0])
        groupsTesty[8].append(testData[i][1])
    elif(testDataLabels[i][0]>=0.5):
        if testDataLabels[i][1] >= 0.5:
            if testDataLabels[i][2]>=0.5:
                #111
                groupsTestx[6].append(testData[i][0])
                groupsTesty[6].append(testData[i][1])
            else:
                #110
                groupsTestx[7].append(testData[i][0])
                groupsTesty[7].append(testData[i][1])
        else:
            if testDataLabels[i][2]>=0.5:
                #101
                groupsTestx[5].append(testData[i][0])
                groupsTesty[5].append(testData[i][1])
            else:
                #100
                groupsTestx[4].append(testData[i][0])
                groupsTesty[4].append(testData[i][1])
    else:
        if testDataLabels[i][1] >= 0.5:
            if testDataLabels[i][2]>=0.5:
                #011
                groupsTestx[2].append(testData[i][0])
                groupsTesty[2].append(testData[i][1])
            else:
                #010
                groupsTestx[3].append(testData[i][0])
                groupsTesty[3].append(testData[i][1])
        else:
            if testDataLabels[i][2]>=0.5:
                #001
                groupsTestx[1].append(testData[i][0])
                groupsTesty[1].append(testData[i][1])
            else:
                #000
                groupsTestx[0].append(testData[i][0])
                groupsTesty[0].append(testData[i][1])
                
plt.rcParams['figure.figsize'] = [15, 15]
plt.scatter(groupsTestx[0],groupsTesty[0], c='b')
plt.scatter(groupsTestx[1],groupsTesty[1], c='m')
plt.scatter(groupsTestx[2],groupsTesty[2], c='c')
plt.scatter(groupsTestx[3],groupsTesty[3], c='r')
plt.scatter(groupsTestx[4],groupsTesty[4], c='g')
plt.scatter(groupsTestx[5],groupsTesty[5], c='y')
plt.scatter(groupsTestx[6],groupsTesty[6], c='k')
plt.scatter(groupsTestx[7],groupsTesty[7], c='tab:orange')
plt.scatter(groupsTestx[8],groupsTesty[8], c='tab:purple')
plt.show()
```


    
![png](output_8_0.png)
    



```python
testx = np.linspace(-1.5,1.5,200)
testy = np.linspace(-1.5,1.5,200)
dataTest=[]
for i in range(200):
  for j in range(200):
      dataTest.append([testx[i], testy[j]])  
prediction = []

prediction=(model.predict(np.array(dataTest)))
groupsTestx = [[],[],[],[],[],[],[],[],[]]
groupsTesty = [[],[],[],[],[],[],[],[], []]
for i in range(len(dataTest)):
    if(prediction[i][3]>=0.5):
        groupsTestx[8].append(dataTest[i][0])
        groupsTesty[8].append(dataTest[i][1])
    elif(prediction[i][0]>=0.5):
        if prediction[i][1] >= 0.5:
            if prediction[i][2]>=0.5:
                #111
                groupsTestx[6].append(dataTest[i][0])
                groupsTesty[6].append(dataTest[i][1])
            else:
                #110
                groupsTestx[7].append(dataTest[i][0])
                groupsTesty[7].append(dataTest[i][1])
        else:
            if prediction[i][2]>=0.5:
                #101
                groupsTestx[5].append(dataTest[i][0])
                groupsTesty[5].append(dataTest[i][1])
            else:
                #100
                groupsTestx[4].append(dataTest[i][0])
                groupsTesty[4].append(dataTest[i][1])
    else:
        if prediction[i][1] >= 0.5:
            if prediction[i][2]>=0.5:
                #011
                groupsTestx[2].append(dataTest[i][0])
                groupsTesty[2].append(dataTest[i][1])
            else:
                #010
                groupsTestx[3].append(dataTest[i][0])
                groupsTesty[3].append(dataTest[i][1])
        else:
            if prediction[i][2]>=0.5:
                #001
                groupsTestx[1].append(dataTest[i][0])
                groupsTesty[1].append(dataTest[i][1])
            else:
                #000
                groupsTestx[0].append(dataTest[i][0])
                groupsTesty[0].append(dataTest[i][1])
                
plt.rcParams['figure.figsize'] = [15, 15]
plt.scatter(groupsTestx[0],groupsTesty[0], c='b')
plt.scatter(groupsTestx[1],groupsTesty[1], c='m')
plt.scatter(groupsTestx[2],groupsTesty[2], c='c')
plt.scatter(groupsTestx[3],groupsTesty[3], c='r')
plt.scatter(groupsTestx[4],groupsTesty[4], c='g')
plt.scatter(groupsTestx[5],groupsTesty[5], c='y')
plt.scatter(groupsTestx[6],groupsTesty[6], c='k')
plt.scatter(groupsTestx[7],groupsTesty[7], c='tab:orange')
plt.scatter(groupsTestx[8],groupsTesty[8], c='tab:purple')
plt.show()
```


    
![png](output_9_0.png)
    



```python
testPrediction = model.predict(testData)
```


```python
labels = ["fora circulo","fora triangulo","y>0","x<0"]
cm = metrics.multilabel_confusion_matrix((testDataLabels>0.5), (testPrediction>0.5))
```


```python
def print_confusion_matrix(confusion_matrix, axes, class_label, class_names, fontsize=14):

    df_cm = pd.DataFrame(
        confusion_matrix, index=class_names, columns=class_names,
    )

    try:
        heatmap = sns.heatmap(df_cm, annot=True, fmt="d", cbar=False, ax=axes)
    except ValueError:
        raise ValueError("Confusion matrix values must be integers.")
    heatmap.yaxis.set_ticklabels(heatmap.yaxis.get_ticklabels(), rotation=0, ha='right', fontsize=fontsize)
    heatmap.xaxis.set_ticklabels(heatmap.xaxis.get_ticklabels(), rotation=45, ha='right', fontsize=fontsize)
    axes.set_ylabel('True label')
    axes.set_xlabel('Predicted label')
    axes.set_title("Confusion Matrix for the class - " + class_label)
    
fig, ax = plt.subplots(4, 1, figsize=(12, 12))
    
for axes, cfs_matrix, label in zip(ax.flatten(), cm, labels):
    print_confusion_matrix(cfs_matrix, axes, label, ["True", "False"])
    
fig.tight_layout()
plt.show()
```


    
![png](output_12_0.png)
    



```python

```
