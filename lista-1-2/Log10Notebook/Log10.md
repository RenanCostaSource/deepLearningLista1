```python
import numpy as np
import math
import random
import matplotlib.pyplot as plt
import tensorflow
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import to_categorical
```


```python
inputs=[]
outputs=[]
dataSize = 5000
for i in range(dataSize):
        x = random.uniform(1, 10)
        inputs.append([x])
        outputs.append([math.log(x, 10)])
trainingSet = np.array(inputs[:math.floor(len(inputs)*0.8)])
trainingLabels = np.array(outputs[:math.floor(len(outputs)*0.8)])
testingSet = np.array(inputs[math.floor(len(inputs)*0.8):])
testingLabels = np.array(outputs[math.floor(len(outputs)*0.8):])
```


```python
model = Sequential()
model.add(Dense(2, input_shape=(1,), kernel_initializer='normal', activation='sigmoid'))
model.add(Dense(1, activation='linear', kernel_initializer='normal'))
model.summary()
```

    Model: "sequential"
    _________________________________________________________________
     Layer (type)                Output Shape              Param #   
    =================================================================
     dense (Dense)               (None, 2)                 4         
                                                                     
     dense_1 (Dense)             (None, 1)                 3         
                                                                     
    =================================================================
    Total params: 7
    Trainable params: 7
    Non-trainable params: 0
    _________________________________________________________________
    


```python
model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mean_squared_error']) 
history = model.fit(trainingSet, trainingLabels, epochs=20, batch_size=5, verbose=1)
```

    Epoch 1/20
    800/800 [==============================] - 1s 647us/step - loss: 0.1266 - mean_squared_error: 0.1266
    Epoch 2/20
    800/800 [==============================] - 1s 633us/step - loss: 0.0305 - mean_squared_error: 0.0305
    Epoch 3/20
    800/800 [==============================] - 1s 646us/step - loss: 0.0166 - mean_squared_error: 0.0166
    Epoch 4/20
    800/800 [==============================] - 1s 668us/step - loss: 0.0061 - mean_squared_error: 0.0061
    Epoch 5/20
    800/800 [==============================] - 1s 714us/step - loss: 0.0021 - mean_squared_error: 0.0021
    Epoch 6/20
    800/800 [==============================] - 1s 630us/step - loss: 0.0012 - mean_squared_error: 0.0012
    Epoch 7/20
    800/800 [==============================] - 0s 590us/step - loss: 0.0011 - mean_squared_error: 0.0011
    Epoch 8/20
    800/800 [==============================] - 1s 652us/step - loss: 0.0010 - mean_squared_error: 0.0010
    Epoch 9/20
    800/800 [==============================] - 1s 692us/step - loss: 7.8392e-04 - mean_squared_error: 7.8392e-04
    Epoch 10/20
    800/800 [==============================] - 1s 681us/step - loss: 4.9392e-04 - mean_squared_error: 4.9392e-04
    Epoch 11/20
    800/800 [==============================] - 1s 724us/step - loss: 2.9224e-04 - mean_squared_error: 2.9224e-04
    Epoch 12/20
    800/800 [==============================] - 1s 687us/step - loss: 1.6614e-04 - mean_squared_error: 1.6614e-04
    Epoch 13/20
    800/800 [==============================] - 1s 677us/step - loss: 9.3609e-05 - mean_squared_error: 9.3609e-05
    Epoch 14/20
    800/800 [==============================] - 1s 721us/step - loss: 5.9429e-05 - mean_squared_error: 5.9429e-05
    Epoch 15/20
    800/800 [==============================] - 1s 725us/step - loss: 4.2563e-05 - mean_squared_error: 4.2563e-05
    Epoch 16/20
    800/800 [==============================] - 1s 666us/step - loss: 3.3782e-05 - mean_squared_error: 3.3782e-05
    Epoch 17/20
    800/800 [==============================] - 1s 631us/step - loss: 2.9511e-05 - mean_squared_error: 2.9511e-05
    Epoch 18/20
    800/800 [==============================] - 1s 627us/step - loss: 2.7531e-05 - mean_squared_error: 2.7531e-05
    Epoch 19/20
    800/800 [==============================] - 1s 629us/step - loss: 2.6425e-05 - mean_squared_error: 2.6425e-05
    Epoch 20/20
    800/800 [==============================] - 1s 642us/step - loss: 2.6352e-05 - mean_squared_error: 2.6352e-05
    


```python
test_results = model.evaluate(testingSet, testingLabels, verbose=1)
print(f'Test results - Loss: {test_results[0]} - Accuracy: {test_results[1]}%')
```

### Test results - Loss: 2.934691474365536e-05


```python
# summarize history for loss
plt.plot(history.history['loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
```


    
![png](output_6_0.png)
    



```python
x = np.linspace(1,10,100)
ypredict = (model.predict(x))
y = np.log10(x)
plt.plot(x,ypredict, 'r', ls = '-')
plt.plot(x,y, 'b', ls = '-')
plt.show()
```


    
![png](output_7_0.png)
    



```python

```
