```python
import numpy as np
```


```python
def sigmoid(x):
	return 1.0/(1.0 + np.exp(-x))

def sigmoid_der(x):
	return x*(1.0 - x)

```


```python
class NN:
    def __init__(self, inputs):
        self.inputs = inputs
        self.l=len(self.inputs)
        self.li=len(self.inputs[0])
        self.wi=np.random.random((self.li, self.l))
        self.wh=np.random.random((self.l, 1))

    def predict(self, inp):
        s1=sigmoid(np.dot(inp, self.wi))
        s2=sigmoid(np.dot(s1, self.wh))
        return s2

    def train(self, inputs,outputs, it):
        for i in range(it):
            l0=inputs
            l1=sigmoid(np.dot(l0, self.wi))
            l2=sigmoid(np.dot(l1, self.wh))
            l2_err=outputs - l2
            l2_delta = np.multiply(l2_err, sigmoid_der(l2))
            l1_err=np.dot(l2_delta, self.wh.T)
            l1_delta=np.multiply(l1_err, sigmoid_der(l1))
            self.wh+=np.dot(l1.T, l2_delta)
            self.wi+=np.dot(l0.T, l1_delta)
```


```python
inputs=np.array([[0.1,0.1], [0.1,1.1], [1.1,0.0], [1.1,1.1] ])
outputs=np.array([ [0], [1],[1],[0] ])
```

# Predição antes do treinamento


```python
n=NN(inputs)
n.predict(inputs)
```




    array([[0.79033984],
           [0.84094325],
           [0.82070941],
           [0.86489648]])



# Predição depois do treinamento


```python
n.train(inputs, outputs, 10000)
n.predict(inputs)
```




    array([[0.01695659],
           [0.98367841],
           [0.98334581],
           [0.01411782]])


