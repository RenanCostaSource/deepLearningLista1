import numpy as np
import random
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