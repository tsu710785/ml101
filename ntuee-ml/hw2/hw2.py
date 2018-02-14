import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.datasets import mnist
import matplotlib.pyplot as plt
import numpy as np
from numpy import genfromtxt

X_train = genfromtxt('X_train', delimiter=',')
Y_train = genfromtxt('Y_train', delimiter=',')
X_test = genfromtxt('X_test', delimiter=',')

X_train = np.array(X_train[1:])
Y_train = np.array(Y_train[1:])
X_test = np.array(X_test[1:])

dim = len(X_train[0])

model = Sequential()
classes=2
Y_train = keras.utils.to_categorical(Y_train, classes)


model.add(Dense(input_dim=dim,units=500,activation='tanh'))
model.add(Dropout(0.2))
model.add(Dense(units=500,activation='tanh'))
model.add(Dropout(0.2))

model.add(Dense(units=classes,activation='softmax'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()
model.fit(X_train, Y_train, epochs=10, batch_size=200, verbose=2,validation_split=0.05)

Y_test=model.predict_classes(X_test, verbose=1)
print(Y_test)
with open('Y_test', 'w') as f:
    f.write('id,label\n')
    for i, l in enumerate(Y_test):
        f.write('%d,%d\n' % ((i+1),l))
