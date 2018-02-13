import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.datasets import mnist
import matplotlib.pyplot as plt

(x_train, y_train), (x_test, y_test) = mnist.load_data()

pixel_num = x_train.shape[1] * x_train.shape[2]
x_train = x_train.reshape(x_train.shape[0], pixel_num).astype('float32')
x_test = x_test.reshape(x_test.shape[0], pixel_num).astype('float32')

# normalize to 0-1
x_train /= 255
x_test /= 255

classes=10
y_train = keras.utils.to_categorical(y_train, classes)
y_test = keras.utils.to_categorical(y_test, classes)

model = Sequential()

model.add(Dense(input_dim=pixel_num,units=500,activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(units=500,activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(units=classes,activation='softmax'))


model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(x_train, y_train, epochs=10, batch_size=200, verbose=2)



scores = model.evaluate(x_test, y_test)
print('Loss', scores[0])
print('Accuracy', scores[1])

result = model.predict(x_test)
print(result)

