from keras.datasets import mnist
from keras import models
from keras import layers
from tensorflow.keras.utils import to_categorical

(train_X,train_Y), (test_X,test_Y) = mnist.load_data()

network = models.Sequential()
network.add(layers.Dense(512,activation='relu', input_shape=(28*28,)))
network.add(layers.Dense(10,activation='softmax'))

network.compile(optimizer='rmsprop',
                 loss='categorical_crossentropy',
                 metrics=['accuracy'])

train_X = train_X.reshape((60000, 28 * 28))
train_X = train_X.astype('float32') / 255

test_X = test_X.reshape((10000, 28 * 28))
test_X = test_X.astype('float32') / 255

train_Y = to_categorical(train_Y)
test_Y = to_categorical(test_Y)

network.fit(train_X,train_Y,epochs=5, batch_size=128)

test_loss, test_acc = network.evaluate(test_X, test_Y)
print('test_acc: ',test_acc)
