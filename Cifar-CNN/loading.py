from keras.models import load_model
import h5py
from keras.callbacks import EarlyStopping
from keras.datasets import cifar10
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Flatten
from keras.layers.convolutional import Conv2D
from keras.optimizers import Adam
from keras.layers.pooling import MaxPooling2D
from keras.utils import to_categorical

(X_train, Y_train), (X_test, Y_test) = cifar10.load_data()

model = load_model('cifar10.h5')

scores = model.evaluate(X_test / 255.0, to_categorical(Y_test))
print('Loss: %.3f' % scores[0])
print('Accuracy: %.3f' % scores[1])
