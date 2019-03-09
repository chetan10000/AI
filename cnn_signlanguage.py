import numpy as np # linear algebra
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from keras.utils.np_utils import to_categorical # convert to one-hot-encoding
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D, BatchNormalization,MaxPooling2D,Dropout
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import LearningRateScheduler
import tensorflow as tf
from keras.models import load_model

train_file = "train.csv"
test_file = "test.csv"
output_file = "submission.csv"

raw_data = np.loadtxt(train_file, skiprows=1, dtype='int', delimiter=',')
x_train, x_test, y_train, y_test = train_test_split(
    raw_data[:,1:], raw_data[:,0], test_size=0.1)


x_train = x_train / 255
x_test = x_test / 255
x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)

model = Sequential()
model.add(Conv2D(64, kernel_size=(3,3), activation = 'relu', input_shape=(28, 28 ,1) ))
model.add(MaxPooling2D(pool_size = (2, 2)))

model.add(Conv2D(64, kernel_size = (3, 3), activation = 'relu'))
model.add(MaxPooling2D(pool_size = (2, 2)))

model.add(Conv2D(64, kernel_size = (3, 3), activation = 'relu'))
model.add(MaxPooling2D(pool_size = (2, 2)))

model.add(Flatten())
model.add(Dense(128, activation = 'relu'))
model.add(Dropout(0.20))
model.add(Dense(25, activation = tf.nn.softmax))
model.compile(loss = 'sparse_categorical_crossentropy', optimizer='adam',
              metrics=['accuracy'])
model.fit(x_train, y_train, validation_data = (x_test, y_test), epochs=3, batch_size=32)
class_names=["A","B","C","D","E","F","G","H","I","J","K","L","M","N","O","P","Q","R","S","T","U","V","W","X","y","Z"]
pred=model.predict([x_test])
i=np.argmax(pred[0])
guess_word=class_names[i]
print(guess_word)
plt.imshow(x_test[0].reshape(28,28),cmap='gray')
plt.show()


  

