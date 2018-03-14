import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K

from input_xy_data import xy_data
xyd=xy_data()
xyd.read_in(5)

batch_size = 128
num_classes = 2
epochs = 30

# input image dimensions
img_rows, img_cols = 36, 36

# the data, split between train and test sets
#(x_train, y_train), (x_test, y_test) = mnist.load_data()
xyd.shape(img_rows)
xyd.make_batch(batch_size)
_, x_train, y_train =xyd.out_perm()
x_test,y_test=xyd.out_test()

print("\nx_train shape: " + str(x_train.shape))
print("y_train shape: " + str(y_train.shape))
print("x_test shape: " + str(x_test.shape))
print("y_test shape: " + str(y_test.shape))

x_test_good,y_test_good=xyd.out_test_good(0,500)
print("x_test_good shape= " +  str(x_test_good.shape))
print("y_test_good shape= " +  str(y_test_good.shape))

x_test_bad,y_test_bad=xyd.out_test_bad(0,81)
print("x_test_bad shape= " +  str(x_test_bad.shape))
print("y_test_bad shape= " +  str(y_test_bad.shape))

input_shape = (img_rows, img_cols, 2)

model = Sequential()
model.add(Conv2D(32, kernel_size=(5, 5),
                 activation='relu',
                 input_shape=input_shape))
model.add(Conv2D(64, (5, 5), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(num_classes, activation='softmax'))

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])

model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(x_test, y_test))

score = model.evaluate(x_test, y_test, verbose=0)
print('\nTest loss:', score[0])
print('Test accuracy:', score[1])

scoreg = model.evaluate(x_test_good, y_test_good, verbose=0)
print('\nTest good loss:', scoreg[0])
print('Test good accuracy:', scoreg[1])

scoreb = model.evaluate(x_test_bad, y_test_bad, verbose=0)
print('\nTest bad loss:', scoreb[0])
print('Test bad accuracy:', scoreb[1])




