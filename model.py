import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Flatten, Lambda, Activation, MaxPooling2D, Dropout
from keras.layers.convolutional import Convolution2D
from keras.optimizers import Adam
import hDrvUtils as hU

tf.python.control_flow_ops = tf

nb_epochs  = 17
nb_samples = 20083
nb_validtns = 6400
learnRate = 1e-3
Relu  = 'relu'
SoftM = 'softmax'

targetDir = hU.tDir
batchSize = 64*12

"""
Train the weights of our network to min( MSE )  the steering command output by the network & 
the command of either the human driver, or the adjusted steering command for off-center and rotated images 
9 layers, including a normalization layer, 5 convolutional layers & 3 fully connected layers.

1L image normalization hardcoded
2~4L 2x2 stride & 5x5 kernel CNN
5,6L non-strided CNN 3x3 Kernel
7~9L fully conn = output inverse turning radius 
# Not sure which parts of network is feature extractor & controller
"""
model = Sequential()
model.add(Lambda(lambda x: x/127.5-1, input_shape=(64,64,3)))  #1 image normalization hardcoded
nbFilter, nbRow, nbCol = 24,5,5                                #2
model.add(Convolution2D(nbFilter, nbRow, nbCol, border_mode='same', subsample=(2,2), activation=Relu))
model.add(MaxPooling2D(pool_size=(2,2), strides=(1,1)))
nbFilter = 36                                                  #3
model.add(Convolution2D(nbFilter, nbRow, nbCol, border_mode='same', subsample=(2,2), activation=Relu))
model.add(MaxPooling2D(pool_size=(2,2), strides=(1,1)))
nbFilter = 48                                                  #4
model.add(Convolution2D(nbFilter, nbRow, nbCol, border_mode='same', subsample=(2,2), activation=Relu))
model.add(MaxPooling2D(pool_size=(2,2), strides=(1,1)))
nbFilter, nbRow, nbCol = 64,3,3                                #5
model.add(Convolution2D(nbFilter, nbRow, nbCol, border_mode='same', subsample=(2,2), activation=Relu))
model.add(MaxPooling2D(pool_size=(2,2), strides=(1,1)))
nbFilter = 64                                                  #6
model.add(Convolution2D(nbFilter, nbRow, nbCol, border_mode='same', subsample=(2,2), activation=Relu))
model.add(MaxPooling2D(pool_size=(2,2), strides=(1,1)))
model.add( Flatten() )
model.add(Dropout(.5))
#model.add(Activation(Relu)) # Recommended by Udacity
for denShape in [1164, 100, 40, 10]:
    model.add(Dense(denShape, activation=Relu))  #7~
model.add( Dense(1) ) # Final
model.summary()
model.compile(optimizer=Adam(learnRate), loss='mse')

test_gen    = hU.generateNewBatch(targetDir, batchSize, hU.tFnm)
mTrain_gen  = hU.generateNewBatch(targetDir, batchSize, hU.mFnm)
validat_gen = hU.generateNewBatch(targetDir, batchSize, hU.vFnm)

print('\nStarting At: ', targetDir)

history = model.fit_generator(mTrain_gen,
                              samples_per_epoch=nb_samples, 
                              nb_epoch=nb_epochs,
                                  validation_data=validat_gen,
                                  nb_val_samples=nb_validtns,
                                  verbose=1)

#predict = model.validation_data(
hU.saveModel(targetDir, model)
evaluate = model.evaluate_generator(test_gen, 1008)
print("model.evaluate", evaluate)
