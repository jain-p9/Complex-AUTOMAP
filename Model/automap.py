#AUTOMAP MODEL

from tensorflow.keras import layers, models

inputshape = (64,64,2)
fc2_shape = inputshape[0]*inputshape[1]

model = models.Sequential()
model.add(layers.Flatten(input_shape=(inputshape[0], inputshape[1], inputshape[2])))
model.add(layers.Dense(fc2_shape, activation='tanh'))
model.add(layers.Dense(fc2_shape, activation='tanh'))
model.add(layers.Reshape((inputshape[0], inputshape[1], 1)))
model.add(layers.Conv2D(64, (5, 5), activation='relu', padding='same'))
model.add(layers.Conv2D(64, (5, 5), activation='relu', padding='same'))
model.add(layers.Conv2DTranspose(1, kernel_size=7, strides=1, padding='same'))

model.summary()
