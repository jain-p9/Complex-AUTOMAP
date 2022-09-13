#COMPLEX AUTOMAP MODEL

from tensorflow.keras import layers, models
from Model.complex_layers import CFlatten, CDense, CReshape, CConv2D, CConv2DTranspose, abs

inputshape = (64,64,2)
fc2_shape = inputshape[0]*inputshape[1]*inputshape[2]

model = models.Sequential()
model.add(layers.Input(inputshape))
model.add(CFlatten())
model.add(CDense(fc2_shape))
model.add(layers.Activation('tanh'))
model.add(CReshape(inputshape[0], inputshape[1], inputshape[2]))
model.add(CConv2D(64, [5,5], padding='same'))
model.add(layers.Activation('relu'))
model.add(CConv2D(64, [5,5], padding='same'))
model.add(layers.Activation('relu'))
model.add(CConv2DTranspose(2, [7,7], padding='same'))
model.add(abs())

model.summary()
