import tensorflow as tf
from tensorflow.keras.optimizers import RMSprop

from Model import automap, complex_automap
import Data.data_loader as dl
import config 

model = complex_automap.model

train_data_loader = dl.train_dataset
test_data_loader = dl.test_dataset

epochs = config.epochs
rms = RMSprop(learning_rate=0.00002,
              rho=0.9,
              momentum=0.0,
              epsilon=1e-07,
              name="RMSprop")

model.compile(loss='mean_squared_error', optimizer=rms, metrics=['mean_squared_error'])

logs = "logs/"
tboard_callback = tf.keras.callbacks.TensorBoard(log_dir = logs,
                                                 histogram_freq = 1,
                                                 profile_batch = '500,520')

model.load_weights(config.saved_weights)    

print("fitting...")
history = model.fit_generator(generator=train_data_loader, validation_data=test_data_loader, callbacks = tboard_callback, epochs=epochs, verbose=1)

model.save_weights(config.final_weights) 