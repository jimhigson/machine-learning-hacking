# ---
# jupyter:
#   jupytext:
#     formats: py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.4.2
#   kernelspec:
#     display_name: machine_learning
#     language: python
#     name: machine_learning
# ---

# +
# somewhat similar (but this written mostly before reading that)
#https://www.pluralsight.com/guides/deep-learning-model-add
# -

!#pip install livelossplot --quiet

# +
import tensorflow as tf
# tf.logging.set_verbosity(tf.logging.ERROR)
from keras.models import Sequential
from keras import optimizers
from keras.layers import Dense, Activation, Flatten
from keras.optimizers import Adam

import numpy as np
# -

trainingDataSize = 10000
epochs = 51

X = np.random.normal( loc=0, scale=100, size=[trainingDataSize,2] )


Y = X.sum(axis=1)
Ynoisy = Y + np.random.normal( scale = 3, size=trainingDataSize )

Y

Ynoisy

model = Sequential([
    #Flatten(), # input_shape=(2,)
    Dense(units=2, activation='relu'),
    Dense(units=20, activation='relu'),
	Dense(units=20, activation='relu'),
    Dense(units=1)
])
model.compile(optimizer=Adam(learning_rate=0.001),
              loss='mse',
              metrics=['mae'])

# +
from IPython.display import SVG
from keras.utils import model_to_dot

SVG(model_to_dot(model).create(prog='dot', format='svg'))
# -

model.fit(X, Ynoisy, epochs=epochs, batch_size=100)


model.predict(np.array([[101,102]]))

model.predict(np.array([[95,100]]))

model.predict(np.array([[1,2]]))

model.predict(np.array([[2,2]]))

model.predict(np.array([[100,-250]]))

model.predict(np.array([[10000,-2500]]))




