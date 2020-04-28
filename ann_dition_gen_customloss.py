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

# somewhat similar (but this written mostly before reading that)
# https://www.pluralsight.com/guides/deep-learning-model-add

# for custom loss functions:
# https://heartbeat.fritz.ai/how-to-create-a-custom-loss-function-in-keras-637bd312e9ab

!#pip install livelossplot --quiet

# +
import tensorflow as tf
# tf.logging.set_verbosity(tf.logging.ERROR)
from keras.models import Sequential
from keras import optimizers
from keras.layers import Dense, Activation, Flatten
from keras.optimizers import Adam

import keras.backend as kb

import numpy as np
# -

trainingDataSize = 10000
epochs = 51

X = np.random.normal( loc=0, scale=100, size=[trainingDataSize,2] )


Y = X.sum(axis=1)
Ynoisy = Y + np.random.normal( scale = 3, size=trainingDataSize )

Y

Ynoisy

#https://keras.io/losses/
def lossFuncLikeMSE(yActual, yPred):
    customLossValue = kb.mean(kb.sum(kb.square((yActual - yPred)/10)))
    return customLossValue;

# the loss function can handle singles:
lossFuncLikeMSE(2,2)

# +
# the loss function can handle multiples:

ktrue = kb.variable(value=np.array([1,2,3,4]), dtype='float64', name='example_var')
kpredGood = kb.variable(value=np.array([1,2.1,2.9,4]), dtype='float64', name='example_var')
kpredBad = kb.variable(value=np.array([1,200,3,4]), dtype='float64', name='example_var')
# -

lossFuncLikeMSE(ktrue,kpredGood)

lossFuncLikeMSE(ktrue,kpredBad)

model = Sequential([
    #Flatten(), # input_shape=(2,)
    Dense(units=2, activation='relu'),
    Dense(units=20, activation='relu'),
	Dense(units=20, activation='relu'),
    Dense(units=1)
])
model.compile(optimizer=Adam(learning_rate=0.001),
              loss=lossFuncLikeMSE,
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
