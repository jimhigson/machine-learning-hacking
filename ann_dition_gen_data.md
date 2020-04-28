---
jupyter:
  jupytext:
    formats: ipynb,md
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.2'
      jupytext_version: 1.4.2
  kernelspec:
    display_name: machine_learning
    language: python
    name: machine_learning
---

```python
# somewhat similar (but this written mostly before reading that)
#https://www.pluralsight.com/guides/deep-learning-model-add
```

```python
!#pip install livelossplot --quiet
```

```python
import tensorflow as tf
# tf.logging.set_verbosity(tf.logging.ERROR)
from keras.models import Sequential
from keras import optimizers
from keras.layers import Dense, Activation, Flatten
from keras.optimizers import Adam

import numpy as np
```

```python
trainingDataSize = 10000
epochs = 50
```

```python
X = np.random.normal( loc=0, scale=100, size=[trainingDataSize,2] )

```

```python
Y = X.sum(axis=1)
Ynoisy = Y + np.random.normal( scale = 3, size=trainingDataSize )
```

```python
Y
```

```python
Ynoisy
```

```python
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
```

```python
from IPython.display import SVG
from keras.utils import model_to_dot

SVG(model_to_dot(model).create(prog='dot', format='svg'))
```

```python

```

```python
model.fit(X, Ynoisy, epochs=epochs, batch_size=100)

```

```python
model.predict(np.array([[101,102]]))
```

```python
model.predict(np.array([[95,100]]))
```

```python
model.predict(np.array([[1,2]]))
```

```python
model.predict(np.array([[2,2]]))
```

```python
model.predict(np.array([[100,-250]]))
```

```python
model.predict(np.array([[10000,-2500]]))
```

```python

```

```python

```
