import tensorflow as tf
import numpy as np
from tensorflow import keras
#layyer has shape of 1 neuron 1 input
model=tf.keras.Sequential([keras.layers.Dense(units=1,input_shape=[1])])
#stochatic gradient descent loss= mean square error (vector distance)
model.compile(optimizer='sgd',loss='mean_squared_error')
X=np.array([-1.0, 0.0, 1.0, 2.0, 3.0, 4.0],dtype=float)
Y=np.array([-2.0, 1.0, 4.0, 7.0, 10.0, 13.0],dtype=float)
#train 500 times
model.fit(X, Y, epochs=500)
print(model.predict([20]))