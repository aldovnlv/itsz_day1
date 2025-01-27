# adsoft 
import numpy as np
import os
#import matplotlib.pyplot as plt


# TensorFlow
import tensorflow as tf
 
print(tf.__version__)

X = np.arange(-10.0, 10.0, 1e-2)
print(X)
np.random.shuffle(X)
y =  2.0 * X + 1.0
print(y)

train_end = int(0.6 * len(X))
#print (train_end)
test_start = int(0.8 * len(X))
#print (test_start)

## datos de entrenamiento 
X_train, y_train = X[:train_end], y[:train_end]

## datos de prueba
X_test, y_test = X[test_start:], y[test_start:]

X_val, y_val = X[train_end:test_start], y[train_end:test_start]


## creacion del modeo
tf.keras.backend.clear_session()                       #                      dimension del tensor
linear_model = tf.keras.models.Sequential([                 #                    ^ 
                                           tf.keras.layers.Dense(units=1, input_shape=[1], name='Single')
                                           ])

## compilando el modelo
# python 3.8
##                                         Gradente decendiente               Media del error al cuadrado
##                                                   ^                                 ^
linear_model.compile(optimizer=tf.keras.optimizers.SGD(), loss=tf.keras.losses.mean_squared_error)
# python 3.12
# linear_model.compile(optimizer=tf.keras.optimizers.SGD(), loss=tf.keras.losses.MeanSquaredError)

## se imprime la arquitectura del modelo
print(linear_model.summary())


## Se entrena el modelo
##                                                                   ciclos
##                                                                       ^

## impresion de los pesos y bias del modelo
w,b = linear_model.weights
print('===== before')
print(w)
print(w.numpy())
print(b)
print(b.numpy())

linear_model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=20)

print('===== after')
print(w)
print(w.numpy())
print(b)
print(b.numpy())




#print(linear_model.predict([ [0.0], [2.0], [3.1], [4.2], [5.2] ] ).tolist() )   


## prueba del modelo con algunos datos de prueba
##                                                       
##                                                       ^
print(linear_model.predict(tf.constant([ [100.0], [1.0], [200.0], [300.0], [400.0] ] ) )) 

export_path = 'linear-model/1/'
tf.saved_model.save(linear_model, os.path.join('./',export_path))
