# -*- coding: utf-8 -*-

from keras.preprocessing.image import ImageDataGenerator, load_img
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D,Activation, Dropout, Flatten, Dense
from keras import backend as K
from keras.preprocessing import image
from keras.models import Model
from keras.optimizers import Adam
import os
import numpy as np
import matplotlib.pyplot as plt
# %matplotlib inline

import zipfile
for file in os.listdir():
    zip_ref = zipfile.ZipFile('chest-xray-pneumonia.zip', 'r')
    zip_ref.extractall()
    zip_ref.close()

#Ruta del dataset
print(os.listdir("chest_xray"))

print(os.listdir("chest_xray/train"))

print(os.listdir("chest_xray/test"))

#
img_width, img_height = 150,150

#Crear directorios de entrenamiento, test y validacion 
directorio_entrenamiento = 'chest_xray/train'
directorio_validacion = 'chest_xray/val'
directorio_test = 'chest_xray/test'

#Cantidad de imagenes y epocas, tamano del batch
muestras_entrenamiento = 5216
muestras_validacion = 16
epochs = 10
batch_size = 16

if K.image_data_format() == 'channels_first':
    input_shape = (3, img_width, img_height)
else:
    input_shape = (img_width, img_height, 3)

model = Sequential()
model.add(Conv2D(32, (3, 3), input_shape=input_shape))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(32, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(1))
model.add(Activation('sigmoid'))

model.layers

model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

reescalamiento_entrenamiento = ImageDataGenerator(
    rescale=1. / 255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)

reescalamiento_test = ImageDataGenerator(rescale=1. / 255)


generador_entrenamiento = reescalamiento_entrenamiento.flow_from_directory(
    directorio_entrenamiento,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='binary')


generador_validaciones = reescalamiento_test.flow_from_directory(
    directorio_validacion,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='binary')

generador_tests = reescalamiento_test.flow_from_directory(
    directorio_test,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='binary')

resultados = model.fit(
    generador_entrenamiento,
    steps_per_epoch=muestras_entrenamiento//muestras_validacion,
    epochs=epochs,
    validation_data=generador_validaciones,
    validation_steps=muestras_validacion//batch_size)

promedio = model.evaluate(generador_validaciones,steps=1)
print("\n%s: %.2f%%" % (model.metrics_names[0], promedio[1]*100))
promedio = model.evaluate(generador_entrenamiento,steps=1)
print("\n%s: %.2f%%" % (model.metrics_names, promedio[1]*100))
promedio = model.evaluate(generador_tests,steps=1)
print("\n%s: %.2f%%" % (model.metrics_names, promedio[1]*100))



# Curvas de aprendizaje
epochs = range(1, epochs + 1, 1)
fig, ax = plt.subplots(1, 2)
train_acc = resultados.history['accuracy']
train_loss = resultados.history['loss']
val_acc = resultados.history['val_accuracy']
val_loss = resultados.history['val_loss']

fig.set_size_inches(10,5)

ax[0].plot(epochs , train_acc , 'go-' , label = 'Training Accuracy')
ax[0].plot(epochs , val_acc , 'ro-' , label = 'Validation Accuracy')
ax[0].set_title('Training & Validation Accuracy')
ax[0].legend()
ax[0].set_xlabel("Epochs")
ax[0].set_ylabel("Accuracy")

ax[1].plot(epochs , train_loss , 'g-o' , label = 'Training Loss')
ax[1].plot(epochs , val_loss , 'r-o' , label = 'Validation Loss')
ax[1].set_title('Training & Validation Loss')
ax[1].legend()
ax[1].set_xlabel("Epochs")
ax[1].set_ylabel("Loss")
plt.show()





