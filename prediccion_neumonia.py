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

#Ruta del dataset
print(os.listdir("chest_xray"))

print(os.listdir("chest_xray/train"))

print(os.listdir("chest_xray/test"))

# Scaling para imagenes
img_width, img_height = 150,150

#Crear directorios de entrenamiento, test y validacion 
directorio_entrenamiento = 'chest_xray/train'
#directorio_validacion = 'chest_xray/val'
directorio_test = 'chest_xray/test'

#Cantidad de imagenes de entrenamiento, epocas, tamano del batch
muestras_entrenamiento = 5216
muestras_validacion = 16
epochs = 30
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

# Augmentation de Imagenes
reescalamiento_entrenamiento = ImageDataGenerator(
    rescale=1. / 255,
    shear_range=0.2,
    zoom_range=0.2,
    validation_split=0.2,
    horizontal_flip=True)

reescalamiento_test = ImageDataGenerator(rescale=1. / 255)

# Configurar generadores desde directorios
generador_entrenamiento = reescalamiento_entrenamiento.flow_from_directory(
    directorio_entrenamiento,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='binary',
    subset='training')


generador_validaciones = reescalamiento_test.flow_from_directory(
    directorio_entrenamiento,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='binary',
    subset='validation')

generador_tests = reescalamiento_test.flow_from_directory(
    directorio_test,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='binary')

resultados = model.fit(
    generador_entrenamiento,
    steps_per_epoch=generador_entrenamiento.samples//batch_size,
    epochs=epochs,
    validation_data=generador_validaciones,
    validation_steps=generador_validaciones.samples//batch_size)


print("Evaluar con set de Test:")
scores = model.evaluate(generador_tests,steps=generador_tests.samples//batch_size)
print("\nTest: %s: %.2f%%, %.2f%%" % (model.metrics_names, scores[0]*100, scores[1]*100))


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





