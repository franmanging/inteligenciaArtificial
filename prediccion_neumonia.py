# -*- coding: utf-8 -*-

from keras.preprocessing.image import ImageDataGenerator, load_img
from keras.models import Model
from keras.preprocessing import image
import os
import numpy as np
import matplotlib.pyplot as plt
# %matplotlib inline

# Colab library to upload files to notebook
from google.colab import files

# Install Kaggle library
!pip install -q kaggle

from google.colab import files
files.upload()

!ls

!mkdir -p ~/.kaggle
!cp kaggle.json ~/.kaggle/
!chmod 600 ~/.kaggle/kaggle.json

!kaggle datasets download paultimothymooney/chest-xray-pneumonia

import os
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
train_data_dir = 'chest_xray/train'
validation_data_dir = 'chest_xray/val'
test_data_dir = 'chest_xray/test'

#Cantidad de imagenes y epocas, tamano del batch
training_samples = 5216
validation_samples = 16
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
