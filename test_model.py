# -*- coding: utf-8 -*-

import warnings
warnings.filterwarnings('ignore')


# Dependencias

from keras.preprocessing.image import ImageDataGenerator, load_img
import matplotlib.pyplot as plt
import tensorflow as tf

# Directorios de test
test_data_dir = 'chest_xray/test'

img_width, img_height = 150,150
batch_size = 16

# Cargar el modelo entrenado previamente 
model = tf.keras.models.load_model('model_1')

# Generador para set de Test
test_datagen = ImageDataGenerator(rescale=1. / 255)

test_generator = test_datagen.flow_from_directory(
    test_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='binary')


print("Evaluar con set de Test:")
scores = model.evaluate(test_generator,steps=test_generator.samples//batch_size)
print("\nTest: %s: %.2f%%, %.2f%%" % (model.metrics_names, scores[0]*100, scores[1]*100))

