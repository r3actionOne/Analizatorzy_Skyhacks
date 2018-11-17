import keras
import numpy as np
from keras import Sequential
from keras.preprocessing import image
from keras_preprocessing.image import ImageDataGenerator
from keras.layers import Convolution2D, MaxPooling2D, Flatten, Dense

img_width, img_height = 150, 150

test_image = image.load_img('random2.jpg', target_size=(img_width,img_height))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis = 0)

classifier = keras.models.load_model('catsdogs.h5')
result = classifier.predict(test_image)

train_dataen = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)

training_set = train_dataen.flow_from_directory(
    'C:\images\Training',
    target_size=(img_width, img_height),
    batch_size=24,
    class_mode='binary'
)

if result [0][0] >= 0.5:
    prediction = 'dog'
else:
    prediction = 'cat'

print(prediction)