from keras.models import Sequential
from keras.layers import Convolution2D, MaxPooling2D, Flatten, Dense, Conv2D, Activation, Dropout, K

# img_width, img_height = 1280 / 8, 1024 / 8
img_width, img_height = 320, 256

train_data_dir = 'C:\images\Training'
validation_data_dir = 'C:\images\Validation'

nb_train_samples = 2500
nb_validation_samples = 400
epochs = 2
batch_size = 24

# classifier = Sequential()
# classifier.add(Convolution2D(32,3,3, input_shape = (img_width, img_height, 3), activation='relu'))
# classifier.add(MaxPooling2D(pool_size=(2,2)))
# classifier.add(Flatten())
#
# classifier.add(Dense(output_dim = 128, activation='relu'))
# classifier.add(Dense(output_dim = 1, activation='sigmoid'))

if K.image_data_format() == 'channels_first':
    input_shape = (3, img_width, img_height)
else:
    input_shape = (img_width, img_height, 3)

classifier = Sequential()
classifier.add(Conv2D(32, (3, 3), input_shape=input_shape))
classifier.add(Activation('relu'))
classifier.add(MaxPooling2D(pool_size=(2, 2)))

classifier.add(Conv2D(32, (3, 3)))
classifier.add(Activation('relu'))
classifier.add(MaxPooling2D(pool_size=(2, 2)))

classifier.add(Conv2D(64, (3, 3)))
classifier.add(Activation('relu'))
classifier.add(MaxPooling2D(pool_size=(2, 2)))

classifier.add(Flatten())
classifier.add(Dense(64))
classifier.add(Activation('relu'))
classifier.add(Dropout(0.5))
classifier.add(Dense(1))
classifier.add(Activation('sigmoid'))

classifier.compile(optimizer='adam', loss = 'binary_crossentropy', metrics=['accuracy'])

from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

training_set = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_width,img_height),
    batch_size=batch_size,
    class_mode='binary'
)

validation_set = test_datagen.flow_from_directory(
    validation_data_dir,
    target_size=(img_width,img_height),
    batch_size=batch_size,
    class_mode='binary'
)

from IPython.display import display
from PIL import Image

classifier.fit_generator(
    training_set,
    steps_per_epoch=nb_train_samples // batch_size,
    epochs=epochs,
    validation_data=validation_set,
    validation_steps=nb_validation_samples // batch_size
)

classifier.save('asd.h5')