import os
os.environ["KERAS_BACKEND"] = "tensorflow"
from keras import models
from keras import layers
from keras import optimizers
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dropout, Flatten, Dense
from keras.layers import Dropout, Flatten, Dense
from keras.applications import VGG16
from keras import backend as K
import glob
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
import numpy as np
from os.path import basename
from keras.callbacks import ModelCheckpoint
class KaggleImageSolver:



    def __init__(self, train_dir, test_dir,  num_categories, preprocess=None, postprocess=None, percentage=1, id_column='id', result_column='label'):
        self.train_dir = train_dir
        self.test_dir  = test_dir
        self.id_column = id_column
        self.result_column = result_column
        self.preprocess = preprocess
        self.num_categories = num_categories
        self.postprocess = postprocess
        self.percentage = percentage


    def train_model(self, img_height, img_width, model_dir, steps_per_epoch=100, epochs=30, batch_size=20):
        # dimensions of our images.

        if K.image_data_format() == 'channels_first':
            input_shape = (3, img_height, img_width)
        else:
            input_shape = (img_height, img_width, 3)

        conv_base = VGG16(weights='imagenet',
                          include_top=False,
                          input_shape=input_shape)
        model = models.Sequential()
        model.add(conv_base)
        model.add(layers.Flatten())
        model.add(layers.Dense(256, activation='relu'))
        model.add(layers.Dense(self.num_categories, activation='sigmoid'))

        conv_base.trainable = False

        train_datagen = ImageDataGenerator(
            rescale=1. / 255,
            rotation_range=40,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            fill_mode='nearest')

        train_generator = train_datagen.flow_from_directory(
            self.train_dir,
            target_size=(img_width, img_height),
            batch_size=batch_size,
            classes=['cat','dog'],
            class_mode='categorical')

        model.compile(loss='categorical_crossentropy',
                      optimizer=optimizers.RMSprop(lr=2e-5),
                      metrics=['acc'])

        filepath = model_dir+'weights-improvement-{epoch:02d}-{val_acc:.2f}.hdf5'
        checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
        callbacks_list = [checkpoint]

        history = model.fit_generator(
            train_generator,
            steps_per_epoch=steps_per_epoch,
            epochs=epochs)
        self.model=model
        return model

    def print_result(self, model, img_height, img_width, probability=True):

        test_datagen = ImageDataGenerator(rescale=1. / 255)

        all_classes = {}
        for img_path in glob.glob(self.test_dir+'/*.jpg'):
            img = load_img(img_path, target_size=(img_height, img_width))
            img_name = basename(img_path).split('.')[0]
            img_array = img_to_array(img)
            x = np.expand_dims(img_array , axis=0)
            if probability:
                cat_or_dog = model.predict_proba(x)[0]
                cat_or_dog = (cat_or_dog / sum(cat_or_dog))[0]
                print("{} : {%.2f}".format(img_name, cat_or_dog))
            else:
                cat_or_dog = model.predict_classes(x)[0]

            all_classes[int(img_name)] = cat_or_dog
        print(all_classes)