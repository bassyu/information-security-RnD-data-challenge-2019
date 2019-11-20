from keras_efficientnets import EfficientNetB7 # use B7
from keras.preprocessing.image import ImageDataGenerator
from keras.models import load_model
import tensorflow as tf
import PIL.Image

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)

PIL.Image.MAX_IMAGE_PIXELS = None


input_size = (256, 256)
input_shape = (256, 256, 3)
train_data_path = '../data/train'
val_data_path = '../data/validation'


# data
train_datagen = ImageDataGenerator(rescale=1./255)
train_generator = train_datagen.flow_from_directory(
        train_data_path,
        target_size=input_size,
        batch_size=4,
        class_mode='categorical')

val_datagen = ImageDataGenerator(rescale=1./255)
val_generator = val_datagen.flow_from_directory(
        val_data_path,
        target_size=input_size,    
        batch_size=4,
        class_mode='categorical')

# model
model = EfficientNetB7(input_shape, classes=2, include_top=True)
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# train
model.fit_generator(
        train_generator,
        steps_per_epoch=5000,
        epochs=32,
        validation_data=val_generator,
        validation_steps=20)

# evaluation
print('-- Evaluate --')
scores = model.evaluate_generator(val_generator, steps=5)
print("%s: %.2f%%" %(model.metrics_names[1], scores[1]*100))

# model save
model.save('model.h5')
