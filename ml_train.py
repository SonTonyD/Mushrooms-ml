# -*- coding: utf-8 -*-

#  Copyright (c) 2021. @Lucas

import splitfolders
from tensorflow.keras.layers import Flatten, Dense, Dropout
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
import matplotlib.pyplot as plt
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

def split_datas(ratio):
    """ This function allows to split input data in 3 differents dir : train/test/val according the ratio
        @ratio (0.6, 0.2, 0.2)
    """

    input_folder = "Mushrooms/input_dataset"
    output_folder = "Mushrooms/processed_Data"
    splitfolders.ratio(input_folder, output_folder, seed=42, ratio=ratio)

# split_datas((0.75, 0.15, 0.1))

IMG_HEIGHT, IMG_WIDTH = (224, 224)  # height width
BATCH_SIZE = 32  # nombre d'image processed en meme temps  (https://arxiv.org/pdf/1404.5997.pdf)
TRAIN_DIR = r"Mushrooms/processed_data/train"
VALIDATION_DIR = r"Mushrooms/processed_data/val"
TEST_DIR = r"Mushrooms/processed_data/test"
EPOCHS = 5
pretrain = "inception_v3"
preprocess_input = getattr(__import__("tensorflow.keras.applications."+pretrain, fromlist=["preprocess_input"]), "preprocess_input")
model_name = pretrain+"/e"+str(EPOCHS)+"_b"+str(BATCH_SIZE)+"_boolean_bolete"
print("Save model to : saved_model/{}.h5".format(model_name))

train_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,  # preprocess input from RESNET50, convert RGB to BGR
    shear_range=0.2,  # image will be distorted along an axis
    zoom_range=0.2,  # zoom de l'image
    horizontal_flip=True,  # allow horizontal flip
    vertical_flip=True,  # allow vertical flip
    validation_split=0.2  # Float. Fraction of images reserved for validation (strictly between 0 and 1).
)

train_generator = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    subset="training"
)

valid_generator = train_datagen.flow_from_directory(
    VALIDATION_DIR,
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    subset="validation"
)
test_generator = train_datagen.flow_from_directory(
    TEST_DIR,
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=1,
    class_mode="categorical",
    subset="validation"
)

data_augmentation = Sequential(
  [
    layers.experimental.preprocessing.RandomFlip("horizontal_and_vertical"),  # flip vertical ou horizontal
    layers.experimental.preprocessing.RandomRotation(0.2),  # range : [-20% * 2pi, 20% * 2pi].
    layers.experimental.preprocessing.RandomZoom(0.1),  # random zoom

  ]
)
if pretrain == "resnet50":
    base_model = ResNet50(include_top=False, weights='imagenet')
elif pretrain == "vgg16":
    base_model = VGG16(include_top=False, weights='imagenet')
else:
    base_model = InceptionV3(include_top=False, weights='imagenet')

for layer in base_model.layers:
    layer.trainable = False  # layer from ResNet network won't be trained

model = Sequential()
model.add(data_augmentation)
model.add(base_model)  # We don't use fully connected layer
model.add(Flatten())  # Need to flatten the cnn
model.add(Dropout(0.3))
model.add(Dense(2048, activation="relu"))
model.add(Dropout(0.3))
model.add(Dense(2048, activation="relu"))
model.add(Dropout(0.3))
model.add(Dense(1024, activation="relu"))
model.add(Dropout(0.3))
model.add(Dense(512, activation="relu"))
model.add(Dropout(0.2))
model.add(Dense(512, activation="relu"))
model.add(Dropout(0.1))
model.add(Dense(256, activation="relu"))
model.add(Dense(train_generator.num_classes, activation='softmax'))
    
model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=['accuracy'])
history = model.fit(train_generator, validation_data=valid_generator, epochs=EPOCHS)
model.save("saved_model/"+model_name+".h5")
model.summary()
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(EPOCHS)

plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()
