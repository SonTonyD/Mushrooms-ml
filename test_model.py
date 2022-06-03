# Copyright (c) 2021. @Lucas # -*- coding: utf-8 -*-
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sn
import tensorflow as tf
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True


IMG_HEIGHT, IMG_WIDTH = (224,224)
BATCH_SIZE = 32
TRAIN_DIR = r"Mushrooms/processed_data/train"
VALIDATION_DIR = r"Mushrooms/processed_data/val"
TEST_DIR = r"Mushrooms/processed_data/test"
pretrain = "inception_v3"
preprocess_input = getattr(__import__("tensorflow.keras.applications."+pretrain, fromlist=["preprocess_input"]), "preprocess_input")

train_datagen = ImageDataGenerator(preprocessing_function=preprocess_input,
                                   shear_range=0.2,
                                   zoom_range=0.2,
                                   horizontal_flip=True,
                                   validation_split=0.4)
test_generator = train_datagen.flow_from_directory(
    TEST_DIR,
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=1,
    shuffle=True,
    class_mode="categorical",
    subset="validation"
)
class_names = list(test_generator.class_indices.keys())
#model = tf.keras.models.load_model("saved_model/resnet50_e10_b32_boolean_bolete.h5")
model = tf.keras.models.load_model("saved_model/inception_v3/e5_b32_boolean_bolete.h5")

def get_loss_acc():
    test_loss, test_acc = model.evaluate(test_generator, verbose=2)
    print("Accuracy : {}".format(test_acc))
    print("Loss : {}".format(test_loss))

def plot_image(predictions_array, true_label, img):
  plt.grid(False)
  plt.xticks([])
  plt.yticks([])

  plt.imshow(img.astype("uint8"))

  predicted_label = np.argmax(predictions_array)
  if predicted_label == true_label:
    color = 'blue'
  else:
    color = 'red'

  plt.xlabel("{} {:2.0f}% ({})".format(class_names[predicted_label],
                                100*np.max(predictions_array),
                                class_names[true_label]),
                                color=color)

def plot_value_array(predictions_array, true_label):
  plt.grid(False)
  plt.xticks(range(len(predictions_array)))
  plt.yticks([])
  thisplot = plt.bar(range(len(predictions_array)), predictions_array, color="#777777")
  plt.ylim([0, 1])
  predicted_label = predictions_array.argmax()

  thisplot[predicted_label].set_color('red')
  thisplot[true_label].set_color('blue')

def show_matrix():

    filenames = test_generator.filenames
    nb_samples = len(test_generator)
    y_prob=[]
    y_act=[]
    test_generator.reset()
    for _ in range(nb_samples):
        X_test, Y_test = test_generator.next()
        y_prob.append(model.predict(X_test))
        y_act.append(Y_test)

    predicted_class = [list(test_generator.class_indices.keys())[i.argmax()] for i in y_prob]
    actual_class = [list(test_generator.class_indices.keys())[i.argmax()] for i in y_act]

    out_df = pd.DataFrame(np.vstack([predicted_class, actual_class]).T, columns=["predicted_class", "actual_class"])
    confusion_matrix = pd.crosstab(out_df["actual_class"], out_df["predicted_class"], rownames=["Actual"], colnames=["Predicted"], normalize="columns")

    sn.heatmap(confusion_matrix, cmap="Blues", annot=True, fmt=".2f")
    plt.show()
    print('test accuracy : {}'.format((np.diagonal(confusion_matrix).sum()/confusion_matrix.sum().sum()*100)))

def show_first_picture(col,rows):
    num_rows = rows
    num_cols = col
    num_images = num_rows * num_cols
    plt.figure(figsize=(2 * 2 * num_cols, 2 * num_rows))
    for i in range(num_images):
        preimg, lbl = test_generator.next()
        img = 127 + preimg[0]
        label = lbl.argmax()
        prediction = model.predict(preimg)[0]
        plt.subplot(num_rows, 2 * num_cols, 2 * i + 1)
        plot_image(prediction, label, img)
        plt.subplot(num_rows, 2 * num_cols, 2 * i + 2)
        plot_value_array(prediction, label)
    plt.tight_layout()
    plt.show()

show_matrix()
show_first_picture(5,8)