from styleup.registry import save_model

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.style as style

# for image loading
from tensorflow.keras.utils import load_img, save_img
from tensorflow.keras.preprocessing.image import img_to_array
# for CNN model
import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras import layers, Sequential
from tensorflow.keras import optimizers
from tensorflow.keras.callbacks import EarlyStopping

import os

from tensorflow.keras import Model, models


def create_data_path():
    # define the path of the picture categories
    train_data_path = os.path.join("..","raw_data","living-room-categories","train_data")
    return train_data_path

# get train data
train_data_path = create_data_path()

# list of categories
categories = os.listdir(train_data_path)


# 2. Load data
# Parameters for Loading Data
batch_size = 32
img_height = 250
img_width = 250

# 2.1. Load train data
def get_train_data_and_class_names(train_data_path, batch_size, img_height, img_width):
    train_ds = tf.keras.utils.image_dataset_from_directory(train_data_path, validation_split=0.2, subset="training",
    seed=123, image_size=(img_height, img_width), batch_size=batch_size)
    # Get Class names of train data
    class_names = train_ds.class_names
    # One hot encode train y
    train_ds = train_ds.map(lambda x, y: (x, tf.one_hot(y,9)))

    return train_ds, class_names

# Run it
train_ds, class_names = get_train_data_and_class_names(train_data_path, batch_size, img_height, img_width)


#2.2. Load validation data

def get_val_data(train_data_path, batch_size, img_height, img_width):
    # Validation Data
    val_ds = tf.keras.utils.image_dataset_from_directory(train_data_path, validation_split=0.2, subset="validation",
    seed=123, image_size=(img_height, img_width), batch_size=batch_size)
    # One hot encode the y ov validation data
    val_ds = val_ds.map(lambda x, y: (x, tf.one_hot(y,9)))

    return val_ds

# Run it
val_ds = get_val_data(train_data_path, batch_size, img_height, img_width)

# 2.3. Load test data

def get_test_data_path():
    return os.path.join("..","raw_data","living-room-categories","test_data")

# run it
test_data_path = get_test_data_path()

def get_test_data(test_data_path, batch_size, img_height, img_width):
    test_ds = tf.keras.utils.image_dataset_from_directory(test_data_path, image_size=(img_height, img_width), batch_size=batch_size)
    # One hot encode the y
    test_ds = test_ds.map(lambda x, y: (x, tf.one_hot(y,9)))
    return test_ds

# run it
test_ds = get_test_data(test_data_path, batch_size, img_height, img_width)

# 3. Autotune to load data faster

def autotune_it(train_ds, val_ds,test_ds):
    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)
    test_ds = test_ds.cache().prefetch(buffer_size=AUTOTUNE)

    return train_ds, val_ds, test_ds

#run it
train_ds, val_ds, test_ds = autotune_it(train_ds, val_ds,test_ds)


# 4. ResNet50

#4.1 Preprocessing
# Using the Resnet preprocessing function
def preprocessing_data(train_ds, val_ds, test_ds):
    train_ds_proc = train_ds.map(lambda x, y: (tf.keras.applications.resnet50.preprocess_input(x), y))
    val_ds_proc = val_ds.map(lambda x, y: (tf.keras.applications.resnet50.preprocess_input(x), y))
    test_ds_proc = test_ds.map(lambda x, y: (tf.keras.applications.resnet50.preprocess_input(x), y))
    return train_ds_proc, val_ds_proc, test_ds_proc

#run it
train_ds_proc, val_ds_proc, test_ds_proc = preprocessing_data(train_ds, val_ds, test_ds)

# 4.1. Load the ResNet50 model
def load_model():
    model = tf.keras.applications.resnet50.ResNet50(
    include_top=False,
    weights='imagenet',
    input_tensor=None,
    input_shape=None,
    pooling=None,
    classes=9)
    return model

# 4.2. Set to nontrainable
def set_nontrainable_layers(model):
    model.trainable = False
    return model


# 4.3. Build the complete model
def build_model():

    # Define the layers
    base_model = load_model()
    base_model = set_nontrainable_layers(base_model)
    flattening_layer = layers.GlobalAveragePooling2D()
    dense_layer = layers.Dense(100, activation='relu')
    prediction_layer = layers.Dense(9, activation='softmax')

    # Here is the pipeline
    model_rn = Sequential([
      base_model,
      flattening_layer,
      dense_layer,
      prediction_layer
    ])
    return model_rn

# Run it to build model
#model_rn = build_model()

# 4.4. Compile the model
learning_rate = 1e-4
def compile_model(model, learning_rate):
    opt = optimizers.Adam(learning_rate=learning_rate)
    model.compile(loss='categorical_crossentropy',
                  optimizer=opt,
                  metrics=['accuracy'])
    return model

# 4.5 Fit the model
epochs = 100
batch_size = 32
patience = 5

def fit_model(train_ds_proc, val_ds_proc, model_rn, epochs, batch_size, patience):

    print("Shape of train_proc")
    for x, y in train_ds_proc:
        print(x.shape)
        print(y.shape)
        break

    print("Shape of val_proc")
    for x, y in val_ds_proc:
        print(x.shape)
        print(y.shape)
        break


    es = EarlyStopping(monitor = 'val_accuracy',
                       mode = 'max',
                       patience = patience,
                       verbose = 1,
                       restore_best_weights = True)

    history_rn = model_rn.fit(train_ds_proc,
                        validation_data=val_ds_proc,
                        epochs=epochs,
                        batch_size=batch_size,
                        callbacks=[es],
                        verbose=1)
    return history_rn

# Run it
# Initiate mode_rn
#model_rn = build_model()
#model_rn = compile_model(model_rn)
# Fit the model
#history_rn = fit_model(model_rn)

# 4.6. Evaluate Model
#model_rn.evaluate(test_ds_proc)


# 5. Save the model

# 5.1. Path to save all
model_save_path = os.path.join("..","saved-models")

# 5. 2. Define metrics and params
# Compute the validation metric (min val mae of the holdout set)
# No history_rn in the moment - change when build model
#metrics = dict(mae=np.min(history_rn.history['val_accuracy']))

# Save Parameter
params = dict(learning_rate=1e-4, batch_size=batch_size, patience=5)

# Run the save function from registry.py
#save_model(model_rn, params=params, metrics=metrics, model_save_path=model_save_path)


# 6. Load new picture and process it


pic_name = "scan2.jpg"
def open_new_pic(pic_name):
    img_path = os.path.join("..","raw_data", "test-pics",pic_name )
    img_pic = load_img(path=img_path, color_mode="rgb",target_size=(250,250))
    return img_pic

# Run it
#new_img = open_new_pic(pic_name)

def new_pic_as_array(new_img):
    X_new = img_to_array(new_img).astype("int").reshape(1,250,250,3)
    X_new_proc = tf.keras.applications.resnet50.preprocess_input(X_new)
    return X_new_proc

# 7. Predict the categories

# Get a dictionary with style names
def style_result_dictionary(y_new):
    categories_dict= {0:"asian", 1:"elestic", 2: "farmhouse", 3: "industrial", 4: "mediterranean",
                    5:"midcentury", 6:"modern", 7:"scandinavian", 8: "traditional" }
    pred_dict = {}
    counter = 0
    for value in y_new[0]:
        cat_name = categories_dict[counter]
        pred_dict[cat_name] = value
        counter +=1
    pred_dict_sorted = dict(sorted(pred_dict.items(), key=lambda x: x[1],reverse=True))
    return pred_dict_sorted

# Run it
#pred_dict_sorted = style_result_dictionary(y_new)

number_of_styles = 2

def top_style_results(pred_dict_sorted, number_of_styles):
    top_x = number_of_styles
    i = 0
    top_pred_dict = {}
    for key, value in pred_dict_sorted.items():
        if i < top_x:
            top_pred_dict[key] = value
            i +=1
        else:
            break

    return top_pred_dict

# run it
#result_top_designs = top_style_results(pred_dict_sorted)


def top_styles_string(result_top_designs):
    return [key for key in result_top_designs.keys()]


# Run it
#final_top_styles = top_style_results(pred_dict_sorted)

num_per_style = 2
def translate_style_adjective(top_desings_string, num_adj_per_style):

    cat_style_adj = {"scandinavian": ["light", "functional", "organic"],
                     "industrial" : ["dark", "casual", "industrial"],
                     "farmhouse" : ["rustic", "oldfashioned", "charming"],
                     "midcentury": ["dull", "retro", "metallic"],
                     "modern" : ["clean", "rectilinear", "modern"],
                     "elestic": ["bright", "boho", "vintage"],
                     "asian" : ["ink", "asian", "shiny"],
                     "traditional" : ["neutral", "classic", "solid"],
                     "mediterranean" : ["colorful", "mediterranean", "chic"]}

    res_list = []
    for design in top_desings_string:
        i = 0
        for adj in cat_style_adj[design]:
            if i < num_adj_per_style:
                res_list.append(adj)
                i +=1
    return res_list










if __name__ == '__main__':
    try:
        pass
    except:
        pass
