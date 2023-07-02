from styleup.color_detection import palette_perc, open_picture, convert_to_array, make_clusters, get_color_names, \
all_color_names, list_complentary_colors, final_color_list, list_final_color_names

from styleup.style_detection import create_data_path, get_train_data_and_class_names, get_val_data, get_test_data_path, \
get_test_data, autotune_it, preprocessing_data, load_model, set_nontrainable_layers, build_model, compile_model, fit_model, \
new_pic_as_array, open_new_pic, style_result_dictionary, top_style_results, top_styles_string, translate_style_adjective
from styleup.registry import save_model, load_model_rn
from styleup.main_functions import string_output

from sklearn.cluster import KMeans
import os
import numpy as np

#tensorflow
import tensorflow as tf







################################    Upload a pictur - Here we take sample pic right now ################

pic_name = "elestic1.jpg"
# 1. Open Picture and convert it to Numpy Array
img = open_picture(pic_name)

# Convert image to array and set to int
img_array = convert_to_array(img)


###################################      PART 1 - GET COLOR NAMES    #############################################################


# Clustering colors - determine number of clusters
k_cluster = 10
clt_1 = make_clusters(k_cluster,img_array)

# Get dictionary of colors in picture and color palette
palette, dic_color_per = palette_perc(clt_1)


# Get dictionary with string names for the colors
color_dict = get_color_names()

# Attach name of colors in our dictionary with the colorcodes from user picture
pic_cname_perc_rgb = all_color_names(dic_color_per,color_dict)

# Get dictionary with complementary colors
comp_colors_cname_perc_rgb = list_complentary_colors(dic_color_per,color_dict)

# Get the final color list

#final_color_list(pic_cname_perc_rgb, comp_colors_cname_perc_rgb)
# number of colors from real color and from complementary colors
colors_real_pic = 2
colors_complementary = 1
final_list_colors_with_rgb = final_color_list(pic_cname_perc_rgb, comp_colors_cname_perc_rgb,colors_real_pic,colors_complementary)
final_color_names = list_final_color_names(final_list_colors_with_rgb)

#################################### END PART 1 - COLOR NAMES ################################################################





###############################   PART 2 - STYLE DETECTION ###############################################################

# 1. Input from user upload
# Picture as img
img
# Picture as array
img_array


# 2. Load data
# Data pathes of training (validation is training path) and test
train_data_path = create_data_path()
test_data_path = get_test_data_path()
categories = os.listdir(train_data_path)

# Parameters for Loading Data
batch_size = 32
img_height = 250
img_width = 250

# 2.1. Load train data
train_ds, class_names = get_train_data_and_class_names(train_data_path, batch_size, img_height, img_width)
# 2.2. Load validation data
val_ds = get_val_data(train_data_path, batch_size, img_height, img_width)
# 2.3. Load test data
test_ds = get_test_data(test_data_path, batch_size, img_height, img_width)
# 2.4. Autotune it to run faster
train_ds, val_ds, test_ds = autotune_it(train_ds, val_ds,test_ds)

# 3. Preprocess the loaded data - Only the X
train_ds_proc, val_ds_proc, test_ds_proc = preprocessing_data(train_ds, val_ds, test_ds)

# 4. Set up ResNet50 Model
training = False
if training:
    # Parameter for training
    learning_rate = 1e-4
    epochs = 1
    batch_size = 32
    patience = 5


    # 4.1. Load the model
    model = load_model()
    # 4.2. Set it to non trainable
    model = set_nontrainable_layers(model)
    # 4.3. Build the model with our Dens layers
    model_rn = build_model()
    # 4.4. Compile Model
    model_rn = compile_model(model, learning_rate)
    # 4. 5. Fit the Model - Training / !!!!Just run this if you want to retrain it!!!
    ################ Here problems it does not run yet ####################
    history_rn = fit_model(train_ds_proc, val_ds_proc, model_rn, epochs, batch_size, patience)
    ################ Here problems it does not run yet ####################
    # 4. 6. Evaluate the model
    model_evaluate = model_rn.evaluate(test_ds_proc)


    # 4.7. Save the model
    # 4.7.1. Model save path
    model_save_path = os.path.join("..","saved-models")
    # 4.7.2. Define metrics and params
    metrics = dict(mae=np.min(history_rn.history['val_accuracy']))
    params = dict(
    learning_rate=learning_rate,
    batch_size=batch_size,
    patience=patience)
    # 4.8. Save the model
    save_model(model_rn, params=params, metrics=metrics, model_save_path=model_save_path)

else:
    model_save_path = os.path.join("..","saved-models")
    model_rn= load_model_rn(model_save_path=model_save_path)
#print(model_rn.summary())

# 5. Predict Style of new pic
# 5.1. Open image in 350x250
img_pic = open_new_pic(pic_name)

# 5.2. Preprocess image - shape
X_new_proc = new_pic_as_array(img_pic)

# 5.3. Get the prediction
y_new = model_rn.predict(X_new_proc)

# 5.4. Make a sorted dictionary of the results
pred_dict_sorted = style_result_dictionary(y_new)

# 5.5. Just show the X top result design styles
number_of_styles = 2
top_designs = top_style_results(pred_dict_sorted, number_of_styles)

# 5.6 Get the result as lists
top_desings_string = top_styles_string(top_designs)

# 5.7. Convert the style types to descriptive adjectives (Scandinavian --> clena, simple, modern)
# Number of adjectives per style to use
num_adj_per_style = 2
final_adj_style_list = translate_style_adjective(top_desings_string, num_adj_per_style)


#########################   END 2. COLOR DETECTION   #####################################



##########################   PART 3 - PUT EVERYTHING TOGETHER ############################

output_string = string_output(final_color_names, final_adj_style_list)






if __name__ == '__main__':
    try:
        #print(final_color_names)
        #print(final_adj_style_list)
        print("final string-output:  ",output_string)
    except:
        pass
