from styleup.color_detection import palette_perc, open_picture, convert_to_array, make_clusters, get_color_names, \
all_color_names, list_complentary_colors, final_color_list, list_final_color_names

from styleup.style_detection import create_data_path, get_train_data_and_class_names, get_val_data, get_test_data_path, \
get_test_data, autotune_it, preprocessing_data, load_model, set_nontrainable_layers, build_model, compile_model, fit_model, \
new_pic_as_array, open_new_pic, style_result_dictionary, top_style_results, top_styles_string, translate_style_adjective
from styleup.registry import save_model, load_model_rn
from styleup.main_functions import string_output
from styleup.api import format_picture, get_style_id_list_for_topic, create_multiple_pictures

from sklearn.cluster import KMeans
import os
import numpy as np

#tensorflow
import tensorflow as tf
import streamlit as st
from tensorflow.keras.utils import load_img
import cv2
from decouple import config

############################### START SETTINGS   ################################


# Activate API
generate_with_api = False

# Train model
train_model = False

# Demo - Show model
demo = True
sleep_time = 40

################################# STREAMLIT HEAD     ###########################################
st.set_page_config(
            page_title="StyleUP",
            page_icon="🖼",
            layout="wide",
            initial_sidebar_state="auto") # collapsed

with open('style.css') as f:
    st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

with st.container():
    st.image("https://i.ibb.co/d4Ktn96/background-styleup.jpg", use_column_width="always")
    st.title("Unique artwork, individually created for your room")

################################    Upload a pictur - Here we take sample pic right now ################

# For streamlit flow
img_array = None
pic_theme = None
pic_format = None

# User upload the file
with st.container():
    st.write("---")

    st.set_option('deprecation.showfileUploaderEncoding', False)

    upload_img = st.file_uploader("Upload a picture of your room")
    st.write("##")

    if upload_img is not None:

        # Show the uploaded image
        #st.image(upload_img)

        # Convert image to array and set to int
        tmp_array = np.fromstring(upload_img.getvalue(), np.uint8)
        img_array = cv2.imdecode(tmp_array, cv2.IMREAD_COLOR)
        img_array_small = cv2.resize(img_array, (250,250))

        # Shape uploaded pic
        #st.markdown(img_array.shape)
        output = cv2.resize(img_array, (400,240))
        output = cv2.imencode('.jpg',output)[1].tobytes()

        # Shape downsized small pic
        #st.markdown(img_array_small.shape)
        # show smal pic
        st.image(output)
        st.write("##")
        st.write("##")
        st.write("##")

# Streamlit - Show file


###################################      PART 1 - GET COLOR NAMES    #############################################################



# Flow in streamlit
if img_array is not None:
    print(pic_theme)
    print(pic_format)



    # Select subject section
    with st.container():
        'Select a topic for your artwork'
        st.image("https://i.ibb.co/0fRMvrc/bgl.jpg")

        pic_theme = st.radio('', ('First', 'Abstract', 'Portrait', 'Mountains', 'Beach', 'City', 'Fruit', 'Plants', 'Flowers'))

        # no selection. first option invisible
        st.markdown(
        """ <style>
                div[role="radiogroup"] >  :first-child{
                    display: none !important;
                }
            </style>
            """,
        unsafe_allow_html=True
        )

        #if pic_theme is not None:
        #    st.write("Choosen subject")
        #    st.markdown(pic_theme)







        # Select Picture format
    with st.container():
        st.write("##")
        'Select the format'
        col1, col2 = st.columns((1, 18))
        pf = col1.image("https://i.ibb.co/pysQpr2/pf.jpg")

        lf = col2.image("https://i.ibb.co/GF2mynj/lf.jpg")

        #st.image("https://i.ibb.co/713Bvwr/rbf.jpg")
        pic_format = st.radio('', ('First', 'Portrait format', 'Landscape format'))

        #if format is not None:
        #    st.write("Chosen format:")
        #    st.markdown(pic_format)

    print(pic_theme)
    print(pic_format)

    if (pic_theme is not "-") and (pic_format is not "-"):

        st.write("---")
        st.write("##")
        if st.button('Generate'):
            'Your individual artwork will be painted...'

            import time

            if demo == False:

                # Clustering colors - determine number of clusters
                k_cluster = 15
                clt_1 = make_clusters(k_cluster,img_array)

                # Get dictionary of colors in picture and color palette
                palette, dic_color_per = palette_perc(clt_1)


                # Get dictionary with string names for the colors
                color_dict = get_color_names()

                # Attach name of colors in our dictionary with the colorcodes from user picture
                pic_cname_perc_rgb = all_color_names(dic_color_per,color_dict)
                #SL - Print list of all colors detected
                #st.markdown("Detected colors: ")
                #st.markdown(pic_cname_perc_rgb)


                # Get dictionary with complementary colors
                comp_colors_cname_perc_rgb = list_complentary_colors(dic_color_per,color_dict)
                #st.markdown("Complementory colors: ")
                #st.markdown(comp_colors_cname_perc_rgb)

                # Get the final color list

                #final_color_list(pic_cname_perc_rgb, comp_colors_cname_perc_rgb)
                # number of colors from real color and from complementary colors
                colors_real_pic = 2
                colors_complementary = 2
                final_list_colors_with_rgb = final_color_list(pic_cname_perc_rgb, comp_colors_cname_perc_rgb,colors_real_pic,colors_complementary)
                final_color_names = list_final_color_names(final_list_colors_with_rgb)

                #SL# - Show final color names
                #st.write("##")
                #st.markdown("this are the final 3 colors:")
                #st.markdown(final_color_names)

                #################################### END PART 1 - COLOR NAMES ################################################################





                ###############################   PART 2 - STYLE DETECTION ###############################################################



                # 2. Load data

                if train_model:
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
                    categories = ['traditional', 'elestic',  'industrial',  'midcentury', 'asian', 'farmhouse',  'mediterranean',
                                'modern', 'scandinavian']
                    #print(model_rn.summary())

                # 5. Predict Style of new pic
                # 5.1. Open image in 250x250
                img_pic = img_array_small

                # 5.2. Preprocess image - shape
                X_new_proc = new_pic_as_array(img_pic)

                # 5.3. Get the prediction
                y_new = model_rn.predict(X_new_proc)

                # 5.4. Make a sorted dictionary of the results
                pred_dict_sorted = style_result_dictionary(y_new)

                # 5.5. Just show the X top result design styles
                number_of_styles = 2
                top_designs = top_style_results(pred_dict_sorted, number_of_styles)

                #SL# - Show top 2 designs
                #st.markdown("These is the room style:")
                #st.markdown(top_designs)

                # 5.6 Get the result as lists
                top_desings_string = top_styles_string(top_designs)

                # 5.7. Convert the style types to descriptive adjectives (Scandinavian --> clena, simple, modern)
                # Number of adjectives per style to use
                num_adj_per_style = 2
                final_adj_style_list = translate_style_adjective(top_desings_string, num_adj_per_style)


                #########################   END 2. STYLE DETECTION   #####################################



                ##########################   PART 3 - PUT EVERYTHING TOGETHER ############################

                output_models = string_output(final_color_names, final_adj_style_list)
                # Combine them
                #prompt = pic_theme + output_models
                # fake-prompt to get good pictures
                prompt = "City skyline sunset royalblue mustardyellow clean bright"
                #st.write("##")
                #st.write("This is the output for our AI Generator")
                #st.markdown(prompt)


                #####################   PART 4 - API Call     -###################################

                # Header for API call - with API
                HEADERS = {'Authorization': config('API_KEY'), 'Content-Type': 'application/json'}

                # Input for API - from User Input
                pic_width, pic_height = format_picture(pic_format)
                styles_for_topic = get_style_id_list_for_topic(pic_theme)

                #st.write("These styles are used:")
                #st.markdown(styles_for_topic)



                # Define number of pics to create
                num_pictures_to_create = 3

            else:
                time.sleep(sleep_time)




            # Create the pictures Loop with API calls
            if generate_with_api:
                api_run_finished = create_multiple_pictures(prompt, pic_width, pic_height,
                                                        num_pictures_to_create,
                                                        styles_for_topic, HEADERS)
            else:
                api_run_finished = True


            if api_run_finished:
                with st.container():
                    '✅ Your pictures are ready!'
                    st.write("---")
                    # Show the results
                    st.write("##")
                    #image-0
                    img_path_0 = (os.path.join("pic-results","image-0.jpg"))
                    if pic_format == "Landscape format":
                        imgage_0 = load_img(path=img_path_0, target_size=(533,800), color_mode="rgb")
                    elif pic_format == "Portrait format":
                        imgage_0 = load_img(path=img_path_0, target_size=(800,533), color_mode="rgb")
                    with st.container():
                        st.write("### Your Individual Artwork - 1")
                        st.image(imgage_0, caption="Picture-1")
                        if demo:
                            st.write("##")
                            img_path_0b = (os.path.join("pic-results","image-0b.jpeg"))
                            imgage_0b = load_img(path=img_path_0b, color_mode="rgb")
                            st.image(imgage_0b, caption="In your living room")

                            st.write("##")
                            img_path_0c = (os.path.join("pic-results","image-0c.jpeg"))
                            imgage_0c = load_img(path=img_path_0c, color_mode="rgb")
                            st.image(imgage_0c, caption="With picture Frame")

                        st.write("#")

                    #image-1

                    img_path_1 = (os.path.join("pic-results","image-1.jpg"))
                    if pic_format == "Landscape format":
                        imgage_1 = load_img(path=img_path_1, target_size=(533,800), color_mode="rgb")
                    elif pic_format == "Portrait format":
                        imgage_1 = load_img(path=img_path_1, target_size=(800,533), color_mode="rgb")
                    with st.container():
                        st.write("#")
                        st.write("### Your Individual Artwork - 2")
                        st.image(imgage_1, caption="Picture-2")
                        if demo:
                            st.write("##")
                            img_path_1b = (os.path.join("pic-results","image-1b.jpeg"))
                            imgage_1b = load_img(path=img_path_1b, color_mode="rgb")
                            st.image(imgage_1b, caption="In your living room")

                            st.write("##")
                            img_path_1c = (os.path.join("pic-results","image-1c.jpeg"))
                            imgage_1c = load_img(path=img_path_1c, color_mode="rgb")
                            st.image(imgage_1c, caption="With picture Frame")

                        st.write("#")

                    #image-2
                    img_path_2 = (os.path.join("pic-results","image-2.jpg"))
                    if pic_format == "Landscape format":
                        imgage_2 = load_img(path=img_path_2, target_size=(533,800), color_mode="rgb")
                    elif pic_format == "Portrait format":
                        imgage_2 = load_img(path=img_path_2, target_size=(800,533), color_mode="rgb")
                    with st.container():
                        st.write("#")
                        st.write("### Your Individual Artwork - 3")
                        st.image(imgage_2, caption="Picture-3")
                        if demo:
                            st.write("##")
                            img_path_2b = (os.path.join("pic-results","image-2b.jpeg"))
                            imgage_2b = load_img(path=img_path_2b, color_mode="rgb")
                            st.image(imgage_2b, caption="In your living room")

                            st.write("##")
                            img_path_2c = (os.path.join("pic-results","image-2c.jpeg"))
                            imgage_2c = load_img(path=img_path_2c, color_mode="rgb")
                            st.image(imgage_2c, caption="With picture Frame")







#if __name__ == '__main__':
    #try:
        #print(final_color_names)
        #print(final_adj_style_list)
        #print("final string-output:  ",output_string)
    #except:
        #pass
