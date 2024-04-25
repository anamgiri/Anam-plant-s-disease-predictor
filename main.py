import streamlit as st
import tensorflow as tf
import numpy as np


#Tensorflow Model Prediction
def model_prediction(test_image):
    model = tf.keras.models.load_model('trained__model.keras')
    image = tf.keras.preprocessing.image.load_img(test_image,target_size=(128,128))
    input_arr = tf.keras.preprocessing.image.img_to_array(image)
    input_arr = np.array([input_arr]) #convert single image to batch
    predictions = model.predict(input_arr)
    return np.argmax(predictions) #return index of max element

#Sidebar
st.sidebar.title("Dashboard")
app_mode = st.sidebar.selectbox("Sidebar",["Home","Identify your Plant's Disease"])

#Main Page
if(app_mode=="Home"):
    st.header("Anam's Plant disease identifier")
    image_path='photoforapp.jpg'
    st.image(image_path,use_column_width=True)
    st.markdown("""
     Welcome to Anam's Plant Disease Identifier! 
    
    I built this project to help Nepali farmers in agriculture. Though Nepal's more than 80% population is engaged in agriculture, most of the agricultural crops are imported from foreign countries. This is mainly due to plant disease and lack of use of modern techonology. So, I built a model to help Nepali farmers to identify any signs of disease in their plants with the help of plant leaf's picture. Use this to make sure you have a healthy harvest!

    #How to Use it?
    1. Upload your plant's leaf's image on Disease Recognition Page
    2. Then our system will identify the disease in your plant. (Currently, it can identify 50+ disease in the most common plants of Nepali agriculture such as rice( rice leaf smut, rice brown smut), tomato(Tomato Mosaic Virus, Tomato target spot), apple, etc. )

   Why Use this web app?
    -This was made with advanced AI and machine learning techniques and it is over 98 percent accurate
    - You can know about your plant's disease within seconds and also use the chatbot to learn about the disease and how can you cure it to make sure you have maximum production of crops.
                
    Also, I want to thank all the people who uploaded different disease's datasets on kaggle. I used those datasets to train my model.

    """)

#Prediction Page
elif(app_mode=="Identify your Plant's Disease"):
    st.header("Identify your Plant's Disease")
    test_image = st.file_uploader("Choose an Image:")
    if(st.button("Show Image")):
        st.image(test_image,width=4,use_column_width=True)
    #Predict button
    if(st.button("Predict")):
        st.snow()
        st.write("Our Prediction")
        result_index = model_prediction(test_image)
        #Reading Labels
        class_name = ['Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy', 'Blueberry___healthy', 'Cherry_(including_sour)___Powdery_mildew', 'Cherry_(including_sour)___healthy', 'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot', 'Corn_(maize)___Common_rust_', 'Corn_(maize)___Northern_Leaf_Blight', 'Corn_(maize)___healthy', 'Grape___Black_rot', 'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 'Grape___healthy', 'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot', 'Peach___healthy', 'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy', 'Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy', 'Raspberry___healthy', 'Rice_Bacterial_leaf_blight', 'Rice_Brown_spot', 'Rice_Leaf_smut', 'Soybean___healthy', 'Squash___Powdery_mildew', 'Strawberry___Leaf_scorch', 'Strawberry___healthy', 'Tomato___Bacterial_spot', 'Tomato___Early_blight', 'Tomato___Late_blight', 'Tomato___Leaf_Mold', 'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites Two-spotted_spider_mite', 'Tomato___Target_Spot', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus', 'Tomato___healthy']
        st.success("Model is Predicting it's a {}".format(class_name[result_index]))
