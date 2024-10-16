import numpy as np
import tensorflow as tf
from tensorflow.keras import models, layers
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
import pickle
import streamlit as st
from PIL import Image
from io import BytesIO

loaded_model = tf.keras.models.load_model(
    'C:/Users/User/Desktop/Python Projects/potato disease detection/notebooks/training.keras',
    compile=False
)
loaded_model.compile(optimizer='adam', loss='categorical_crossentropy')

def main():
    def pred(uploaded_file):
        image = Image.open(uploaded_file)
        image = image.convert('RGB')
        image = image.resize((224, 224))

        img_array = tf.keras.preprocessing.image.img_to_array(image)
        img_array = tf.expand_dims(img_array, 0)
        predictions = loaded_model.predict(img_array)
        d_class = np.argmax(predictions[0])
        if d_class == 0:
            predicted_class = 'Potato___Early_blight'
        elif d_class == 1:
            predicted_class = 'Potato___Late_blight'
        else:
            predicted_class = 'Potato___healthy'
        confidence = round(100 * (np.max(predictions[0])), 2)
        return predicted_class, confidence
    
    #title
    st.title('Potato Disease Classification')
    

    st.title("Upload Image here")
    

    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])


    if uploaded_file is not None:

       file_content = uploaded_file.read()
    
 
       image = Image.open(BytesIO(file_content))
    

       image_array = np.array(image)
    

       st.image(image, caption='Uploaded Image.', use_column_width=True)
       st.write(f"Image shape: {image_array.shape}")
    else:
       st.write("Please upload an image file.")
    
    #code for prediction
    diagnosis = ''
    
    if st.button('Result'):
        diagnosis = pred(uploaded_file)
        
        
    st.success(diagnosis)
    
    
if __name__ == '__main__':
    main()
    
    