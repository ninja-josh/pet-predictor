import streamlit as st
import numpy as np
from lime import lime_image
from skimage.segmentation import mark_boundaries
import matplotlib.pyplot as plt
from PIL import Image
from src import ResnetModel

#Create the model
model = ResnetModel()

#Create a header for the webpage
st.title('Is That a Dog or a Cat?')

#create an upload widget to receive an image file.  
#Give it a name and define allowed file types
image = st.file_uploader("Upload Your Pet!", type=["png","jpeg","jpg"])

#make a prediction and return the results on a new text line
if image:
    result = st.empty()
    st.image(image)
    result.write('Inspecting Image...')
    response = model.predict_pet(image)
    result.write(response)

if image:
    st.header('I think this because I see...\nGreen gives me confidence, red gives me doubt')

    st.image(model.explain_prediction(), use_column_width='always')
