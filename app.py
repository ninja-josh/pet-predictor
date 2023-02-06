import streamlit as st
import numpy as np
from lime import lime_image
from skimage.segmentation import mark_boundaries
import matplotlib.pyplot as plt
from PIL import Image
from src import *

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
    conv_image = model.convert_image(image)
    explainer = lime_image.LimeImageExplainer()
    exp = explainer.explain_instance(conv_image[0],
                                model.model.predict,)
    
    image, mask = exp.get_image_and_mask(0,
                                     positive_only=False, 
                                     negative_only=False,
                                     hide_rest=False,
                                     min_weight=.05
                                    )
    st.image(mark_boundaries(image, mask), use_column_width='always')
