import streamlit as st
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

