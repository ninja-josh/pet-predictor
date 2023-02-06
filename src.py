from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import InputLayer, Flatten, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications import ResNet50V2
import pickle
from PIL import Image
import numpy as np
import base64
from io import BytesIO
import re
from lime import lime_image
from skimage.segmentation import mark_boundaries

class ResnetModel():
    #
    def __init__(self):
        """instantiate the model object"""
        self.model = self.create_ResNet()
    
    def create_ResNet(self):
        """Builds the model using a ResNet50V2 pretrained on imagenet as the first layers 
        and loads 2 pretrained hidden dense layers and an output layer from weights."""
        
        resnet = ResNet50V2(include_top=False, weights='imagenet')
        
        dense_1 = Dense(128, activation='relu')
        dense_2 = Dense(128, activation='relu')
        dense_3 = Dense(1, activation='sigmoid')


        model = Sequential()
        model.add(InputLayer(input_shape=(100, 100, 3)))
        model.add(resnet)
        model.add(Flatten())
        model.add(dense_1)
        model.add(dense_2)
        model.add(dense_3)
        
        dense_1_weights = pickle.load(open('weights/dense_1_weights.pkl', 'rb'))
        dense_2_weights = pickle.load(open('weights/dense_2_weights.pkl', 'rb'))
        dense_3_weights = pickle.load(open('weights/dense_3_weights.pkl', 'rb'))

        dense_1.set_weights(dense_1_weights)
        dense_2.set_weights(dense_2_weights)
        dense_3.set_weights(dense_3_weights)
        
        #It is not necessary to compile a model in order to make a prediction

        return model
            
    def convert_image(self, image):
        """Convert an image file into the right format and size for the model"""
        
        img = Image.open(image)
        img = img.resize((100,100))
        img = np.asarray(img)
        img = img.reshape((1,100,100,3))
        img = img / 255
        self.img = img
        
        return img
        
    def predict_pet(self, image):
        """Return a prediction, dog or cat, and confidence for a passed image file"""
        
        self.convert_image(image)
        proba = self.model.predict(self.img)[0][0]
        
        if proba >= .6:
            certainty = int(proba * 100)
            return f"I am {certainty}% certain this is a dog"
        elif proba <= .4: 
            certainty = int((1 - proba)*100)
            return f"I am {certainty}% certain this is a cat"
        else:
            return f"I don't have a clue what this is.  Would you like to try a different image?"
    
    def explain_prediction(self):
        try:

            explainer = lime_image.LimeImageExplainer()
            exp = explainer.explain_instance(self.img[0],
                                        self.model.predict)
            image, mask = exp.get_image_and_mask(0,
                                             positive_only=False, 
                                             negative_only=False,
                                             hide_rest=False,
                                             min_weight=0.1
                                            )
            return mark_boundaries(image, mask)
        except AttributeError:
            print("Model not fit yet.  Please make a prediction")
