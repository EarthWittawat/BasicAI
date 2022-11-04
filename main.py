"""Create an Image Classification Web App using PyTorch and Streamlit."""
# import libraries
from this import d
from PIL import Image
from fastai.vision.widgets import *
from fastai.vision.all import *
from torchvision import models, transforms
import torch
import streamlit as st
import time
import os
import os.path
import urllib.request
from pathlib import Path
from csv import writer
import pathlib

st.set_page_config(
    page_title="Crossec ML",
    layout="wide",
)  
file_exists = os.path.exists('crossec_model.pkl')

plt = platform.system()
if plt == 'Windows': pathlib.PosixPath = pathlib.WindowsPath
if file_exists == False: 
     MODEL_URL = "https://dl.dropboxusercontent.com/s/k66f4yi8i0mlalp/crossec_model.pkl?dl=0"
     urllib.request.urlretrieve(MODEL_URL,"crossec_model.pkl")
learn_inf = load_learner(Path()/'crossec_model.pkl',cpu=True)

tissue = [
 'ใบพืช C3', 'ใบพืช C4', 'รากพืชใบเลี้ยงคู่ระยะปฐมภูมิ','รากพืชใบเลี้ยงคู่ระยะทุติยภูมิ','ลำต้นพืชใบเลี้ยงคู่ระยะปฐมภูมิ','ลำต้นพืชใบเลี้ยงคู่ระยะทุติยภูมิ','รากพืชใบเลี้ยงเดี่ยวระยะปฐมภูมิ','ลำต้นพืชใบเลี้ยงเดี่ยวระยะปฐมภูมิ'
]

def predict(image, learn):
        """Return top 5 predictions ranked by highest probability.

        Parameters
        ----------
        :param image: uploaded image
        :type image: jpg
        :rtype: list
        :return: top 5 predictions ranked by highest probability
        """
        # create a ResNet model
        pred, pred_idx, pred_prob = learn.predict(image)

        classes = tissue[int(pred_idx)]
        
        return [(classes, pred_prob[pred_idx])]


with st.container():
    st.subheader("Classifier")
    st.write("")
    file_up = st.file_uploader("Upload an image", type = ["jpg","png"])
    if file_up is not None:
        # display image that user uploaded
        image = Image.open(file_up)
        st.image(image, caption = 'Uploaded Image.', use_column_width = True)
        st.write("")
        pred = predict(np.asarray(image), learn_inf)
        class_names = ['ใบพืช C3', 'ใบพืช C4', 'รากพืชใบเลี้ยงคู่ระยะปฐมภูมิ','รากพืชใบเลี้ยงคู่ระยะทุติยภูมิ','ลำต้นพืชใบเลี้ยงคู่ระยะปฐมภูมิ','ลำต้นพืชใบเลี้ยงคู่ระยะทุติยภูมิ','รากพืชใบเลี้ยงเดี่ยวระยะปฐมภูมิ','ลำต้นพืชใบเลี้ยงเดี่ยวระยะปฐมภูมิ']
        result = class_names[np.argmax(pred)]
        output = 'The image is a ' + result
            # print out the top 5 prediction labels with scores
        st.success(output)
    st.write("")


