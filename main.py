import torch
import streamlit as st
from streamlit import components
from unet.model import UNet as Model
from utils.utils import Interpolate
from utils.spherical import Spherical
from utils.spherical_deprojection import SphericalDeprojection
from utils.visualizers import Visualizers
import os
import omegaconf.omegaconf
import io
from PIL import Image
import numpy as np

def init_model(choice:str):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if choice == 'UNet':
        #load model
        #read configuration file
        conf_path = './conf/unet/model_unet.yaml'
        model_pt_path = './weights/unet/unet.pth'
        if not os.path.isfile(model_pt_path):
                error_message = f"Missing the serialized model weights file({model_pt_path})"
                st.error(error_message)
                raise RuntimeError(error_message)
        if not os.path.isfile(conf_path):
                error_message = f"Missing the model's configuration file({conf_path})"
                st.error(error_message)
                raise RuntimeError(error_message)
        conf = omegaconf.OmegaConf.load(conf_path)
        model = Model(conf.model.configuration)
        checkpoint = torch.load(model_pt_path,
            map_location=lambda storage, loc: storage
            )
        model.load_state_dict(checkpoint['state_dict'], False)
    
    model.to(device)
    model.eval()
    
    return model,device

def preprocess(bytes):
    raw = io.BytesIO(bytes)
    image = Image.open(raw)
    image = np.array(image) # cvt color?
    image = image.transpose(2, 0, 1)
    image = torch.from_numpy(image).unsqueeze(0).float() / 255.0

    #return resized image
    resize = Interpolate(width=512,height=256)
    image = resize(image)

    return image

def inference(input,model,device):
    with torch.no_grad():
        depth = model(input.to(device))
    
    return depth

def visualise_outputs(depth):
    viz = Visualizers()
    imgs = viz.export_depth(depth)
    #visualise depth map
    st.image(imgs[0].transpose(1, 2, 0))
   



def main():
    st.set_page_config(layout="wide")
    st.title('Pano3D 360 depth estimator')

    menu = ['UNet']
    st.sidebar.header('Model Selection')
    choice = st.sidebar.selectbox('How would you like to be turn ?', menu)
    
    #init model
    model, device = init_model(choice)

    Image = st.file_uploader('Upload your panorama here',type=['jpg','jpeg','png'])
    if Image is not None:
        col1, col2 = st.beta_columns(2)
        Image = Image.read()
        #st.text(type(Image))
        with col1:
            st.image(Image)
        #process image
        input = preprocess(Image)
        #run model
        depth = inference(input,model,device)    
        #visualise outputs
        visualise_outputs(depth)
        text_file = open("./html/test.html", "r")
        #read whole file to a string
        html_string = text_file.read()
        #close file
        text_file.close()
        components.v1.html(html_string, height=600)


if __name__ == '__main__':
    main()
