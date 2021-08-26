from genericpath import exists
import torch
import streamlit as st
from streamlit import components
from streamlit import file_util
import streamlit.report_thread as ReportThread
from streamlit.server.server import Server
from streamlit import caching
from streamlit import script_runner
from unet.model import UNet as Model
from utils.utils import Interpolate
from utils.spherical import Spherical
from utils.spherical_deprojection import SphericalDeprojection
from utils.visualizers import Visualizers
import os
import platform
try:
    from torch.hub import load_state_dict_from_url
except ImportError:
    from torch.utils.model_zoo import load_url as load_state_dict_from_url
import omegaconf.omegaconf
import io
from PIL import Image
import numpy as np
#import open3d as o3d
import time


model_urls = {
    'UNet': 'https://github.com/tzole1155/StreamLitDemo/releases/download/Unet/unet.pth',
}

@st.cache(allow_output_mutation=True, ttl=120000, max_entries=1)
def init_model(choice:str):
    #device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device = torch.device("cpu")
    if choice == 'UNet':
        #load model
        #read configuration file
        conf_path = './conf/unet/model_unet.yaml'
        #model_pt_path = './weights/unet/unet.pth'
        #download weights from github
        # state_dict = load_state_dict_from_url(model_urls[choice],
        #                                       progress=True)
        # if not os.path.isfile(model_pt_path):
        #         error_message = f"Missing the serialized model weights file({model_pt_path})"
        #         st.error(error_message)
        #         raise RuntimeError(error_message)
        if not os.path.isfile(conf_path):
                error_message = f"Missing the model's configuration file({conf_path})"
                st.error(error_message)
                raise RuntimeError(error_message)
        conf = omegaconf.OmegaConf.load(conf_path)
        model = Model(conf.model.configuration)
        # checkpoint = torch.load(model_pt_path,
        #     map_location=lambda storage, loc: storage
        #     )
        checkpoint = load_state_dict_from_url(model_urls[choice],
                                              map_location=lambda storage, loc: storage,
                                              progress=True)

        model.load_state_dict(checkpoint['state_dict'], False)
    
    model.to(device)
    model.eval()
    
    return model,device

def preprocess(bytes):
    raw = io.BytesIO(bytes)
    image = Image.open(raw)
    image = image.resize((512,256),Image.LANCZOS)
    image = np.array(image) # cvt color?
    image = image.transpose(2, 0, 1)
    image = torch.from_numpy(image).unsqueeze(0).float() / 255.0

    return image

def inference(input,model,device):
    with torch.no_grad():
        depth = model(input.to(device))
    
    return depth

def visualise_outputs(color,depth):
    device = depth.get_device()
    viz = Visualizers()
    imgs = viz.export_depth(depth)
    #visualise depth map
    st.image(imgs[0].transpose(1, 2, 0))
    #point cloud exporter
    sgrid = Spherical(width=512,mode='pi',long_offset_pi=-0.5).to(device)(imgs)
    pcloud = SphericalDeprojection().to(device)(depth,sgrid)
    static_path = file_util.get_static_dir()
    pred_xyz = pcloud[0]
    rgb = color[0] * 255.0
    if isinstance(rgb,np.ndarray):
        colors = rgb.reshape(3, - 1).astype(np.uint8)
    else:
        colors = rgb.reshape(3, - 1).cpu().numpy().astype(np.uint8)    
    pred_filename = os.path.join(static_path,'pred_pointcloud.ply')
    if os.path.isfile(pred_filename):
        os.remove(pred_filename)
    pred_xyz = pred_xyz.reshape(3, -1).cpu().numpy()
    #write_ply(pred_filename, pred_xyz, None, colors)
    viz.write_ply(pred_filename, pred_xyz, None, colors)
    time.sleep(0.1)
    #export mesh
    mesh = viz.export_mesh(pred_xyz,colors)
    #save mesh
    # mesh_filename = os.path.join(static_path,"pred_mesh.obj")
    # if os.path.isfile(mesh_filename):
    #     os.remove(mesh_filename)
    # o3d.io.write_triangle_mesh(mesh_filename, mesh, write_triangle_uvs=True)
    # time.sleep(0.1)




def main():
    st.set_page_config(layout="wide")
    st.title('Pano3D 360 depth estimator')

    current_version = platform.release()

    st.write(current_version)

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
        visualise_outputs(input,depth)
        text_file = open("./html/ply.html", "r")
        #read whole file to a string
        html_string = text_file.read()
        #time.sleep(1.5)
        #close file
        text_file.close()
        #breakpoint()
        h1 = components.v1.html(html_string, height=600)
        #visualize mesh
        # text_file = open("./html/mesh.html", "r")
        # #read whole file to a string
        # html_string = text_file.read()
        # #time.sleep(1.5)
        # #close file
        # text_file.close()
        # h2 = components.v1.html(html_string, height=600)
   


if __name__ == '__main__':
    main()
