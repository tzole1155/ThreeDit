from genericpath import exists
import PIL
import torch
import streamlit as st
from streamlit import components
from streamlit import file_util
# try:
#     #import streamlit.ReportThread as ReportThread
#     from streamlit import ReportThread
#     from streamlit import Server
# except:
#     import streamlit.report_thread as ReportThread
#     from streamlit.server.server import Server
# from streamlit import caching
# from streamlit import script_runner
from unet.model import UNet
from pnas.model import MultiBranch
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
import open3d as o3d # this should not be included when deployed in GitHub
from streamlit.web.server.routes import StaticFileHandler
import urllib.request
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import imageio
from pathlib import Path
import tempfile
from os.path import dirname, abspath




model_urls = {
    'UNet': 'https://github.com/tzole1155/ThreeDit/releases/download/Unet/unet.pth',
    'Pnas_pre': 'https://github.com/tzole1155/ThreeDit/releases/download/Pnas/360V_pnas-epoch.49-rmse.0.41.ckpt'
}

@classmethod
def _get_cached_version(cls, abs_path: str):
    #taken from https://discuss.streamlit.io/t/html-file-cached-by-streamlit/9289
    with cls._lock:
        return cls.get_content_version(abs_path)


@st.cache_resource
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
        model = UNet(conf.model.configuration)
        # checkpoint = torch.load(model_pt_path,
        #     map_location=lambda storage, loc: storage
        #     )
        checkpoint = load_state_dict_from_url(model_urls[choice],
                                              map_location=lambda storage, loc: storage,
                                              progress=True)

        model.load_state_dict(checkpoint['state_dict'], False)
    
    elif choice == 'Pnas_pre':
        checkpoint = load_state_dict_from_url(model_urls[choice],
                                              map_location=lambda storage, loc: storage,
                                              progress=True)
        conf_path = './conf/pnas/model_pnas.yaml'
        if not os.path.isfile(conf_path):
                error_message = f"Missing the model's configuration file({conf_path})"
                st.error(error_message)
                raise RuntimeError(error_message)
        conf = omegaconf.OmegaConf.load(conf_path)
        model = MultiBranch(conf.model.configuration)
        model.load_state_dict(checkpoint['state_dict'], False)
        #file = torch.load(checkpoint)
        #model.load_state_dict(toolz.keymap(lambda k: k.replace('module.', ''), file))
        #breakpoint()

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

def get_depth_map(viz,depth,static_path):
    st.markdown("## Predicted depth map", unsafe_allow_html=True)
    pred_filename = os.path.join(static_path,'pred_depth.exr')
    if os.path.isfile(pred_filename):
            os.remove(pred_filename)
    imgs = viz.export_depth(depth,static_path)
    #break
    return imgs

def get_exr_map(depth,static_path):
    pred_filename = os.path.join(static_path,'pred_depth.exr')
    imageio.imwrite(pred_filename, (depth.cpu().numpy())[0, :, :, :].transpose(1,2,0))

def get_point_cloud(viz,depth,color,static_path):
    device = depth.get_device()
    if device == -1:
        device = 'cpu'
    sgrid = Spherical(width=512,mode='pi',long_offset_pi=-0.5).to(device)(depth)
    isPcloud = False
    while not isPcloud:
        pc_warn = st.warning("Creating point cloud...")
        pcloud = SphericalDeprojection().to(device)(depth,sgrid)
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
        viz.write_ply(pred_filename, pred_xyz, None, colors)
        if torch.is_tensor(pcloud):
            isPcloud=True
            pc_warn.empty()
    
    return pred_xyz,colors


def export_mesh(viz,pred_xyz,colors,static_path):
    isMesh = False
    while not isMesh:
        mes_warn = st.warning("Reconstructing mesh...")
        mes_warn_2 =  st.warning("This might take a while...")
        mesh = viz.export_mesh(pred_xyz,colors)
        #save mesh
        mesh_filename = os.path.join(static_path,"pred_mesh.obj")
        if os.path.isfile(mesh_filename):
            os.remove(mesh_filename)
        o3d.io.write_triangle_mesh(mesh_filename, mesh, write_triangle_uvs=True)
        if mesh:
            isMesh = True
            mes_warn.empty()
            mes_warn_2.empty()

def trigger_rerun():

    ctx = ReportThread.get_report_ctx()

    this_session = None
    
    current_server = Server.get_current()
    if hasattr(current_server, '_session_infos'):
        # Streamlit < 0.56        
        session_infos = Server.get_current()._session_infos.values()
    else:
        session_infos = Server.get_current()._session_info_by_id.values()

    for session_info in session_infos:
        s = session_info.session
        this_session = s
        if (
            # Streamlit < 0.54.0
            (hasattr(s, '_main_dg') and s._main_dg == ctx.main_dg)
            or
            # Streamlit >= 0.54.0
            (not hasattr(s, '_main_dg') and s.enqueue == ctx.enqueue)
        ):
            this_session = s

    if this_session is None:
        raise RuntimeError(
            "Oh noes. Couldn't get your Streamlit Session object"
            'Are you doing something fancy with threads?')
    #this_session.request_rerun()
    this_session.handle_rerun_script_request(is_preheat=True)


def main():
    # Do not cache files by filename.
    StaticFileHandler._get_cached_version = _get_cached_version
    st.set_page_config(layout="wide")
    with tempfile.TemporaryDirectory() as static_path:
        st.write('Created temporary directory:', static_path)
    # static_path = file_util.get_static_dir()
    # static_path = './app/static'
    # static_path = "/mount/src/threedit/static"
    if not os.path.exists(static_path):
        # Create the static folder
        os.mkdir(static_path)
    urllib.request.urlretrieve(
        'https://raw.githubusercontent.com/tzole1155/ThreeDit/main/Images/Banner.png',
        os.path.join(static_path,"banner.png"))
    banner = PIL.Image.open(os.path.join(static_path,"banner.png"))
    # static_path = dirname(dirname(abspath(os.path.join(static_path,"banner.png"))))
    st.image(banner, use_column_width  = True)
    st.markdown("<h1 style='text-align: center; color: white;'>Reconstruct your room form a single panorama</h1>", unsafe_allow_html=True)

    text_file = open("intro.md", "r")
    #read whole file to a string
    md_string = text_file.read()
    #close file
    text_file.close()

    readme_text = st.markdown(md_string)

    menu = ['UNet','Pnas_pre']
    st.sidebar.header('Model Selection')
    choice = st.sidebar.selectbox('Please select one of the available models ?', menu)
    
    #init model
    model, device = init_model(choice)

    viz = Visualizers()

    Image = st.file_uploader('Upload your panorama here',type=['jpg','jpeg','png'])
    if Image is not None:
        #col1, col2 = st.beta_columns(2)
        st.markdown("## Input Panorama", unsafe_allow_html=True)
        col1, col2, col3 = st.columns([3,4,3])
        with col1:
            st.write("")
        Image = Image.read()
        #st.text(type(Image))
        with col2:
            st.image(Image,use_column_width=True,caption='Input panorama')
        with col3:
            st.write("")
        #process image
        input = preprocess(Image)
        #run model
        isDepth = False
        while not isDepth:
            md_warn = st.warning("Model is running...")
            depth = inference(input,model,device)
            if torch.is_tensor(depth):
                isDepth = True
                md_warn.empty()
        #visualise outputs
        imgs = get_depth_map(viz,depth,static_path)
        #if st.button('Download depth map'):
        #get_exr_map(depth,static_path)
        col1, col2, col3 = st.columns([3,4,3])
        with col1:
            st.write("")
        with col2:
            st.image(imgs[0].transpose(1, 2, 0),use_column_width=True,caption='Predicted depth map')
            #st.markdown("<p style='text-align: center;'>Predicted depth map</p>", unsafe_allow_html=True)
        with col3:
            st.write("")
        #download depth map
        #TODO: fix this
        #linko= f'<a href="pred_depth.exr" download="pred_depth.exr"><button kind="primary" class="css-15r570u edgvbvh1">Download Predicted Depth Map!</button></a>'
        #st.markdown(linko, unsafe_allow_html=True)
        #point cloud
        pred_xyz,colors = get_point_cloud(viz,depth,input,static_path)
        ply_file_path = f"{static_path}/pred_pointcloud.ply"
        text_file = open("./html/ply.html", "r")
        # read whole file to a string
        # with open('./html/ply.html', 'r') as file:
        #     myfile = file.read().replace('\n', '')
        # html_string = text_file.read()
        # NOTE: DEBUG
        ply_content = b""
        import base64
        with open(ply_file_path, "rb") as f:
            ply_content = f.read()
        # Convert to base64 for embedding in HTML/JS
        ply_base64 = base64.b64encode(ply_content).decode()

        html_string = f"""
                <!DOCTYPE html>
                <html lang="en">

                <head>
                    <title>three.js webgl - PLY</title>
                    <meta charset="utf-8">
                    <meta name="viewport" content="width=device-width, user-scalable=no, minimum-scale=1.0, maximum-scale=1.0">
                    <meta http-equiv="Cache-Control" content="no-cache, no-store, must-revalidate" />
                    <meta http-equiv="Expires" content="-1" />

                </head>

                <body>


                    <script type="module">

                        import * as THREE from 'https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.module.js';
                        import {{ PLYLoader }} from 'https://cdn.jsdelivr.net/npm/three@0.121.1/examples/jsm/loaders/PLYLoader.js';
                        import {{ OrbitControls }} from 'https://tzole1155.github.io/360depth/js/controls/OrbitControls.js';
                        let container, stats;
                        let camera, cameraTarget, scene, renderer;

                        THREE.Cache.enabled = false

                        var ply_base64 = '{ply_base64}';
                        var ply_data = base64ToArrayBuffer(ply_base64);
                        
                        init();
                        animate();
                        

                        function init() {{

                            THREE.Cache.clear();

                            container = document.createElement('div');
                            document.body.appendChild(container);
                            camera = new THREE.PerspectiveCamera(20, window.innerWidth / window.innerHeight, 1, 15);
                            camera.position.set(4.25, 2.92, 6.56);
                            cameraTarget = new THREE.Vector3(0, - 0.1, 0);
                            scene = new THREE.Scene();
                            scene.background = new THREE.Color(0x333F49);
                            const controls = new OrbitControls(camera, container);
                            controls.enableDamping = true;

                            let loader = new PLYLoader();
                            let geometry = loader.parse(ply_data);
                            
                            geometry.computeFaceNormals();

                            let material = new THREE.PointsMaterial({{ size: 0.01 }});
                            material.vertexColors = true;
                            material.needsUpdate = true;
                            geometry.needsUpdate = true;
                            let mesh = new THREE.Points(geometry, material);
                            mesh.position.x = 0;
                            mesh.position.y = -1;
                            mesh.position.z = 0;
                            mesh.rotateY(Math.PI / 2);
                            mesh.rotateZ(Math.PI);
                            mesh.scale.multiplyScalar(0.5);
                            mesh.castShadow = true;
                            mesh.receiveShadow = true;
                            scene.add(mesh);

                            // Lights

                            scene.add(new THREE.HemisphereLight(0x443333, 0x111122));

                            // renderer
                            renderer = new THREE.WebGLRenderer({{ antialias: true }});
                            renderer.setPixelRatio(window.devicePixelRatio);
                            renderer.setSize(window.innerWidth, window.innerHeight);
                            renderer.outputEncoding = THREE.sRGBEncoding;
                            renderer.shadowMap.enabled = true;
                            container.appendChild(renderer.domElement);
                            // resize
                            window.addEventListener('resize', onWindowResize);
                        }}

                        function addShadowedLight(x, y, z, color, intensity) {{

                            const directionalLight = new THREE.DirectionalLight(color, intensity);
                            directionalLight.position.set(x, y, z);
                            scene.add(directionalLight);

                            directionalLight.castShadow = true;

                            const d = 1;
                            directionalLight.shadow.camera.left = - d;
                            directionalLight.shadow.camera.right = d;
                            directionalLight.shadow.camera.top = d;
                            directionalLight.shadow.camera.bottom = - d;

                            directionalLight.shadow.camera.near = 1;
                            directionalLight.shadow.camera.far = 4;

                            directionalLight.shadow.mapSize.width = 1024;
                            directionalLight.shadow.mapSize.height = 1024;

                            directionalLight.shadow.bias = - 0.001;

                        }}

                        function onWindowResize() {{

                            camera.aspect = window.innerWidth / window.innerHeight;
                            camera.updateProjectionMatrix();
                            renderer.setSize(window.innerWidth, window.innerHeight);

                        }}

                        function animate() {{
                            requestAnimationFrame(animate);
                            render();
                        }}

                        function render() {{
                            renderer.render(scene, camera);
                        }}

                        // Decoding base64 to ArrayBuffer
                        function base64ToArrayBuffer(base64) {{
                            var binary_string = window.atob(base64);
                            var len = binary_string.length;
                            var bytes = new Uint8Array(len);
                            for (var i = 0; i < len; i++) {{
                                bytes[i] = binary_string.charCodeAt(i);
                            }}
                            return bytes.buffer;
                        }}

                    </script>
                </body>

                </html>
        """
        # NOTE: DEBUG
        # Inject the ply_file_path into the JavaScript code
        # html_string = html_string.replace('{ PLYLoader }', '{{ PLYLoader }}')
        # html_string = html_string.replace('{ OrbitControls }', '{{ OrbitControls }}')
        # print(html_string)
        # st.markdown(f"html_string {html_string}", unsafe_allow_html=True)
        # html_string = html_string.format(
        #     ply_file_path
        #     )
        # html_string = html_string.replace('{ply_file_path}', ply_content)
        # st.markdown(f"html_string {html_string}", unsafe_allow_html=True)
        # st.markdown(f"static path {os.listdir(static_path)}", unsafe_allow_html=True)
        # st.markdown("## Predicted Point Cloud", unsafe_allow_html=True)
        # st.markdown("Inspect the predicted point cloud through the interactive 3D Model Viewer", unsafe_allow_html=True)
        h1 = components.v1.html(html_string, height=600)
        st.success("Point cloud has been created!")
        # linko= f'<a href="{static_path}/pred_pointcloud.ply" download="pred_pointcloud.ply"><button kind="primary" class="css-15r570u edgvbvh1">Download Point Cloud!</button></a>'
        # st.markdown(linko, unsafe_allow_html=True)
        # Ensure the file exists
        try:
            with open(f"{static_path}/pred_pointcloud.ply", 'rb') as file:
                st.download_button(
                    label="Download Point Cloud!",
                    data=file,
                    file_name="pred_pointcloud.ply",
                    mime="application/octet-stream",
                )
        except FileNotFoundError:
            st.error('File not found!')
        #mesh
        # if st.button('Reconstruct mesh'):
        #     export_mesh(viz,pred_xyz,colors,static_path)
        #     text_file = open("./html/mesh.html", "r")
        #     #read whole file to a string
        #     html_string = text_file.read()
        #     text_file.close()
        #     st.markdown("## Reconstructed Mesh", unsafe_allow_html=True)
        #     st.markdown("Inspect the reconstructed mesh through the interactive 3D Model Viewer", unsafe_allow_html=True)
        #     h2 = components.v1.html(html_string, height=600)
        #     st.success("Mesh has been created!")
        #     linko= f'<a href="pred_mesh.obj" download="pred_mesh.obj"><button kind="primary" class="css-15r570u edgvbvh1">Download Reconstructed Mesh!</button></a>'
        #     st.markdown(linko, unsafe_allow_html=True)
        #     #Acknow
        #     text_file = open("Ackn.md", "r")
        #     #read whole file to a string
        #     md_string = text_file.read()
        #     #close file
        #     text_file.close()
        #     ack_text = st.markdown(md_string)

   


if __name__ == '__main__':
    main()
