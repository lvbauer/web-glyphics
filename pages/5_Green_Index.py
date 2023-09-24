import streamlit as st
import cv2
import numpy as np
from PIL import Image
from func import greenindex as gidx

st.set_page_config(page_title="Green Index Calculator")

def main():

    st.title("Green Index Calculator")
    st.header("Image Upload")

    # Create tabs for upload images or using camera
    input1, input2 = st.tabs(["Upload Image", "Use Camera"])
    with input1:
        user_image_upload = st.file_uploader("Picture to analyze:", type=["png","jpg","tiff", "tif","jpeg"], 
                                             accept_multiple_files=True, key="file_uploader")
    with input2:    
        user_image_camera = st.camera_input("Use Device Camera")

    # Give priority to uploaded images
    if (user_image_upload is not None):
        user_image = user_image_upload
    else:
        user_image = list()
        user_image.append(user_image_camera)

    if (len(user_image) == 1):

        # Load file into array using PIL
        img = Image.open(user_image[0])
        img = np.array(img)
        
        # Original image display
        st.header("Original Image")
        st.image(img)

        # Instantiate Final Image
        final_image = gidx.calc_green_index_array(img)
        st.header("Green Index Images")
        st.image(final_image)

    elif (len(user_image) > 1):
        st.warning("Multiple images currently not supported.")
        st.stop()
        pass

if __name__ == "__main__":
    main()