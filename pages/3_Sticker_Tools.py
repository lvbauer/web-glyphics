import streamlit as st
import cv2
import numpy as np
from PIL import Image
from func import airisquare as asq

# Set page name
st.set_page_config(page_title="Astrobotany Sticker Tools")

# Overall tool options list
TOOL_OPTIONS = [
    "Show Marker", 
    "Get Scale", 
    "Color Correction"
    ]

# Methods
SCALE_METHODS = [
    "MARKER",
    "STICKER"
]

# Help Text
MARKER_SPILLOVER_HELP = "If selected, marker scale calculation method will be used on detected markers if all 4 tags on the astrobotany sticker are not detected."

METHOD_SELECT_HELP = """The 'STICKER' method calculates scale from the entire Astrobotany sticker. 
The 'MARKER' method calculates scale based only on the sticker markers
, which is useful if the Astrobotany is partially covered in the image."""

def main():

    st.title("Astrobotany Sticker Tools")
    st.write("Note: The current toolset supports images with only 1 marker in the image.")

    st.header("Image Upload")

    # Create tabs for upload images or using camera
    input1, input2 = st.tabs(["Upload Image(s)", "Use Camera"])
    with input1:
        user_image_upload = st.file_uploader("Picture to analyze:", type=["png","jpg","tiff", "tif","jpeg"], 
                                             accept_multiple_files=False, key="file_uploader")
    with input2:    
        user_image_camera = st.camera_input("Use Device Camera")

    # Give priority to uploaded images
    if (user_image_upload is not None):
        user_image = user_image_upload
    else:
        user_image = user_image_camera
    
    # Dropdown for desired function
    user_function = st.selectbox("Select Astrobotany Sticker Tool", 
                                 options=TOOL_OPTIONS)

    if (user_image is not None):
        # Load file into array using PIL
        img = Image.open(user_image)
        img = np.array(img)
        
        # Original image display
        st.header("Original Image")
        st.image(img)

        
        

        if TOOL_OPTIONS.index(user_function) == 0:

            st.header("Find Marker")
            st.write("Locates and displays markers in the given image.")

            st.subheader("Marker Found")
            final_image = asq.asq_show_marker(img)
            st.image(final_image)


        elif TOOL_OPTIONS.index(user_function) == 1:
            
            st.header("Get Scale")

            st.subheader("Scale Options")

            user_scale_method = st.selectbox("Scale Calculation Method", options=SCALE_METHODS)
            user_spillover = st.checkbox("Marker Spillover", value=True, help=MARKER_SPILLOVER_HELP)
            scale_val, unit = asq.asq_find_scale(img, user_scale_method, spillover=user_spillover)

            st.subheader("Calculated Scale")
            st.metric("Scale Value", value=f"{scale_val} pixels per meter")

        elif TOOL_OPTIONS.index(user_function) == 2:
            
            st.header("Color Correct Image")

            final_image = asq.asq_color_correct(img)
            st.image(final_image)


            with st.expander("Show Difference in Images"):
                difference_image = img - final_image
                st.image(difference_image)
                st.write(np.mean(difference_image))
            


if __name__ == "__main__":
    main()