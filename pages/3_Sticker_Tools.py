import streamlit as st
import cv2
import numpy as np
from PIL import Image
from io import BytesIO
from func import airisquare as asq
import json

# Set page name
st.set_page_config(page_title="Astrobotany Sticker Tools")

# Overall tool options list
TOOL_OPTIONS = [
    "Show Marker", 
    "Get Scale", 
    "Color Correction",
    "Color Standard",
    "Adjust Frame"
    ]

# Methods
SCALE_METHODS = [
    "MARKER",
    "STICKER"
]

# Sticker Orientation Types
STICKER_DESTINATION = [
        "TOP_LEFT", "TOP_CENTER", "TOP_RIGHT",
        "CENTER_LEFT", "CENTER", "CENTER_RIGHT",
        "BOTTOM_LEFT", "BOTTOM_CENTER", "BOTTOM_RIGHT"
        ]
    

# Help Text
MARKER_SPILLOVER_HELP = "If selected, marker scale calculation method will be used on detected markers if all 4 tags on the astrobotany sticker are not detected."

METHOD_SELECT_HELP = """The 'STICKER' method calculates scale from the entire Astrobotany sticker. 
The 'MARKER' method calculates scale based only on the sticker markers
, which is useful if the Astrobotany is partially covered in the image."""

MARKER_ADJUST_PAD_HELP = """Pad value is the number of pixels offset from the image edge the marker will end up. 
This is automatically designated based on the Sticker Destination. If no pad is desired, leave as '0' integer."""

# Error case
def marker_error():
    st.error("Marker not found in image.")
    st.stop()

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

        # Instantiate final_image and standard_square
        final_image = None
        standard_square = None

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

            try:
                scale_val, unit = asq.asq_find_scale(img, user_scale_method, spillover=user_spillover)
            except:
                marker_error()

            st.subheader("Calculated Scale")
            st.metric("Scale Value", value=f"{scale_val} pixels per meter")

        elif TOOL_OPTIONS.index(user_function) == 2:
            
            st.header("Color Correct Image")

            try:
                final_image = asq.asq_color_correct(img)
            except:
                marker_error()

            st.image(final_image)


            with st.expander("Show Difference in Images"):
                difference_image = img - final_image
                st.image(difference_image)
                st.write(np.mean(difference_image))

        elif TOOL_OPTIONS.index(user_function) == 3:
            
            st.header("Get Color References")
            

            try:
                standard_square = asq.asq_color_standards(img, list_convert=True)
            except:
                marker_error()
        
            st.subheader("Color Standard Data")
            st.json(standard_square, expanded=False)
            
        elif TOOL_OPTIONS.index(user_function) == 4:

            st.header("Frame Adjust")

            # Collect User options
            user_sticker_destination = st.selectbox("Sticker Destination", STICKER_DESTINATION)
            user_sticker_rotation = st.selectbox("Sticker Rotation", [0,1,2,3], format_func=lambda x: x*90)
            user_sticker_pad = st.number_input("Pad Value", value=0, step=1, help=MARKER_ADJUST_PAD_HELP)

            final_image = asq.asq_adjust_image(img, rot=user_sticker_rotation, position=user_sticker_destination, pad_val=user_sticker_pad)
            
            st.subheader("Adjusted Image")
            st.image(final_image)

        # Download button for full sized image
        if (final_image is not None):
            user_image_format = st.selectbox("Image Download Format", options=["JPEG", "TIFF", "PNG"])

            # Make new name for corrected image download
            user_image_name_trim = user_image.name.split(".")[0]
            final_image_name = user_image_name_trim + f"_corrected.{user_image_format.lower()}"
            
            # Prepare image for download
            im_pil = Image.fromarray(final_image)
            buf = BytesIO()
            im_pil.save(buf, format=user_image_format)
            bytes_img = buf.getvalue()
            st.download_button(
                label="Download Full-Sized Image", 
                data=bytes_img, 
                file_name=final_image_name,
                mime="image/jpeg"
                )
            
        if (standard_square is not None):
            json_string = json.dumps(standard_square)
            user_image_name_trim = user_image.name.split(".")[0]
            st.download_button(
                label="Download Image Color Reference",
                file_name=f"{user_image_name_trim}_color_reference.json",
                mime="application/json",
                data=json_string
            )

if __name__ == "__main__":
    main()