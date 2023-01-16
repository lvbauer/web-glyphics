import streamlit as st
import cv2
import numpy as np
from PIL import Image
from func import astrosquare as asq


def main():

    st.title("Adjust Image")

    st.header("Image Upload")
    user_image = st.file_uploader("Picture to analyze:", type=["png","jpg","tiff", "tif","jpeg"], accept_multiple_files=False, key="file_uploader")

    if (user_image is not None):
        # Load file into array using PIL
        img = Image.open(user_image)
        img = np.array(img)
        
        # Original image display
        st.header("Original Image")
        st.image(img)

        # Detect astrobotany square
        locate_output = asq.locate(img)
        validate_output = asq.validate(locate_output)

        # Draw markers and show annotated image
        img_copy = np.copy(img)
        pt_output = [tup[1] for tup in validate_output]
        pt_output = np.asarray(pt_output, dtype=np.float32)
        cv2.aruco.drawDetectedMarkers(img_copy, pt_output)

        # Annotated image display
        st.header("Validated Squares Image")
        st.image(img_copy)

        # Provide additional information in tabs
        st.header("Additional Information")
        tab1, tab2 = st.tabs(["Validation Output", "Location Output"])
        
        with tab1:
            st.subheader("Squares Validated:")
            st.code(pt_output)
        
        with tab2:
            st.subheader("Located Squares:")
            st.code(locate_output[:5])

if __name__ == "__main__":
    main()