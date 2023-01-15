import streamlit as st
import cv2
import numpy as np
from PIL import Image

# define names of each possible ArUco tag OpenCV supports
ARUCO_DICT = {
	"DICT_4X4_50": cv2.aruco.DICT_4X4_50,
	"DICT_4X4_100": cv2.aruco.DICT_4X4_100,
	"DICT_4X4_250": cv2.aruco.DICT_4X4_250,
	"DICT_4X4_1000": cv2.aruco.DICT_4X4_1000,
	"DICT_5X5_50": cv2.aruco.DICT_5X5_50,
	"DICT_5X5_100": cv2.aruco.DICT_5X5_100,
	"DICT_5X5_250": cv2.aruco.DICT_5X5_250,
	"DICT_5X5_1000": cv2.aruco.DICT_5X5_1000,
	"DICT_6X6_50": cv2.aruco.DICT_6X6_50,
	"DICT_6X6_100": cv2.aruco.DICT_6X6_100,
	"DICT_6X6_250": cv2.aruco.DICT_6X6_250,
	"DICT_6X6_1000": cv2.aruco.DICT_6X6_1000,
	"DICT_7X7_50": cv2.aruco.DICT_7X7_50,
	"DICT_7X7_100": cv2.aruco.DICT_7X7_100,
	"DICT_7X7_250": cv2.aruco.DICT_7X7_250,
	"DICT_7X7_1000": cv2.aruco.DICT_7X7_1000,
	"DICT_ARUCO_ORIGINAL": cv2.aruco.DICT_ARUCO_ORIGINAL,
	"DICT_APRILTAG_16h5": cv2.aruco.DICT_APRILTAG_16h5,
	"DICT_APRILTAG_25h9": cv2.aruco.DICT_APRILTAG_25h9,
	"DICT_APRILTAG_36h10": cv2.aruco.DICT_APRILTAG_36h10,
	"DICT_APRILTAG_36h11": cv2.aruco.DICT_APRILTAG_36h11
}

def main():

    st.title(":mag_right: Visualize Markers")

    st.header("Image Upload")
    user_image = st.file_uploader("Picture to analyze:", type=["png","jpg","tiff", "tif","jpeg"], accept_multiple_files=False, key="file_uploader")
    user_dictionary = st.selectbox("Choose marker dictionary:", ARUCO_DICT.keys())


    if (user_image is not None):
        
        # Load file into array using PIL
        img = Image.open(user_image)
        img = np.array(img)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Original image display
        st.header("Original Image")
        st.image(img)

        # Detect markers in user image
        active_dict = cv2.aruco.Dictionary_get(ARUCO_DICT[user_dictionary])
        aruco_params = cv2.aruco.DetectorParameters_create()
        corners, ids, rejected_points = cv2.aruco.detectMarkers(img, active_dict, parameters=aruco_params)


        # Draw markers and show annotated image
        img_copy = np.copy(img)
        cv2.aruco.drawDetectedMarkers(img_copy, corners, ids)

        # Annotated image display
        st.header("Annotated Image")
        st.image(img_copy)

        if (ids == None):
            st.warning(f"No markers detected in '{user_image.name}'.")
            st.stop()

        # Provide additional information in tabs
        st.header("Additional Information")
        tab1, tab2, tab3 = st.tabs(["IDs", "Corner Values", "Rejected Markers"])
        
        with tab1:
            st.subheader("Marker IDs Found:")
            st.code(ids)
        
        with tab2:
            st.subheader("Marker Corner Points:")
            st.code(corners)

        with tab3:
            st.subheader("Rejected Points:")
            rejected_img = np.copy(img)
            cv2.aruco.drawDetectedMarkers(rejected_img, rejected_points)
            st.image(rejected_img)

if __name__ == "__main__":
    main()
