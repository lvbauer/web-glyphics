import streamlit as st
import cv2
import numpy as np
from PIL import Image
from func import frame_adjust as adj

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
    st.title("Adjust Image")

    st.header("Image Upload")
    user_image = st.file_uploader("Picture to analyze:", type=["png","jpg","tiff", "tif","jpeg"], accept_multiple_files=False, key="file_uploader")
    user_dictionary = st.selectbox("Choose marker dictionary:", ARUCO_DICT.keys())

    # Calculate number of 
    dictionary_num_value = user_dictionary.split("_")[-1]
    if (dictionary_num_value.isdigit()):
        dictionary_num_value = int(dictionary_num_value)
    elif (dictionary_num_value.upper() == "ORIGINAL"):
        dictionary_num_value = 1024
    else:
        st.error(f"Dictionary {user_dictionary} is not supported. Please choose ArUco marker dictionary.")
        st.stop()

    col1, col2 = st.columns(2)

    with col1:
        user_cardinal_pt = st.number_input("Choose ID number of cardinal point:", min_value=0, max_value=dictionary_num_value-1, step=1, value=0)
        user_other_pt = st.number_input("Choose ID number of other points:", min_value=0, max_value=dictionary_num_value-1, step=1, value=1)
    
    with col2:
        user_rotation_value = st.selectbox("Choose rotation value:", options=[0,1,2,3,4], format_func=lambda x : x*90)
        user_inset_value = st.number_input("Choose inset: (px)", min_value=0, max_value=1500, step=1)

    if (user_cardinal_pt == user_other_pt):
        st.warning("Point designations cannot be equal. Choose different values for marker ID.")
        st.stop()

    if (user_image is not None):
        
        # Load file into array using PIL
        img = Image.open(user_image)
        img = np.array(img)

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
        if (ids is None):
            st.warning(f"No markers detected in '{user_image.name}'.")
            st.stop()

        # Apply transformation to the image
        st.header("Adjusted Image")
        adj_img = adj.expand_correct_image(img, card_id=user_cardinal_pt, normal_id=user_other_pt, rotation=user_rotation_value, inset=user_inset_value)
        st.image(adj_img)

if __name__ == '__main__':
    main()