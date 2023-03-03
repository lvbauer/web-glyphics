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

METHODS = [
    "MAINTAIN",
    "MAINTAIN_EXPAND",
    "MAINTAIN_EXPAND_CORNER",
    "RECTANGLE",
    "SQUARE"
]

SCALE_OPTIONS = [
    "SEGMENTS_MEAN",
    "SEGMENTS_MEDIAN",
    "AREA_MEAN",
    "AREA_MEDIAN"
]

def main():
    st.title("Adjust Image")

    st.header("Image Upload")
    
    # Create tabs for upload images or using camera
    input1, input2 = st.tabs(["Upload Image", "Use Camera"])
    with input1:
        user_image_upload = st.file_uploader("Picture to analyze:", type=["png","jpg","tiff", "tif","jpeg"], accept_multiple_files=False, key="file_uploader")
    with input2:    
        user_image_camera = st.camera_input("Use Device Camera")

    # Give priority to uploaded images
    if (user_image_upload is not None):
        user_image = user_image_upload
    else:
        user_image = user_image_camera    
        
    st.subheader("Adjustment Settings")
    # User inputs
    user_dictionary = st.selectbox("Choose marker dictionary:", ARUCO_DICT.keys())
    user_correction_method = st.selectbox("Choose correction method:", METHODS)

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
        user_rotation_value = st.selectbox("Choose counterclockwise rotation value:", options=[0,1,2,3,4], format_func=lambda x : x*90)
        user_inset_value = st.number_input("Choose inset: (px)", step=1)
    
    user_auto_inset_value = st.checkbox("Use Auto Inset Correction", help="Automatically calculates an inset which includes the full markers in the final image for use in finding scale. "
        "NOTE: Overrides user inset input when selected.")

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

        # Calculate inset if auto-inset is selected
        if user_auto_inset_value:
            
            # Calculate scale based on markers
            img_scale = adj.get_scale(img, size=1, method="SEGMENTS_MEDIAN", 
                marker_ids=[user_cardinal_pt, user_other_pt], dictionary=ARUCO_DICT[user_dictionary])
            
            # Auto-inset = scale_value * AUTO_INSET_ADJ
            AUTO_INSET_ADJ = 1.5
            user_inset_value = int(img_scale * AUTO_INSET_ADJ)

            st.write("Scale Value Calculated from Auto-Inset Calculation:")
            st.code(img_scale)

        # Apply transformation to the image
        if user_correction_method.upper() == "RECTANGLE":
            adj_img = adj.expand_correct_image(img, ARUCO_DICT[user_dictionary], card_id=user_cardinal_pt, normal_id=user_other_pt, rotation=user_rotation_value, inset=user_inset_value)
        elif user_correction_method.upper() == "SQUARE":
            adj_img = adj.square_correct_image(img, ARUCO_DICT[user_dictionary], card_id=user_cardinal_pt, normal_id=user_other_pt, rotation=user_rotation_value, inset=user_inset_value)
        elif user_correction_method.upper() == "MAINTAIN":
            adj_img = adj.maintain_correct_image(img, ARUCO_DICT[user_dictionary], card_id=user_cardinal_pt, normal_id=user_other_pt, rotation=user_rotation_value, inset=user_inset_value)
        elif user_correction_method.upper() == "MAINTAIN_EXPAND":
            adj_img = adj.maintain_expand_correct_image(img, ARUCO_DICT[user_dictionary], card_id=user_cardinal_pt, normal_id=user_other_pt, rotation=user_rotation_value, inset=user_inset_value)
        elif user_correction_method.upper() == "MAINTAIN_EXPAND_CORNER":
            adj_img = adj.maintain_expand_corner_correct_image(img, ARUCO_DICT[user_dictionary], card_id=user_cardinal_pt, normal_id=user_other_pt, rotation=user_rotation_value, inset=user_inset_value)

        # Display adjusted image
        st.header("Adjusted Image")
        final_rot = st.selectbox("Choose final rotation:", options=[0,1,2,3,4], format_func=lambda x : x*90)
        adj_img = np.rot90(adj_img, k=final_rot)

        st.image(adj_img)

        # Prompt for scale finding
        st.header("Image Scale Calculation")
        user_scale_method = st.selectbox("Select Scale Calculation Method", options=SCALE_OPTIONS)
        user_scale_value = st.number_input("Scale Value", min_value=0, value=1)

        img_scale = adj.get_scale(adj_img, size=user_scale_value, method=user_scale_method, 
            marker_ids=[user_cardinal_pt, user_other_pt], dictionary=ARUCO_DICT[user_dictionary])
        
        st.subheader("Scale")

        if (img_scale is None):
            st.warning("No Scale Calculated. To calculate scale, make sure markers are readable in final image.")
        else:
            st.metric("Scale (Pixels per unit)", round(img_scale, 2))

if __name__ == '__main__':
    main()