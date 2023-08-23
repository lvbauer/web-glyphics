import streamlit as st
import numpy as np
import cv2
import statistics as stat
import math

# Marker Values
TOP_LEFT = 48
TOP_RIGHT = 49
BOTTOM_LEFT = 47
BOTTOM_RIGHT = 46

# ArUco Marker Side Length
MARKER_LENGTH_VALUE = 0.008
MARKER_LENGTH_UNIT = "Meter"

# Astrobotany Sticker Side Length
STICKER_LONG_SIDE_LENGTH = 0.04526
STICKER_SHORT_SIDE_WIDTH = 0.03658
STICKER_SIDE_UNIT = "Meter"

# Methods
SCALE_METHODS = [
    "MARKER",
    "STICKER"
]

# Square Destination Size
MARKER_HEIGHT = 208
MARKER_WIDTH = 176

def get_validate_square_ids(rgb_image):
    """Finds CV tag Astrosquare sticker in an image and returns list of corner points for the sticker.
    Note: Only works when 1 sticker is present in the image
    TODO Add handling for multiple sticker in image
    """

    # Prep list of corner markers of square
    marker_id_list_reference = [TOP_LEFT, TOP_RIGHT, BOTTOM_RIGHT, BOTTOM_LEFT]

    # Load dictionary and detect markers
    arucoDict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_4X4_50)
    arucoParams = cv2.aruco.DetectorParameters_create()
    corners, ids, rejected = cv2.aruco.detectMarkers(rgb_image, arucoDict, parameters=arucoParams)

    # Find markers belonging to Astrobotany Square
    marker_cord_list = []
    marker_id_list = []

    marker_id_corner_list = list(zip(ids, corners))
    marker_id_corner_list = sorted(marker_id_corner_list, key=lambda x : x[0])


    for id, marker in marker_id_corner_list:

        if id in marker_id_list_reference:
            marker_cord_list.append(marker)
            marker_id_list.append(id[0])

    return marker_cord_list, marker_id_list

def get_aruco_points(marker_corners):
    """
    Turn iterable of points into list of tuples of marker centers
    """

    point_list = []
           
    for marker in marker_corners:
		
        corner_list = marker[0].tolist()
        x_sum = y_sum = 0
		
        for x_val, y_val in corner_list:
            x_sum += x_val
            y_sum += y_val
		
        point_centroid_tuple = (int(x_sum*0.25), int(y_sum*0.25))
        point_list.append(point_centroid_tuple)

    return point_list

def get_validate_square(rgb_image):
    """Finds CV tag Astrosquare sticker in an image and returns list of corner points for the sticker.
    Note: Only works when 1 sticker is present in the image
    TODO Add handling for multiple sticker in image
    """

    # Prep list of corner markers of square
    marker_id_list = [TOP_LEFT, TOP_RIGHT, BOTTOM_RIGHT, BOTTOM_LEFT]

    # Load dictionary and detect markers
    arucoDict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_4X4_50)
    arucoParams = cv2.aruco.DetectorParameters_create()
    corners, ids, rejected = cv2.aruco.detectMarkers(rgb_image, arucoDict, parameters=arucoParams)

    # Find markers belonging to Astrobotany Square
    marker_cord_list = []

    marker_id_corner_list = list(zip(ids, corners))
    marker_id_corner_list = sorted(marker_id_corner_list, key=lambda x : x[0])

    for id, marker in marker_id_corner_list:
        if id in marker_id_list:
            marker_cord_list.append(marker)

    return marker_cord_list

def get_marker_scale(marker_pt_list):
    """Returns a scale based on size of computer vision markers on Astrobotany Sticker.
    Uses a square root method for finding scale.
    """
    contour_areas = [cv2.contourArea(cnt) for cnt in marker_pt_list]
    mean_area = stat.mean(contour_areas)
    calculated_scale = math.sqrt(mean_area) / (MARKER_LENGTH_VALUE) 
    return calculated_scale, MARKER_LENGTH_UNIT

def get_sticker_scale(marker_pt_list):
    """Returns a scale based on entire Astrobotany sticker size."""

    # Define short and long side IDs
    id_order = sorted([TOP_LEFT, TOP_RIGHT, BOTTOM_LEFT, BOTTOM_RIGHT])
    short_sides = [(TOP_LEFT,TOP_RIGHT), (BOTTOM_LEFT,BOTTOM_RIGHT)]
    long_sides = [(TOP_LEFT,BOTTOM_LEFT), (TOP_RIGHT,BOTTOM_RIGHT)]

    # Get marker centroids and make marker dict
    marker_centroid_list = get_aruco_points(marker_pt_list)
    marker_dict = {id : marker_centroid_list[idx] for idx, id in enumerate(id_order)}

    # Get side lengths
    long_side_lengths = get_lengths(long_sides, marker_dict)
    short_side_lengths = get_lengths(short_sides, marker_dict)

    # Find side averages
    mean_long_length = stat.mean(long_side_lengths)
    mean_short_length = stat.mean(short_side_lengths)

    # Adjust by coefficients
    adj_side_lengths = stat.mean([mean_long_length / STICKER_LONG_SIDE_LENGTH, mean_short_length / STICKER_SHORT_SIDE_WIDTH])

    return adj_side_lengths, STICKER_SIDE_UNIT

def get_lengths(sides, marker_dict):
    """Calculate side lengths for all sets of sides
    """
    side_list = []
    for vert1, vert2 in sides:
        side_len = math.dist(marker_dict[vert1], marker_dict[vert2])
        side_list.append(side_len)
    return side_list

def channel_hist_correct(corr_channel, standard_channel):
    hmax = 255
    data_type = np.uint8

    img_copy = np.copy(corr_channel)
    
    hist, bins = np.histogram(standard_channel, bins='auto')
    max1 = np.amax(bins)
    alpha = hmax / float(max1)
    corrected = np.asarray(np.where(img_copy <= max1, np.multiply(alpha, img_copy), hmax), data_type)

    return corrected


def asq_hist_correct(img, astrosquare):
    # Modified from the PlantCV implementation

    hmax = 255
    data_type = np.uint8

    c1, r1 = img[:,:,0], astrosquare[:,:,0]
    c2, r2 = img[:,:,1], astrosquare[:,:,1]
    c3, r3 = img[:,:,2], astrosquare[:,:,2]

    channel1 = channel_hist_correct(c1, r1)
    channel2 = channel_hist_correct(c2, r2)
    channel3 = channel_hist_correct(c3, r3)
    
    corrected = np.dstack((channel1, channel2, channel3))

    return corrected

def asq_color_correct(img):
    """Main function for histogram color correction.
    """
    
    # Get marker points
    try:
        marker_pt_list = get_validate_square(img)
    except TypeError:
        st.error("Marker not found in image.")
        return img    

    if (len(marker_pt_list) < 4):
        st.error(f"Marker not found in image.")
        return img

    marker_pt_list = get_aruco_points(marker_pt_list)

    dest_pts = [
         [18, 18],
         [18, 157],
         [190, 157],
         [190, 18]
    ]

    # (X,Y) to (Y,X)
    dest_pts_correct = [[pt[1], pt[0]] for pt in dest_pts]

    # Make array to warp into
    marker_arr = np.zeros((MARKER_HEIGHT, MARKER_WIDTH, 3), dtype=np.uint8)

    marker_pt_array = np.array(marker_pt_list, dtype=np.float32)
    dest_pt_array = np.array(dest_pts_correct, dtype=np.float32)

    H = cv2.findHomography(marker_pt_array, dest_pt_array, cv2.LMEDS)
    marker_stretch_img = cv2.warpPerspective(img, H[0], (marker_arr.shape[1], marker_arr.shape[0]))

    final_image = asq_hist_correct(img, marker_stretch_img)
    
    # Return RGB Image
    return final_image

def asq_find_scale(img, method, spillover=True):
    """Main function for finding scale.
    """

    # Slice image array to specified crop values
    try:
        marker_points_list = get_validate_square(img)
    except TypeError:
        return None

    if (len(marker_points_list) < 4):
        return None

    if (method == "MARKER") or ((len(marker_points_list) < 4) and (spillover == True)):
        scale_val, unit = get_marker_scale(marker_points_list)
    else:
        scale_val, unit = get_sticker_scale(marker_points_list)

    return scale_val, unit

def asq_show_marker(img):
    """Main function for showing detected markers."""
    
    # Find square markers in image
    try:
        marker_cords, marker_ids = get_validate_square_ids(img)
    except TypeError:
        st.error("Marker not found in image.")
        return img

    # Check if marker was found
    if (len(marker_cords) < 4):
        st.error("Marker not found in image.")
        return img

    # Convert marker ids to array
    marker_ids = np.asarray(marker_ids)

    # Make image copy
    marker_img = np.copy(img)

    # Draw marker on image
    cv2.aruco.drawDetectedMarkers(marker_img, marker_cords, marker_ids)

    # Get center points of markers
    marker_points = get_aruco_points(marker_cords)

    # Draw Marker Box
    circle_size = stat.mean((img.shape[0], img.shape[1])) // 100
    for idx, p in enumerate(marker_points):
        cv2.line(marker_img, p, marker_points[(idx+1)%(len(marker_points))], (255,0,0), int(circle_size//2))

    return marker_img
