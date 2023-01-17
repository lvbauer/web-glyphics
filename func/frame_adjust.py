import cv2
import math
import sys, os
import numpy as np


def get_aruco_points(marker_corners):
	"""
	Turn iterable of points into list of tuples of marker centers
	"""
	
	point_list = []
	
	for idx, marker in enumerate(marker_corners):
		
		corner_list = marker_corners[idx][0].tolist()
		x_sum = y_sum = 0
		
		for x_val, y_val in corner_list:
			x_sum += x_val
			y_sum += y_val
		
		point_list.append((int(x_sum*0.25), int(y_sum*0.25)))
		
	return point_list
		
			
def angle_between_points(p1, p2):
	"""
	Calculates clockwise angle between points in left-hand coordinate system
	Origin is the corner point
	"""
	
	d1 = p2[0] - p1[0]
	d2 = p2[1] - p1[1]
	if d1 == 0:
		if d2 == 0:  # same points?
			deg = 0
		else:
			deg = 0 if p1[1] > p2[1] else 180
	elif d2 == 0:
		deg = 90 if p1[0] < p2[0] else 270
	else:
		deg = math.atan(d2 / d1) / math.pi * 180
		lowering = p1[1] < p2[1]
		if (lowering and deg < 0) or (not lowering and deg > 0):
			deg += 270
		else:
			deg += 90
	return deg

def angle_points_centroid(p1, p2, centroid):
	"""Calculates angle p1-centroid-p2 in radians
	"""
	v0 = np.array(p1) - np.array(centroid)
	v1 = np.array(p2) - np.array(centroid)
	angle = np.math.atan2(np.linalg.det([v0,v1]),np.dot(v0,v1))
	return angle

def order_aruco_clockwise(ids, marker_centers, card_id=0, normal_id=1):
	"""Calculates a clockwise-ordered marker center points
	Returns list which starts with the cardinal market and progresses clockwise about marker centroid
	
	"""
	
	# Make main list
	point_id_list = list(zip(ids, marker_centers))
	
	# Get centroid
	centroid = []
	num_markers = len(marker_centers)
	for axis in range(2):
		axis_sum = sum([point[axis] for point in marker_centers])
		axis_mean = int(axis_sum//num_markers)
		centroid.append(axis_mean)
	centroid = tuple(centroid)

	# Get cardinal corner
	for marker_id, point in point_id_list:
		if (marker_id == card_id):
			card_id = marker_id
			card_pt = point
			break

	# Find point order
	angle_sort_list = sorted([point for marker_id, point in point_id_list if ((marker_id != card_id) and (marker_id == normal_id))], key=lambda x: angle_points_centroid(card_pt, centroid, x))
	angle_sort_list.insert(0, card_pt)
	return angle_sort_list
	


def keystone_correct(rgb_img, src_points, dest_points):
	"""Keystone correct image orientation from one list of points to another

	Args:
		rgb_img: NumPy array representing an image in RGB colorspace format
		src_points (list): list of tuples representing source points in original image, 
			format [(x1,y1),(x2,y2),...]
		dest_points (list): list of tuples representing destination points in relative to image, 
			format [(x1,y1),(x2,y2),...]

	Returns:
		dest_img: Keystone corrected image as numpy array in RGB colorspace

	"""

	# Check that lists are of matched length
	if len(src_points) != len(dest_points):
		raise Exception(f"Number of points in input lists not equal. src_points: {len(src_points)}, dest_points: {len(dest_points)}")

	# Keystone correction functionality
	dstD = np.zeros(rgb_img.shape,dtype=np.uint8)
	H = cv2.findHomography(np.array(src_points,dtype=np.float32),np.array(dest_points,dtype=np.float32),cv2.LMEDS)
	dest_img=cv2.warpPerspective(rgb_img,H[0],(dstD.shape[1],dstD.shape[0]))
	return dest_img

def rotate_list(arr,d,n):
	"""Rotates list arr of length n by number of positions d
	"""
	arr=arr[:]
	arr=arr[d:n]+arr[0:d]
	return arr


def expand_correct_image(image, card_id=0, normal_id=1, inset=0, rotation=0):
	"""
	Wholistic function to correct image to a expanded view
	"""
	
	# Load dictionary and detect markers
	arucoDict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_4X4_50)
	arucoParams = cv2.aruco.DetectorParameters_create()
	corners, ids, rejected = cv2.aruco.detectMarkers(image, arucoDict, parameters=arucoParams)

	# Get points from corners
	points = get_aruco_points(corners)
	
	# Order points (Find source points)
	order_point_list = order_aruco_clockwise(ids, points, card_id=card_id, normal_id=normal_id)
	
	# Rotate correction if specified
	if rotation > 0:
		order_point_list = rotate_list(order_point_list, rotation, len(order_point_list))
	
	# Calculate destination points
	y_len, x_len, z_len = image.shape
	
	dest_points = [
		(0+inset,0+inset),
		(x_len-inset,0+inset),
		(x_len-inset,y_len-inset),
		(0+inset,y_len-inset),
	]
	
	# Keystone image and return
	return keystone_correct(image, order_point_list, dest_points)

def square_correct_image(image, card_id=0, normal_id=1, rotation=0):
	"""
	Wholistic function to correct image to a expanded view
	"""
	
	# Load dictionary and detect markers
	arucoDict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_4X4_50)
	arucoParams = cv2.aruco.DetectorParameters_create()
	corners, ids, rejected = cv2.aruco.detectMarkers(image, arucoDict, parameters=arucoParams)

	# Get points from corners
	points = get_aruco_points(corners)
	
	# Order points (Find source points)
	order_point_list = order_aruco_clockwise(ids, points, card_id=card_id, normal_id=normal_id)
	
	# Rotate correction if specified
	if rotation > 0:
		order_point_list = rotate_list(order_point_list, rotation, len(order_point_list))
	
	# Calculate destination points
	y_len, x_len, z_len = image.shape
	pad_length = (x_len - y_len) // 2

	dest_points = [
		(pad_length,0),
		(x_len-pad_length,0),
		(x_len-pad_length,y_len),
		(pad_length,y_len),
	]
	
	# Keystone image and return
	return keystone_correct(image, order_point_list, dest_points)
