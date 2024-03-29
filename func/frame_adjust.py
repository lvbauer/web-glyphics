import cv2
import math
import numpy as np
from statistics import mean, median, mode, pvariance, stdev


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

def keystone_correct_resize(rgb_img, src_points, dest_points, new_shape=None):
	"""Keystone correct image orientation from one list of points to another
	This version finds the maximums from the destination points and creates the final keystone image shape based on that.

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

	# Find new image array shape using maximum for each of first 2 dimensions for new image array
	if (new_shape == None):
		z = rgb_img.shape[2]
		x_vals, y_vals = list(zip(*dest_points))
		x_max = int(max(x_vals))
		y_max = int(max(y_vals))
		new_shape = (y_max, x_max, z)
	
	# Keystone correction functionality
	dstD = np.zeros(new_shape, dtype=np.uint8)
	H = cv2.findHomography(np.array(src_points,dtype=np.float32),np.array(dest_points,dtype=np.float32),cv2.LMEDS)
	dest_img=cv2.warpPerspective(rgb_img,H[0],(dstD.shape[1],dstD.shape[0]))
	return dest_img

def rotate_list(arr,d,n):
	"""Rotates list arr of length n by number of positions d
	"""
	arr=arr[:]
	arr=arr[d:n]+arr[0:d]
	return arr


def expand_correct_image(image, dictionary, card_id=0, normal_id=1, inset=0, rotation=0):
	"""
	Wholistic function to correct image to a expanded view
	"""
	
	# Load dictionary and detect markers
	arucoDict = cv2.aruco.Dictionary_get(dictionary)
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

def square_correct_image(image, dictionary, card_id=0, normal_id=1, rotation=0, inset=0):
	"""
	Wholistic function to correct image to a expanded view
	"""
	
	# Load dictionary and detect markers
	arucoDict = cv2.aruco.Dictionary_get(dictionary)
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

	# Landscape image
	if (x_len >= y_len):
		pad_length = (x_len - y_len) // 2

		dest_points = [
			(pad_length+inset,inset),
			(x_len-pad_length-inset,inset),
			(x_len-pad_length-inset,y_len-inset),
			(pad_length+inset,y_len-inset),
		]
	
	# Portrait image
	elif (x_len < y_len):
		pad_length = (y_len - x_len) // 2

		dest_points = [
			(inset,pad_length+inset),
			(x_len-inset,pad_length+inset),
			(x_len-inset,y_len-pad_length-inset),
			(inset,y_len-pad_length-inset),
		]
	
	# Keystone image and return
	return keystone_correct(image, order_point_list, dest_points)

def maintain_correct_image(image, dictionary, card_id=0, normal_id=1, inset=0, rotation=0):
	"""
	Wholistic function to correct image to a squared up verion of itself, maintaining original object size.
	"""
	
	# Load dictionary and detect markers
	arucoDict = cv2.aruco.Dictionary_get(dictionary)
	arucoParams = cv2.aruco.DetectorParameters_create()
	corners, ids, rejected = cv2.aruco.detectMarkers(image, arucoDict, parameters=arucoParams)

	# Get points from corners
	points = get_aruco_points(corners)
	
	# Order points (Find source points)
	order_point_list = order_aruco_clockwise(ids, points, card_id=card_id, normal_id=normal_id)
	
	# Calculate segment lengths
	# TODO replace with a function
	seg_length_list = []
	for i in range(len(order_point_list)):
		pt_tup = (order_point_list[i-1], order_point_list[i])
		calc_dist = math.dist(*pt_tup)
		seg_length_list.append((pt_tup, calc_dist))
	seg_length_list.sort(reverse=True, key = lambda x: x[1])

	# Find long sides
	longest_side = seg_length_list[0]
	opposite_side = [seg for seg in seg_length_list if len(set(longest_side[0]).intersection(set(seg[0]))) == 0][0]
	long_sides = [longest_side, opposite_side]

	# Find short sides
	short_sides = [seg for seg in seg_length_list if seg not in long_sides]

	# Find average side lengths
	long_avg = mean(tup[1] for tup in long_sides)
	short_avg = mean(tup[1] for tup in short_sides)

	# Rotate correction if specified
	if rotation > 0:
		order_point_list = rotate_list(order_point_list, rotation, len(order_point_list))
	
	# Calculate destination points
	y_len, x_len, z_len = image.shape

	if (x_len < y_len):
		# Portrait case
		y_buffer = (y_len - long_avg) // 2
		x_buffer = (x_len - short_avg) // 2

		# Calculate inset values
		inset_scale = long_avg / short_avg

		x_inset = int(inset * 1)
		y_inset = int(inset * inset_scale)
	
	else:
		# Landscape or square case
		y_buffer = (y_len - short_avg) // 2
		x_buffer = (x_len - long_avg) // 2

		# Calculate inset values
		inset_scale = long_avg / short_avg

		x_inset = int(inset * inset_scale)
		y_inset = int(inset * 1)

	dest_points = [
		(x_buffer+x_inset,y_buffer+y_inset),
		(x_len-x_buffer-x_inset,y_buffer+y_inset),
		(x_len-x_buffer-x_inset,y_len-y_buffer-y_inset),
		(x_buffer+x_inset,y_len-y_buffer-y_inset),
	]
	
	# Keystone image and return
	return keystone_correct(image, order_point_list, dest_points)

def maintain_expand_correct_image(image, dictionary, card_id=0, normal_id=1, inset=0, rotation=0, auto_inset=False):
	"""
	Wholistic function to correct image by moving its corner to images (0,0) and its
	"""
	
	# Load dictionary and detect markers
	arucoDict = cv2.aruco.Dictionary_get(dictionary)
	arucoParams = cv2.aruco.DetectorParameters_create()
	corners, ids, rejected = cv2.aruco.detectMarkers(image, arucoDict, parameters=arucoParams)

	# Get points from corners
	points = get_aruco_points(corners)
	
	# Order points (Find source points)
	order_point_list = order_aruco_clockwise(ids, points, card_id=card_id, normal_id=normal_id)
	
	# Calculate segment lengths
	# TODO replace with a function
	seg_length_list = []
	for i in range(len(order_point_list)):
		pt_tup = (order_point_list[i-1], order_point_list[i])
		calc_dist = math.dist(*pt_tup)
		seg_length_list.append((pt_tup, calc_dist))
	seg_length_list.sort(reverse=True, key = lambda x: x[1])

	# Find long sides
	longest_side = seg_length_list[0]
	opposite_side = [seg for seg in seg_length_list if len(set(longest_side[0]).intersection(set(seg[0]))) == 0][0]
	long_sides = [longest_side, opposite_side]

	# Find short sides
	short_sides = [seg for seg in seg_length_list if seg not in long_sides]

	# Find average side lengths
	long_avg = mean(tup[1] for tup in long_sides)
	short_avg = mean(tup[1] for tup in short_sides)

	# Rotate correction if specified
	if rotation > 0:
		order_point_list = rotate_list(order_point_list, rotation, len(order_point_list))
	
	# Calculate destination points
	y_len, x_len, z_len = image.shape

	if (x_len < y_len):
		# Portrait case
		y_buffer = (y_len - long_avg) // 2
		x_buffer = (x_len - short_avg) // 2

		# Calculate inset values
		inset_scale = long_avg / short_avg

		x_inset = int(inset * 1)
		y_inset = int(inset * inset_scale)
	
	else:
		# Landscape or square case
		y_buffer = (y_len - short_avg) // 2
		x_buffer = (x_len - long_avg) // 2

		# Calculate inset values
		inset_scale = long_avg / short_avg

		x_inset = int(inset * inset_scale)
		y_inset = int(inset * 1)

	dest_points = [
		(inset, inset),
		((long_avg * inset_scale)+inset, inset),
		((long_avg * inset_scale)+inset, y_len+inset),
		(inset, y_len+inset),
	]
	
	new_shape = (
		int(y_len+(inset*2)), 
		int((long_avg * inset_scale) + (inset*2)), 
		int(z_len)
		)

	# Keystone image and return
	return keystone_correct_resize(image, order_point_list, dest_points, new_shape=new_shape)

def maintain_expand_corner_correct_image(image, dictionary, card_id=0, normal_id=1, inset=0, rotation=0, auto_inset=False):
	"""
	Wholistic function to correct image by moving its corner to images (0,0) and expanding the bottom right corner to include a square for scale measurement.
	"""
	
	# Load dictionary and detect markers
	arucoDict = cv2.aruco.Dictionary_get(dictionary)
	arucoParams = cv2.aruco.DetectorParameters_create()
	corners, ids, rejected = cv2.aruco.detectMarkers(image, arucoDict, parameters=arucoParams)

	# Get points from corners
	points = get_aruco_points(corners)
	
	# Order points (Find source points)
	order_point_list = order_aruco_clockwise(ids, points, card_id=card_id, normal_id=normal_id)
	
	# Calculate segment lengths
	# TODO replace with a function
	seg_length_list = []
	for i in range(len(order_point_list)):
		pt_tup = (order_point_list[i-1], order_point_list[i])
		calc_dist = math.dist(*pt_tup)
		seg_length_list.append((pt_tup, calc_dist))
	seg_length_list.sort(reverse=True, key = lambda x: x[1])

	# Find long sides
	longest_side = seg_length_list[0]
	opposite_side = [seg for seg in seg_length_list if len(set(longest_side[0]).intersection(set(seg[0]))) == 0][0]
	long_sides = [longest_side, opposite_side]

	# Find short sides
	short_sides = [seg for seg in seg_length_list if seg not in long_sides]

	# Find average side lengths
	long_avg = mean(tup[1] for tup in long_sides)
	short_avg = mean(tup[1] for tup in short_sides)

	# Rotate correction if specified
	if rotation > 0:
		order_point_list = rotate_list(order_point_list, rotation, len(order_point_list))
	
	# Calculate destination points
	y_len, x_len, z_len = image.shape

	if (x_len < y_len):
		# Portrait case
		y_buffer = (y_len - long_avg) // 2
		x_buffer = (x_len - short_avg) // 2

		# Calculate inset values
		inset_scale = long_avg / short_avg

		x_inset = int(inset * 1)
		y_inset = int(inset * inset_scale)
	
	else:
		# Landscape or square case
		y_buffer = (y_len - short_avg) // 2
		x_buffer = (x_len - long_avg) // 2

		# Calculate inset values
		inset_scale = long_avg / short_avg

		x_inset = int(inset * inset_scale)
		y_inset = int(inset * 1)

	dest_points = [
		(0, 0),
		((long_avg * inset_scale), 0),
		((long_avg * inset_scale), y_len),
		(0, y_len),
	]
	
	new_shape = (
		int(y_len+(inset*2)), 
		int((long_avg * inset_scale) + (inset*2)), 
		int(z_len)
		)

	# Keystone image and return
	return keystone_correct_resize(image, order_point_list, dest_points, new_shape=new_shape)

def manual_correct_image(image, dictionary, card_id=0, normal_id=1, rotation=0, img_h=0, img_w=1024, subj_y=0, subj_x=0, subj_h=0, subj_w=0):

	# Load dictionary and detect markers
	arucoDict = cv2.aruco.Dictionary_get(dictionary)
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
	dest_points = [
		(subj_x, subj_y),
		(subj_x+subj_w, subj_y),
		(subj_x+subj_w, subj_y+subj_h),
		(subj_x, subj_y+subj_h)
	]

	# Perform homography correction
	dstD = np.zeros((img_h, img_w), dtype=np.uint8)
	H = cv2.findHomography(np.array(order_point_list,dtype=np.float32),np.array(dest_points,dtype=np.float32),cv2.LMEDS)
	dest_img=cv2.warpPerspective(image, H[0] , (dstD.shape[1],dstD.shape[0]))

	return dest_img


def get_scale(image, size=1, method="SEGMENTS_MEAN", dictionary=cv2.aruco.DICT_4X4_50, marker_ids=[0,1]):
	"""
	Gets scale from image based on CV sticker.
	"""

	# Load dictionary and detect markers
	arucoDict = cv2.aruco.Dictionary_get(dictionary)
	arucoParams = cv2.aruco.DetectorParameters_create()
	corners, ids, rejected = cv2.aruco.detectMarkers(image, arucoDict, parameters=arucoParams)

	# No markers in image
	if (len(corners) == 0):
		return None

	# Side segment based calculations
	if (method.upper() == "SEGMENTS_MEAN") or (method.upper() == "SEGMENTS_MEDIAN"):
		# Calculate scale based on user specified method
		# Calculate all segment lengths between points among all markers
		seg_length_list = []
		for idx, pts in enumerate(corners):

			if (ids[idx] not in marker_ids) and (marker_ids != []):
				continue

			working_seg_lens = []
			for i, pt in enumerate(pts):
				pt_tup = (tuple(pt[i-1].tolist()), tuple(pt[i].tolist()))
				working_seg_lens.append(math.dist(*pt_tup))
			seg_length_list.extend(working_seg_lens)
		
		# Choose statistic to handle segment lengths
		if (method.upper() == "SEGMENTS_MEAN"):
			scale = mean(seg_length_list)
		elif (method.upper() == "SEGMENTS_MEDIAN"):
			scale = median(seg_length_list)
	
	# Area based calculations
	elif (method.upper() == "AREA_MEAN") or (method.upper() == "AREA_MEDIAN"):
		# Use contourArea to generate list of contour areas, then do statistics on them
		
		marker_areas = []
		for idx, marker in enumerate(corners):
			if (ids[idx] not in marker_ids) and (marker_ids != []):
				continue
			marker_areas.append(cv2.contourArea(marker))
		marker_areas = map(math.sqrt, marker_areas)

		if (method.upper() == "AREA_MEAN"):
			scale = mean(marker_areas)
			
		elif (method.upper() == "AREA_MEDIAN"):
			scale = median(marker_areas)
		else:
			pass
	
	else:
		return None

	return scale / size

def calc_marker_range(ids, pts, card_id=0, normal_id=1, with_stats=True):
	"""Calculates the sizes of relevant detected markers and reports variance. 
	This gives a rough idea how much error is introduced by skew in the image.
	"""


	# associate ids with pts
	adj_ids_list = [card_id, normal_id]
	id_list = [id[0] for id in ids.tolist()]
	id_pt_list = list(zip(id_list, pts))

	# Verify id and calculate contour area
	contour_area_list = []
	for id, pt in id_pt_list:
		if (id in adj_ids_list):
			cnt_area = cv2.contourArea(pt)
			contour_area_list.append(cnt_area)

	if not with_stats:
		return contour_area_list

	else:
		stats_dict = dict()

		sorted_area_list = sorted(contour_area_list)
		stats_dict["min"] = float(sorted_area_list[0])
		stats_dict["max"] = float(sorted_area_list[-1])
		stats_dict["range"] = float(stats_dict["max"] - stats_dict["min"])

		stats_dict["mean"] = float(mean(sorted_area_list))
		stats_dict["median"] = float(median(sorted_area_list))
		#stats_dict["mode"] = float(mode(sorted_area_list))

		#stats_dict["variance"] = float(pvariance(sorted_area_list))
		stats_dict["stdev"] = float(stdev(sorted_area_list))
		stats_dict["corr_variation"] = stats_dict["stdev"] / stats_dict["mean"]

		stats_dict["rel_range_mean"] = float(stats_dict["range"] / stats_dict["mean"])
		stats_dict["rel_range_median"] = float(stats_dict["range"] / stats_dict["median"])


		return contour_area_list, stats_dict
	
def dict_delta(dict1, dict2, abs_val=False):
	"""Find the difference between 2 dictionaries."""

	diff_dict = dict()
	for key in dict1.keys():
		if key not in dict2:
			continue
		diff_dict[key] = dict2[key] - dict1[key]

	if abs_val:
		diff_dict = {key: abs(val) for key, val in diff_dict.items()}

	return diff_dict

def list_delta(l1, l2, abs_val=False):
	if abs_val:
		return [abs(a_i - b_i) for a_i, b_i in zip(l2, l1)]
	else:
		return [a_i - b_i for a_i, b_i in zip(l2, l1)]
	
def dict_delta_summary(dict1, dict2, abs_val=False):
	"""Find the difference between 2 dictionaries."""

	diff_dict = dict()
	for key in dict1.keys():
		if key not in dict2:
			continue
		diff_dict[key] = (dict1[key], dict2[key], dict2[key] - dict1[key])

	if abs_val:
		diff_dict = {key: list(map(abs, val)) for key, val in diff_dict.items()}

	return diff_dict