import math
import cv2
import numpy as np
from statistics import mean


def is_adjacent(p1, p2):
	"""
	Calculate adjacentcy between 2 n-dimensional points
	Adjacency calculated as a manhattan distance less than or equal to the number of dimensions
	"""
	x1,y1 = p1
	x2,y2 = p2
	return abs(x1-x2) + abs(y1-y2) >= len(p1)

def calc_error(val1, val2):
	"""Calculate the error between two values.
	"""
	if val1 == 0 or val2 == 0:
		return 1
	area = float((val2-val1)/val2)
	if area > 0:
		return area
	else:
		return 1

def score_contour(contour1, contour2):
	"""
	Calculates score for sorting using calc_error function
	Score = Area / Error
	"""
	area1 = cv2.contourArea(contour1)
	area2 = cv2.contourArea(contour2)
	return area1 / calc_error(area1, area2)

def find_short_sides(points_list):
	# Assumes 4 points
	points_list = points_list.tolist()
	pt_tup_lst = []

	# Good code for generating triangle comparisons
	for idx, i in enumerate(points_list):
		for j in points_list[idx+1:]:
			pt_tup_lst.append(tuple((i,j)))
	
	sort_list = sorted(pt_tup_lst, key=lambda tup: math.dist(tup[0],tup[1]))
	return sort_list[:2]

def get_midpoint(pt_tup):
	
	xmid = ((pt_tup[0][0] + pt_tup[1][0]) // 2)
	ymid = ((pt_tup[0][1] + pt_tup[1][1]) // 2)
	
	return xmid, ymid

def locate(rgb_image):
	"""
	Locates astrobotany square candidates in target RGB image, ranked best to worst

	Returns a list of contour tuples in (box_points, full_contour) form ranked from best candidate to worst.
	"""

	# Get channel "A" from LAB
	img_lab = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2LAB)
	channel_a = img_lab[:,:,1]

	# Gaussian Blur with Edge Detect
	channel_a_blur = cv2.GaussianBlur(channel_a, (5,5) ,cv2.BORDER_DEFAULT)
	edge_image = cv2.Canny(channel_a_blur, 26, 51)

	# Find contours in edge detect image
	contours = cv2.findContours(edge_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

	# Calculate minimum area rectangles for rectangle detection
	min_box_list = []
	for cntr in contours[0]:
		rect = cv2.minAreaRect(cntr)
		box = cv2.boxPoints(rect)
		box = np.int0(box)
		min_box_list.append(box)

	# Sort the list by the metric defined in score_contour
	contour_list = list(zip(contours[0], min_box_list))
	contour_sort_calc = sorted(contour_list, reverse=True,
		key=lambda x: score_contour(x[0], x[1]))

	return contour_sort_calc

def validate(contour_list):
	"""Initial version of validation of Astrobotany stickers using contour approximation
	"""

	margin = 0.05
	true_aspect_ratio = 1.184

	valid_list = []
	for contour, box_pts in contour_list:
		
		# Approximate the contour
		peri = cv2.arcLength(contour, True)
		approx = cv2.approxPolyDP(contour, 0.04 * peri, True)

		# Find contour side lengths if is rectangle
		if (len(approx) == 4):
			dist_list = []
			for i in range(4):
				point_distance = math.dist(approx[i-1][0], approx[i][0])
				dist_list.append(point_distance)

			# Find aspect ratio
			dist_list.sort()
			short_side_avg = mean(dist_list[:2])
			long_side_avg = mean(dist_list[2:])
			aspect_ratio = long_side_avg / short_side_avg

			# Filter based on perimeter
			if (sum(dist_list) < 100):
				pass

			elif (aspect_ratio < (true_aspect_ratio * (1+margin))) and (aspect_ratio > (true_aspect_ratio * (1-margin))):
				valid_list.append((contour, approx))

		else:
			continue

	return valid_list



