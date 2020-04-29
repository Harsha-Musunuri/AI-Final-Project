import numpy as np
from PIL import Image

def resize_image(curr_image):
	im = Image.fromarray((curr_image).astype(np.uint8))
	resized_img = im.resize((32, 32), Image.ANTIALIAS)
	curr_image = np.array(resized_img)
	return curr_image

def readImages(filename, number_of_data_points):
	data_file = open(filename, "r")
	line = data_file.readline()
	line_num = 0
	image_array = []
	single_image_array = []

	while line:
		line = line.replace(" ","0").replace("#","1").replace("+","1").strip()
		line = list(map(int, line)) 

		if(1 in line):
			single_image_array.append(line)

		single_image_array.append(line)

		line_num += 1
		if(line_num % 28 == 0):
			arr = np.array(single_image_array)
			resized_img = resize_image(arr)
			image_array.append(resized_img)
			single_image_array = []
		line = data_file.readline()

	data_file.close()
	return image_array

def readLabels(filename):
	label_file = open(filename, "r")
	line = label_file.readline()
	labels = []
	while line:
		label = int(line.strip())
		labels.append(label)
		line = label_file.readline()
	label_file.close()
	return labels
