import numpy as np
from PIL import Image

def resize_image(curr_image, resize_width, resize_height):
	im = Image.fromarray((curr_image).astype(np.uint8))
	resized_img = im.resize((resize_width, resize_height), Image.ANTIALIAS)
	curr_image = np.array(resized_img)
	return curr_image

def readImages(filename, number_of_data_points, resize_width, resize_height, indices):
	data_file = open(filename, "r")
	line = data_file.readline()
	line_num = 0
	image_array = []
	single_image_array = []
	image_num = 0
	indices_length = len(indices)

	while line:
		line = line.replace(" ","0").replace("#","1").replace("+","1").strip()
		line = list(map(int, line)) 

		if(1 in line):
			single_image_array.append(line)

		single_image_array.append(line)

		line_num += 1
		if(line_num % 28 == 0):
			if((indices_length > 0 and image_num in indices) or indices_length == 0):
				arr = np.array(single_image_array)
				resized_img = resize_image(arr, resize_width, resize_height)
				image_array.append(resized_img)
			image_num += 1
			single_image_array = []
		line = data_file.readline()

	data_file.close()
	return image_array

def readLabels(filename, percentage):
	label_file = open(filename, "r")
	line = label_file.readline()
	labels = []
	while line:
		label = int(line.strip())
		labels.append(label)
		line = label_file.readline()
	label_file.close()

	indices = []
	if(percentage != 100):
		prior_count = [0]*len(list(set(labels)))
		for index in range(0,len(labels)):
			prior_count[labels[index]] += 1

		running_count = [0]*len(list(set(labels)))

		label_file = open(filename, "r")
		line = label_file.readline()
		trim_labels = []
		line_num = 0
		while line:
			label = int(line.strip())
			if(running_count[label] <= (prior_count[label]*percentage/float(100))):
				running_count[label] += 1
				trim_labels.append(label)
				indices.append(line_num)
			line_num += 1
			line = label_file.readline()
		label_file.close()
		return trim_labels, indices

	return labels, indices
