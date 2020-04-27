import sys
import argparse
import numpy as np
from PIL import Image

def readImages(filename, number_of_data_points):
	data_file = open(filename, "r")
	line = data_file.readline()
	curr_image = np.empty([0, 0])
	previous_line = ""
	image_array = []
	counter = 0

	while line:
		curr_line = line.strip()
		individual_pixels = []
		individual_pixels[:0] = line.replace(" ","0").replace("+","1").replace("#","1").strip()
		if(curr_line != "" and previous_line == ""):
			counter += 1
			if(counter != 1):
				image_array.append(curr_image)
			curr_image = np.array(list(map(int, individual_pixels)),ndmin=2)
		elif(curr_line != "" and previous_line != ""):
			arr = np.array(list(map(int, individual_pixels)),ndmin=2)
			curr_image = np.append(curr_image, arr, axis=0)
		elif(counter == number_of_data_points and curr_line == "" and previous_line == ""):
			image_array.append(curr_image)
			counter += 1

		previous_line = line.strip()
		line = data_file.readline()

	data_file.close()

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

def main():
	parser = argparse.ArgumentParser(description='Digit Classification using Perceptron')
	parser.add_argument('--training_data_path', required=True, help='Path to training data')
	parser.add_argument('--training_label_path', required=True, help='Path to training data')
	parser.add_argument('--validation_data_path', required=True, help='Path to validation data')
	parser.add_argument('--validation_label_path', required=True, help='Path to validation data')
	parser.add_argument('--test_data_path', required=True, help='Path to Testing data')
	parser.add_argument('--test_label_path', required=True, help='Path to Testing data')
	args = parser.parse_args()

	training_labels = readLabels(args.training_label_path)
	training_images = readImages(args.training_data_path, len(training_labels))

	validation_labels = readLabels(args.validation_label_path)
	validation_images = readImages(args.validation_data_path, len(validation_labels))

	testing_labels = readLabels(args.test_label_path)
	testing_images = readImages(args.test_data_path, len(testing_labels))


if __name__ == '__main__':
	main()
