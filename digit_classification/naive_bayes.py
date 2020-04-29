import sys
import argparse
import numpy as np
from PIL import Image
import time
import copy
import math
np.set_printoptions(threshold=sys.maxsize)

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

def calc_accuracy(feature_matrix, prior_prob, images, labels, num_classes):
	true_positive_count = 0
	for outer_index in range(0, len(labels)):
		curr_image = images[outer_index]
		curr_ground_truth = labels[outer_index]

		prob = [0]*num_classes
		for index in range(0,num_classes):
			training_samples_prob = 0
			prob[index] += prior_prob[index]
			for row in range(0,images[0].shape[0]):
				for col in range(0,images[0].shape[1]):
					if(curr_image[row][col] == 0):
						ind_prob = feature_matrix[index][0][row][col]
					elif(curr_image[row][col] == 1):
						ind_prob = feature_matrix[index][1][row][col]
					if(ind_prob == 0):
						ind_prob = 0.000000001
					prob[index] += math.log(ind_prob)
		if(curr_ground_truth == np.argmax(prob)):
			true_positive_count += 1

	accuracy = true_positive_count*100/float(len(images))
	return accuracy

def fill_feature_matrix(images, labels, num_classes, gt_count):
	dictionary = {}
	empty_list = [[0]*images[0].shape[0] for _ in range(images[0].shape[1])]
	inner_dict = {}
	inner_dict[0] = copy.deepcopy(empty_list)
	inner_dict[1] = copy.deepcopy(empty_list)

	for index in range(0, num_classes):
		dictionary[index] = copy.deepcopy(inner_dict)

	for index in range(0, len(labels)):
		summation_term = 1/float(gt_count[labels[index]])
		curr_image = images[index]
		for row in range(0,images[0].shape[0]):
			for col in range(0,images[0].shape[1]):
				if(curr_image[row][col] == 0):
					dictionary[labels[index]][0][row][col] += summation_term
				elif(curr_image[row][col] == 1):
					dictionary[labels[index]][1][row][col] += summation_term
	return dictionary

def core_naive_bayes(training_images, training_labels, validation_images, validation_labels, num_classes):
	gt_count = []
	prior_prob = []
	for index in range(0, num_classes):
		gt_count.append(0)
	for index in range(0,len(training_labels)):
		gt_count[training_labels[index]] += 1
	for index in range(0, num_classes):
		prior_prob.append(gt_count[index]/len(training_labels))
	feature_matrix = fill_feature_matrix(training_images, training_labels, num_classes, gt_count)
	return feature_matrix, prior_prob

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
	num_classes = len(set(training_labels))

	validation_labels = readLabels(args.validation_label_path)
	validation_images = readImages(args.validation_data_path, len(validation_labels))

	testing_labels = readLabels(args.test_label_path)
	testing_images = readImages(args.test_data_path, len(testing_labels))

	feature_matrix, prior_prob = core_naive_bayes(training_images, training_labels, validation_images, validation_labels, num_classes)
	test_accuracy = calc_accuracy(feature_matrix, prior_prob, testing_images, testing_labels, num_classes)
	print("Accuracy on Digit Classification Test-Set is %f" %(test_accuracy))

if __name__ == '__main__':
	main()