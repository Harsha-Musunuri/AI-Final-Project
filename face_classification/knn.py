import numpy as np
from PIL import Image
import sys
import argparse
from collections import Counter
from random import shuffle
from timeit import default_timer as timer
np.set_printoptions(threshold=sys.maxsize)

def resize_image(curr_image, resize_width, resize_height):
	im = Image.fromarray((curr_image).astype(np.uint8))
	resized_img = im.resize((resize_width, resize_width), Image.ANTIALIAS)
	curr_image = np.array(resized_img)
	return curr_image

def readImages(filename, number_of_data_points, resize_width, resize_height, indices):
	data_file = open(filename, "r")
	line = data_file.readline()
	line_num = 0
	image_array = []
	single_image_array = []
	indices_length = len(indices)

	while line:
		line = line.replace(" ","0").replace("#","1").replace("+","1").strip()
		line = list(map(int, line))

		if(1 in line):
			single_image_array.append(line)

		single_image_array.append(line)

		line_num += 1
		if(line_num % 70 == 0):
			arr = np.array(single_image_array)
			resized_img = resize_image(arr, resize_width, resize_height)
			image_array.append(resized_img)
			single_image_array = []
		line = data_file.readline()

	data_file.close()

	if len(indices) > 0:
		shuffled_image_array = []
		for value in indices:
			shuffled_image_array.append(image_array[value])
		return shuffled_image_array
	else:
		return image_array

def readLabels(filename, percentage):
	label_file = open(filename, "r")
	line = label_file.readline()
	labels = []

	shuffle_labels = [[]*2 for _ in range(0,2)]
	prior_count = [0]*2

	label_index = 0
	while line:
		label = int(line.strip())
		if(percentage != 100):
			shuffle_labels[label].append(label_index)
			prior_count[label] += 1
		label_index += 1
		labels.append(label)
		line = label_file.readline()
	label_file.close()

	indices = []
	if(percentage != 100):
		trim_labels = []
		for index in range(0,2):
			shuffle(shuffle_labels[index])
			shuffle_labels[index] = shuffle_labels[index][:int(len(shuffle_labels[index])*percentage/float(100))]
		for index in range(0,2):
			for label_index in shuffle_labels[index]:
				indices.append(label_index)
				trim_labels.append(labels[label_index])
		return trim_labels, indices
	return labels, indices

def most_frequent(list):
	occurence_count = Counter(list)
	return occurence_count.most_common(1)[0][0]

def top_euclidean_distance(training_images, training_labels, test_sample, num_neighbours):
	euclidean_dist = {}
	for index in range(0, len(training_labels)):
		train_sample = training_images[index]
		curr_dist = np.sum((train_sample - test_sample)**2)**0.5
		if (curr_dist in euclidean_dist.keys()):
			euclidean_dist[curr_dist].append(training_labels[index])
		else:
			euclidean_dist[curr_dist] = [training_labels[index]]
	top_k = []
	counter = 0
	for key in sorted(euclidean_dist):
		for element in euclidean_dist[key]:
			top_k.append(element)
			counter += 1
			if(counter == num_neighbours):
				break
		if(counter == num_neighbours):
			break
	return most_frequent(top_k)

def calc_accuracy(training_images, training_labels, testing_images, testing_labels, num_classes, num_neighbours):
	true_positive_count = 0
	for index in range(0, len(testing_labels)):
		print ("Processing Image %d/%d" %(index+1,len(testing_labels)) ,end="\r")
		curr_image = testing_images[index]
		curr_ground_truth = testing_labels[index]
		top_match = top_euclidean_distance(training_images, training_labels, curr_image, num_neighbours)
		if(curr_ground_truth == top_match):
			true_positive_count += 1
	accuracy = true_positive_count*100/float(len(testing_labels))
	return accuracy

def main():
	parser = argparse.ArgumentParser(description='Digit Classification using Perceptron')
	parser.add_argument('--image_resize_width', required=True, help='Resize Width')
	parser.add_argument('--image_resize_height', required=True, help='Resize Height')
	parser.add_argument('--training_data_path', required=True, help='Path to training data')
	parser.add_argument('--training_label_path', required=True, help='Path to training data')
	parser.add_argument('--validation_data_path', required=True, help='Path to validation data')
	parser.add_argument('--validation_label_path', required=True, help='Path to validation data')
	parser.add_argument('--test_data_path', required=True, help='Path to Testing data')
	parser.add_argument('--test_label_path', required=True, help='Path to Testing data')
	parser.add_argument('--num_neighbours_start_limit', required=True, help='Number of epochs')
	parser.add_argument('--num_neighbours_end_limit', required=True, help='Number of epochs')
	parser.add_argument('--training_data_percentage', required=True, help='Percentage of Training Data')
	args = parser.parse_args()

	training_data_percentage = int(args.training_data_percentage)
	num_neighbours_start_limit = int(args.num_neighbours_start_limit)
	num_neighbours_end_limit = int(args.num_neighbours_end_limit)
	resize_width = int(args.image_resize_width)
	resize_height = int(args.image_resize_height)
	best_val_acc = -1
	best_k = num_neighbours_start_limit

	while (training_data_percentage <= 100):
		print("Training Percentage", training_data_percentage)
		if(training_data_percentage == 100):
			end_itr = 1
		else:
			end_itr = 10
		for iteration in range(0,end_itr):
			# print("Iteration ", iteration+1)
			training_labels, indices = readLabels(args.training_label_path, training_data_percentage)
			training_images = readImages(args.training_data_path, len(training_labels), resize_width, resize_height, indices)
			num_classes = len(set(training_labels))

			validation_labels, indices = readLabels(args.validation_label_path, 100)
			validation_images = readImages(args.validation_data_path, len(validation_labels), resize_width, resize_height, indices)

			testing_labels, indices = readLabels(args.test_label_path, 100)
			testing_images = readImages(args.test_data_path, len(testing_labels), resize_width, resize_height, indices)

			for num_neighbours in range(num_neighbours_start_limit, num_neighbours_end_limit+1):
				if(num_neighbours_start_limit != num_neighbours_end_limit):
					val_accuracy = calc_accuracy(training_images, training_labels, validation_images, validation_labels, num_classes, num_neighbours)
					if(val_accuracy > best_val_acc):
						best_val_acc = val_accuracy
						best_k = num_neighbours
					# print("Accuracy on Digit Classification Val-Set is %f with k val of %d" %(val_accuracy, num_neighbours))
			start = timer()
			test_accuracy = calc_accuracy(training_images, training_labels, testing_images, testing_labels, num_classes, num_neighbours)
			end = timer()
			print(end - start)
			print("Accuracy on Digit Classification Test-Set is %f with k val of %d" %(test_accuracy, num_neighbours))
		training_data_percentage += 10
		exit(1)

if __name__ == '__main__':
	main()
