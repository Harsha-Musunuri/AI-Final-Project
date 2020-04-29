import util
import sys
import argparse
import numpy as np
from collections import Counter 
np.set_printoptions(threshold=sys.maxsize)

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
	parser.add_argument('--training_data_path', required=True, help='Path to training data')
	parser.add_argument('--training_label_path', required=True, help='Path to training data')
	parser.add_argument('--validation_data_path', required=True, help='Path to validation data')
	parser.add_argument('--validation_label_path', required=True, help='Path to validation data')
	parser.add_argument('--test_data_path', required=True, help='Path to Testing data')
	parser.add_argument('--test_label_path', required=True, help='Path to Testing data')
	parser.add_argument('--num_neighbours', required=True, help='Number of epochs')
	args = parser.parse_args()

	num_neighbours = int(args.num_neighbours)

	training_labels = util.readLabels(args.training_label_path)
	training_images = util.readImages(args.training_data_path, len(training_labels))
	num_classes = len(set(training_labels))

	validation_labels = util.readLabels(args.validation_label_path)
	validation_images = util.readImages(args.validation_data_path, len(validation_labels))

	testing_labels = util.readLabels(args.test_label_path)
	testing_images = util.readImages(args.test_data_path, len(testing_labels))

	test_accuracy = calc_accuracy(training_images, training_labels, testing_images, testing_labels, num_classes, num_neighbours)
	print("Accuracy on Digit Classification Test-Set is %f" %(test_accuracy))

if __name__ == '__main__':
	main()
