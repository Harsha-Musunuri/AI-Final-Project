import sys
import argparse
import numpy as np
from PIL import Image
import time
np.set_printoptions(threshold=sys.maxsize)

def get_pixels_from_string(string):
	individual_pixels =list(map(int, string)) 
	individual_pixels = [elt * 255 for elt in individual_pixels]
	return individual_pixels

def resize_and_append_image(curr_image, image_array):
	im = Image.fromarray((curr_image).astype(np.uint8))
	resized_img = im.resize((32, 32), Image.ANTIALIAS)
	curr_image = np.array(resized_img)
	image_array.append(curr_image)
	return image_array

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
				image_array = resize_and_append_image(curr_image, image_array)

			individual_pixels = get_pixels_from_string(individual_pixels)
			curr_image = np.array(individual_pixels,ndmin=2)

		elif(curr_line != "" and previous_line != ""):
			individual_pixels = get_pixels_from_string(individual_pixels)
			arr = np.array(individual_pixels,ndmin=2)
			curr_image = np.append(curr_image, arr, axis=0)

		elif(counter == number_of_data_points and curr_line == "" and previous_line == ""):
			image_array = resize_and_append_image(curr_image, image_array)
			counter += 1

		previous_line = line.strip()
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

def calc_accuracy(weights, images, labels, num_classes):
	true_positive_count = 0
	for index in range(0,len(labels)):
		curr_image 			= images[index]
		curr_ground_truth	= labels[index]
		curr_image = (curr_image/255).flatten()
		features = np.append(curr_image,1)
		features = features.reshape(1,32*32+1)

		predictions = []

		for inner_index in range(0,num_classes):
			product = np.inner(features, weights[inner_index])
			predictions.append(product)

		prediction_with_highest_confidence = np.argmax(predictions)
		if(prediction_with_highest_confidence == curr_ground_truth):
			true_positive_count += 1

	accuracy = true_positive_count*100/float(len(images))
	return accuracy

def train_perceptron(training_images, training_labels, validation_images, validation_labels, num_classes, num_epochs):
	learning_rate = 0.001
	weights = np.random.rand(num_classes,32*32+1)
	# weights = np.random.uniform(-0.05,0.05,(num_classes,32*32+1))
	best_weights = weights
	best_accuracy = -1

	for epoch in range(0,num_epochs):
		print ("Running on Epoch %d/%d" %(epoch+1,num_epochs) ,end="\r")
		time.sleep(1)
		true_positive_count = 0
		for index in range(0,len(training_labels)):
			curr_image 			= training_images[index]
			curr_ground_truth	= training_labels[index]
			curr_image = (curr_image/255).flatten() #normalizing
			features = np.append(curr_image,1) #adding bias as the last element, considering each pixel as a feature
			features = features.reshape(1,32*32+1)

			predictions = []

			for inner_index in range(0,num_classes):
				product = np.inner(features, weights[inner_index])
				predictions.append(product)
				if(product >= 0) and inner_index != curr_ground_truth:
					weights[inner_index] = weights[inner_index] - (learning_rate * features)
				if(product < 0) and inner_index == curr_ground_truth:
					weights[inner_index] = weights[inner_index] + (learning_rate * features)
			
			prediction_with_highest_confidence = np.argmax(predictions)
			if(prediction_with_highest_confidence == curr_ground_truth):
				true_positive_count += 1
		val_accuracy = calc_accuracy(weights, validation_images, validation_labels, num_classes)
		if(val_accuracy > best_accuracy):
			best_accuracy = val_accuracy
			best_weights = weights
	return best_weights

def main():
	parser = argparse.ArgumentParser(description='Digit Classification using Perceptron')
	parser.add_argument('--training_data_path', required=True, help='Path to training data')
	parser.add_argument('--training_label_path', required=True, help='Path to training data')
	parser.add_argument('--validation_data_path', required=True, help='Path to validation data')
	parser.add_argument('--validation_label_path', required=True, help='Path to validation data')
	parser.add_argument('--test_data_path', required=True, help='Path to Testing data')
	parser.add_argument('--test_label_path', required=True, help='Path to Testing data')
	parser.add_argument('--num_epochs', required=True, help='Number of epochs')
	args = parser.parse_args()

	num_epochs = int(args.num_epochs)

	training_labels = readLabels(args.training_label_path)
	training_images = readImages(args.training_data_path, len(training_labels))
	num_classes = len(set(training_labels))

	validation_labels = readLabels(args.validation_label_path)
	validation_images = readImages(args.validation_data_path, len(validation_labels))

	testing_labels = readLabels(args.test_label_path)
	testing_images = readImages(args.test_data_path, len(testing_labels))

	weights = train_perceptron(training_images, training_labels, validation_images, validation_labels, num_classes, num_epochs)
	test_accuracy = calc_accuracy(weights, testing_images, testing_labels, num_classes)
	print("Accuracy on Digit Classification Test-Set is %f" %(test_accuracy))

if __name__ == '__main__':
	main()
