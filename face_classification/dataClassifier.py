# dataClassifier.py
# -----------------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# For more info, see http://inst.eecs.berkeley.edu/~cs188/sp09/pacman.html

# This file contains feature extraction methods and harness
# code for data classification

import naiveBayes
import perceptron
import samples
import sys
import util
import random
import time
import math

TEST_SET_SIZE = 100
DIGIT_DATUM_WIDTH=28
DIGIT_DATUM_HEIGHT=28
FACE_DATUM_WIDTH=60
FACE_DATUM_HEIGHT=70 

#extracting digit features.
def basicFeatureExtractorDigit(datum):
  """
  Returns a set of pixel features indicating whether
  each pixel in the provided datum is white (0) or gray/black (1)
  """
  a = datum.getPixels() #getPixels() returns a 2d array of edge map and assigns it to a;

  features = util.Counter() #creates a dictionary, where any key is initially mapped to a value 0
  for x in range(DIGIT_DATUM_WIDTH):
    for y in range(DIGIT_DATUM_HEIGHT):
      if datum.getPixel(x, y) > 0:
        features[(x,y)] = 1
      else:
        features[(x,y)] = 0
  return features
  
#................extracting face features................
def basicFeatureExtractorFace(datum):
  """
  Returns a set of pixel features indicating whether
  each pixel in the provided datum is an edge (1) or no edge (0)
  """
  a = datum.getPixels() #getPixels() returns a 2d array of edge map and assigns it to a;

  features = util.Counter() #creates a dictionary, where any key is initially mapped to a value 0
  for x in range(FACE_DATUM_WIDTH): #for each column of the pixels 2d array
    for y in range(FACE_DATUM_HEIGHT): #for each row of the pixels 2d array
    #creating features for every pixel, -> we are considering 60*70 pixels as features -> 4200 features for each sample.
      if datum.getPixel(x, y) > 0: #for grey pixel or black pixel just make the value as 1 for the feature. Here, feature is taken as a tuple (x,y) - x is column and y is row
        features[(x,y)] = 1 
      else:
        features[(x,y)] = 0 #else make the feature value as 0
  return features

def enhancedFeatureExtractorDigit(datum):
  """
  Your feature extraction playground.

  You should return a util.Counter() of features
  for this datum (datum is of type samples.Datum).

  ## DESCRIBE YOUR ENHANCED FEATURES HERE...

  ##
  """
  features =  basicFeatureExtractorDigit(datum)

  "*** YOUR CODE HERE ***"
  # Not used

  return features


def contestFeatureExtractorDigit(datum):
  """
  Specify features to use for the minicontest
  """
  features =  basicFeatureExtractorDigit(datum)
  return features

def enhancedFeatureExtractorFace(datum):
  """
  Your feature extraction playground for faces.
  It is your choice to modify this.
  """
  features =  basicFeatureExtractorFace(datum)
  return features

def analysis(classifier, guesses, testLabels, testData, rawTestData, printImage):
  """
  This function is called after learning.
  Include any code that you want here to help you analyze your results.

  Use the printImage(<list of pixels>) function to visualize features.

  An example of use has been given to you.

  - classifier is the trained classifier
  - guesses is the list of labels predicted by your classifier on the test set
  - testLabels is the list of true labels
  - testData is the list of training datapoints (as util.Counter of features)
  - rawTestData is the list of training datapoints (as samples.Datum)
  - printImage is a method to visualize the features
  (see its use in the odds ratio part in runClassifier method)

  This code won't be evaluated. It is for your own optional use
  (and you can modify the signature if you want).
  """

  # Put any code here...
  # Example of use:
  for i in range(len(guesses)):
      prediction = guesses[i]
      truth = testLabels[i]
      if (prediction != truth):
          print ("===================================")
          print ("Mistake on example %d" % i)
          print ("Predicted %d; truth is %d" % (prediction, truth))
          print ("Image: ")
          print (rawTestData[i])
          continue


## =====================
## You don't have to modify any code below.
## =====================


class ImagePrinter:
  #imagePrinter is to create a image from " ". # +
    def __init__(self, width, height):
      self.width = width
      self.height = height

    def printImage(self, pixels):
      """
      Prints a Datum object that contains all pixels in the
      provided list of pixels.  This will serve as a helper function
      to the analysis function you write.

      Pixels should take the form
      [(2,2), (2, 3), ...]
      where each tuple represents a pixel.
      """
      image = samples.Datum(None,self.width,self.height)
      for pix in pixels:
        try:
            # This is so that new features that you could define which
            # which are not of the form of (x,y) will not break
            # this image printer...
            x,y = pix
            image.pixels[x][y] = 2
        except:
            print ("new features:", pix)
            continue
      print (image)

def default(str):
  return str + ' [Default: %default]'

def readCommand( argv ):
  "Processes the command used to run from the command line."
  from optparse import OptionParser #is a powerful tool to parsing command line options.
  parser = OptionParser(USAGE_STRING)
  #parser.add_option('-f', '--features', help=default('Whether to use enhanced features'), default=False, action="store_true")
  #-f or --features both mean the same option, we can use either of them on the command line.
  #<script> -h will print all the help texts set for each option.
  #default: it sets the option.features to false if the option is not present in command line. but it present always, action is chosen i:e. True


  parser.add_option('-c', '--classifier', help=default('The type of classifier'), choices=['mostFrequent', 'nb', 'naiveBayes', 'perceptron'], default='naiveBayes')
  parser.add_option('-d', '--data', help=default('Dataset to use'), choices=['digits', 'faces'], default='digits')
  parser.add_option('-t', '--training', help=default('The size of the training set'), default=100, type="int")
  parser.add_option('-f', '--features', help=default('Whether to use enhanced features'), default=False, action="store_true")
  parser.add_option('-o', '--odds', help=default('Whether to compute odds ratios'), default=False, action="store_true")
  parser.add_option('-1', '--label1', help=default("First label in an odds ratio comparison"), default=0, type="int")
  parser.add_option('-2', '--label2', help=default("Second label in an odds ratio comparison"), default=1, type="int")
  parser.add_option('-w', '--weights', help=default('Whether to print weights'), default=False, action="store_true")
  parser.add_option('-k', '--smoothing', help=default("Smoothing parameter (ignored when using --autotune)"), type="float", default=2.0)
  parser.add_option('-a', '--autotune', help=default("Whether to automatically tune hyperparameters"), default=False, action="store_true")
  parser.add_option('-i', '--iterations', help=default("Maximum iterations to run training"), default=15, type="int")
  parser.add_option('-s', '--test', help=default("Amount of test data to use"), default=TEST_SET_SIZE, type="int")
  parser.add_option('-n', '--analysis', help=default("Shows which data is wrongly predicted"), default=False, action="store_true")
  parser.add_option('-r', '--random', help=default("Trains the data set using random data and \
   calculates averages for percent accuracy and standard deviation"), default=False, action="store_true")

  options, otherjunk = parser.parse_args(argv)
  if len(otherjunk) != 0: raise Exception('Command line input not understood: ' + str(otherjunk))
  args = {} #empty dictionary to capture the command line inputs.

  # Set up variables according to the command line input. This is the start line of the whole drama.
  print ("Doing classification")
  print ("--------------------")
  print ("Data:\t\t" + options.data)
  print ("Classifier:\t\t" + options.classifier)
  print ("Using enhanced features?:\t" + str(options.features))
  if not options.random:
      print ("Training set size:\t" + str(options.training))

  if(options.data=="digits"):
    printImage = ImagePrinter(DIGIT_DATUM_WIDTH, DIGIT_DATUM_HEIGHT ).printImage #DIGIT_DATUM_WIDTH, DIGIT_DATUM_HEIGHT are global variables

    if (options.features):
      featureFunction = enhancedFeatureExtractorDigit
    else:
      featureFunction = basicFeatureExtractorDigit

  elif(options.data=="faces"):
    printImage = ImagePrinter(FACE_DATUM_WIDTH, FACE_DATUM_HEIGHT).printImage #FACE_DATUM_WIDTH, FACE_DATUM_HEIGHT are global variables 
    #& creating an object of class ImagePrinter.
    #print("ImagePrinter is used")
    if (options.features): #to decide on what to choose b/w enhancedFeatureExtractorFace function or basicFeatureExtractorFace function.
      featureFunction = enhancedFeatureExtractorFace
    else:
      featureFunction = basicFeatureExtractorFace
  else: #if both digits and faces are not what we called on the command prompt.
    print ("Unknown dataset", options.data)
    print (USAGE_STRING)
    sys.exit(2)

  if(options.data=="digits"):
    legalLabels = range(10) #0,1,2,3,4,5,6,7,8,9
  else:
    legalLabels = range(2) #face or not face

  #we are not keeping training <=0 hence, below wont be used  
  if options.training <= 0:
    print ("Training set size should be a positive integer (you provided: %d)" % options.training)
    print (USAGE_STRING)
    sys.exit(2)
  #we are not using smoothing. hence, below wont be used  
  if options.smoothing <= 0:
    print ("Please provide a positive number for smoothing (you provided: %f)" % options.smoothing)
    print (USAGE_STRING)
    sys.exit(2)
  #we are not using odds. hence, below wont be used  
  if options.odds:
    if options.label1 not in legalLabels or options.label2 not in legalLabels:
      print ("Didn't provide a legal labels for the odds ratio: (%d,%d)" % (options.label1, options.label2))
      print (USAGE_STRING)
      sys.exit(2)
  
  #defining decision structure based on asked classifier.   
  if(options.classifier == "naiveBayes" or options.classifier == "nb"):
    classifier = naiveBayes.NaiveBayesClassifier(legalLabels)
    classifier.setSmoothing(options.smoothing)
    if (options.autotune):
        print ("Using automatic tuning for naivebayes")
        classifier.automaticTuning = True
    else:
        print ("Using smoothing parameter k=%f for naivebayes" %  options.smoothing)

  elif(options.classifier == "perceptron"):
    classifier = perceptron.PerceptronClassifier(legalLabels,options.iterations) 
    #creating a PerceptronClassifier object by passing legalLabels and iterations=3 as max iterations to PerceptronClassifier's constructor.

  else:
    print ("Unknown classifier:", options.classifier)
    print (USAGE_STRING)

    sys.exit(2)

  args['classifier'] = classifier #assining classifier as a value to key 'classifier'
  args['featureFunction'] = featureFunction
  args['printImage'] = printImage

  return args, options

USAGE_STRING = """
  USAGE:      python dataClassifier.py <options>
  EXAMPLES:   (1) python dataClassifier.py
                  - trains the default mostFrequent classifier on the digit dataset
                  using the default 100 training examples and
                  then test the classifier on test data
              (2) python dataClassifier.py -c naiveBayes -d digits -t 1000 -f -o -1 3 -2 6 -k 2.5
                  - would run the naive Bayes classifier on 1000 training examples
                  using the enhancedFeatureExtractorDigits function to get the features
                  on the faces dataset, would use the smoothing parameter equals to 2.5, would
                  test the classifier on the test data and performs an odd ratio analysis
                  with label1=3 vs. label2=6
                 """

# Main harness code

def runClassifier(args, options):
    
    featureFunction = args['featureFunction']
    classifier = args['classifier']
    # print('Classifier printing')
    # print(type(classifier))
    printImage = args['printImage']
    
    # Load the data for testing, training and validation
    if(options.random):
        numberOfTestPoints = 150  #number of images that we want to take as test iamges  ?, total testdata given itself is 150 for faces.
        numberOfValidationPoints = 300 #number of images that we want to take as validation iamges  ?, total Validaitondata given itself is 300 for faces.
        totalTrainData = 450  #number of images that we want to take as train iamges  ?, total traindata given itself is 450 for faces.
        numValidation = numberOfValidationPoints
        numTest = numberOfTestPoints
        numTraining = totalTrainData

        if(options.data=="faces"):
            rawTestData = samples.loadDataFile("facedata/facedatatest", numTest,FACE_DATUM_WIDTH,FACE_DATUM_HEIGHT)
            testLabels = samples.loadLabelsFile("facedata/facedatatestlabels", numTest)
            # print('number if test labels ', len(testLabels))

            # print('testLabels')
            # print(testLabels)
            # testLabels
            rawValidationData = samples.loadDataFile("facedata/facedatatrain", numValidation,FACE_DATUM_WIDTH,FACE_DATUM_HEIGHT)
            validationLabels = samples.loadLabelsFile("facedata/facedatatrainlabels", numValidation)
            # print('number if test labels ', len(validationLabels))
            rawTrainingData = samples.loadDataFile("facedata/facedatatrain", numTraining,FACE_DATUM_WIDTH,FACE_DATUM_HEIGHT)
            trainingLabels = samples.loadLabelsFile("facedata/facedatatrainlabels", numTraining)
            # print('number if test labels ', len(trainingLabels))
        


        print ("Extracting features...")
        testData = map(featureFunction, rawTestData) #apply enhancedFeatureExtractorDigit or enhancedFeatureExtractorFace (according to cmd line) on each datum of the rawTestData. testData is list of dictionaries 
        # each dictionary corresponds to an image and its keys corresponding to pixels & values tell the pixel strenth - gray,black,empty
        #where in each dictionary key =tuple (x,y) -> x is column and y is row and value is 0 or 1 based on dark/grey or blank
        # print("features extracted")
        # print(type(testData))
        validationData = map(featureFunction, rawValidationData) #apply enhancedFeatureExtractorDigit or enhancedFeatureExtractorFace (according to cmd line) on each datum of the rawvalidationData. validationData is list of dictionaries; 
        # each dictionary corresponds to an image and its keys corresponding to pixels & values tell the pixel strenth - gray,black,empty
        #where in each dictionary key =tuple (x,y) -> x is column and y is row and value is 0 or 1 based on dark/grey or blank
        trainingData = map(featureFunction, rawTrainingData) #apply enhancedFeatureExtractorDigit or enhancedFeatureExtractorFace (according to cmd line) on each datum of the rawtrainingData. trainingData is list of dictionaries
        # each dictionary corresponds to an image and its keys corresponding to pixels & values tell the pixel strenth - gray,black,empty
        #where in each dictionary key =tuple (x,y) -> x is column and y is row and value is 0 or 1 based on dark/grey or blank
        # print (type(trainingData[0])) # each item of training data is a type util.counter.
        # print (trainingData[0].keys())
        # print("\n")
        # print (type(rawTrainingData)) #it is type of items -> list of Datum objects.
        #print (list(trainingData))
        if (classifier.type == 'perceptron'):
            # print(classifier.type)
            # print("hello")
            indexes_label1=[]
            indexes_label0=[]
            for i in range(0,totalTrainData):
              if(trainingLabels[i]==1):
                indexes_label1.append(i)
              else:
                indexes_label0.append(i)
            totalFaceTrainData=len(indexes_label1)
            totalNotFaceTrainData = len(indexes_label0)
            # print('totalFaceTrainData ',totalFaceTrainData)
            # print('totalNotFaceTrainData', totalNotFaceTrainData)
            FaceTrainData = []
            for i in indexes_label1:
              FaceTrainData.append(trainingData[i])
            NotFaceTrainData=[]
            for i in indexes_label0:
              NotFaceTrainData.append(trainingData[i])

            for percent in range(1,11): #[percent in 1,2,3,4,5,6,7,8,9,10]
                accuracy = [] #empty list
                times = [] #empty list
                print("\n")
                for runCount in range(0,5): #runcount in 0,1,2,3,4
                    # Extract features
                    print("======================================\n")
                    print ( "(" + str(runCount+1) + ")" +  " Extracting random " + str((percent * 10)) + "% of the training data...")
                    numSubTraining_Face = int((percent / 10.0) * totalFaceTrainData) 
                    numSubTraining_NotFace = int((percent / 10.0) * totalNotFaceTrainData)
                    

                    indexes_faces = random.sample(range(0, totalFaceTrainData), numSubTraining_Face)
                    indexes_notfaces = random.sample(range(0, totalNotFaceTrainData), numSubTraining_NotFace) #sample numSubTraining elements from the range 0 to totalTrainData
                    subTrainingData = [] #emptylist
                    subTrainingLabels = [] #emptylist

                    for indx in indexes_faces:
                        subTrainingData.append(FaceTrainData[indx]) #the datum at the indx is pushed into subTrainingData
                        subTrainingLabels.append(1) #the label at the indx is pushed into subTrainingModels
                    for indx in indexes_notfaces:
                        subTrainingData.append(NotFaceTrainData[indx]) #the datum at the indx is pushed into subTrainingData
                        subTrainingLabels.append(0)
                    
                    start = time.time()
                    print ("(" + str(runCount + 1) + ")", "Training on", str(numSubTraining_Face+numSubTraining_NotFace), "data points...")
                    classifier.weights={}
                    for label in [0,1]: #legallabels are face or not face for face and 0 to 9 for numbers
                      classifier.weights[label] = util.Counter()
                    classifier.train(subTrainingData, subTrainingLabels, validationData, validationLabels)
                    end = time.time()
                    elapsed = end - start
                    print ( "(" + str(runCount + 1)+ ")" + " Training completed in %0.4f second(s)" % elapsed)
                    times.append(elapsed)
                    print('perceptron classifier and its validation , testing')
                    # Validation
                    print ("("+str(runCount+1)+")", "Validating...")
                    guesses = classifier.classify(validationData)
                    correct = [guesses[i] == validationLabels[i] for i in range(len(validationLabels))].count(True)
                    print ("("+str(runCount + 1)+") " + str(correct), ("correct out of " + str(len(validationLabels)) + " (%.1f%%).") % (100.0 * correct / len(validationLabels)))
                
                    # Testing
                    print ("("+str(runCount+1)+")", "Testing...")
                    guesses = classifier.classifyTestData(testData)
                    correct = [guesses[i] == testLabels[i] for i in range(len(testLabels))].count(True)
                    print ("("+str(runCount + 1)+") " + str(correct), ("correct out of " + str(len(testLabels)) + " (%.1f%%).") % (100.0 * correct / len(testLabels)) + "\n")
                    accuracy.append(100.0 * correct / len(testLabels))

                avgAccuracy = 0
                print('calculating stdDev')
                avgTime = 0
                for q in range(0, len(accuracy)):
                    avgAccuracy += accuracy[q]
                    avgTime += times[q]
                    
                avgAccuracy = avgAccuracy/len(accuracy)
                avgTime = avgTime/len(times)
                
                print("=================\n")
                print ("Average training time for", str(numSubTraining_Face+numSubTraining_NotFace), "data points: %0.4f" % avgTime)
                print ("Average accuracy of " + str(percent * 10) + ("% data training: "), str(avgAccuracy))
                
                stdDev = 0
                for a in accuracy:
                    temp = a - avgAccuracy
                    stdDev += (temp*temp)
                stdDev = stdDev / (len(accuracy) - 1)
                stdDev = math.sqrt(stdDev)
                print ("Standard deviation of accuracy: %0.4f" % stdDev)
                #print
                    
        else:
            # print('entered NaiveBayes')
            for percent in range(1,2): #[percent in 1,2,3,4,5,6,7,8,9,10]
              accuracy = [] #empty list
              times = [] #empty list
              print("\n")
              for runCount in range(0,5): #runcount in 0,1,2,3,4
                  # Extract features
                  print("======================================\n")
                  print ( "(" + str(runCount+1) + ")" +  " Extracting random " + str((percent * 10)) + "% of the training data...")
                  numSubTraining = int((percent / 10.0) * totalTrainData) #10,20,30,40,50,60,70,80,90,100% of the training data
                  

                  indexes = random.sample(range(0, totalTrainData), numSubTraining) #sample numSubTraining elements from the range 0 to totalTrainData
                  subTrainingData = [] #emptylist
                  subTrainingLabels = [] #emptylist

                  for indx in indexes:
                      subTrainingData.append(trainingData[indx]) #the datum at the indx is pushed into subTrainingData
                      subTrainingLabels.append(trainingLabels[indx]) #the label at the indx is pushed into subTrainingModels
  
                # Conduct training and testing
                
                  start = time.time()
                  print ("(" + str(runCount + 1) + ")", "Training on", numSubTraining, "data points...")
                  classifier.weights={}
                  for label in [0,1]: #legallabels are face or not face for face and 0 to 9 for numbers
                    classifier.weights[label] = util.Counter()
                  classifier.train(subTrainingData, subTrainingLabels, validationData, validationLabels)
                  end = time.time()
                  elapsed = end - start
                  print ( "(" + str(runCount + 1)+ ")" + " Training completed in %0.4f second(s)" % elapsed)
                  times.append(elapsed)

                  # Validation
                  print ("("+str(runCount+1)+")", "Validating...")
                  guesses = classifier.classify(validationData)
                  correct = [guesses[i] == validationLabels[i] for i in range(len(validationLabels))].count(True)
                  print ("("+str(runCount + 1)+") " + str(correct), ("correct out of " + str(len(validationLabels)) + " (%.1f%%).") % (100.0 * correct / len(validationLabels)))
                  
                  # Testing
                  print ("("+str(runCount+1)+")", "Testing...")
                  guesses = classifier.classify(testData)
                  correct = [guesses[i] == testLabels[i] for i in range(len(testLabels))].count(True)
                  print ("("+str(runCount + 1)+") " + str(correct), ("correct out of " + str(len(testLabels)) + " (%.1f%%).") % (100.0 * correct / len(testLabels)) + "\n")
                  accuracy.append(100.0 * correct / len(testLabels))

              avgAccuracy = 0
              avgTime = 0
              for q in range(0, len(accuracy)):
                  avgAccuracy += accuracy[q]
                  avgTime += times[q]
                  
              avgAccuracy = avgAccuracy/len(accuracy)
              avgTime = avgTime/len(times)
              
              print("=================\n")
              print ("Average training time for", numSubTraining, "data points: %0.4f" % avgTime)
              print ("Average accuracy of " + str(percent * 10) + ("% data training: "), str(avgAccuracy))
              
              stdDev = 0
              for a in accuracy:
                  temp = a - avgAccuracy
                  stdDev += (temp*temp)
              stdDev = stdDev / (len(accuracy) - 1)
              stdDev = math.sqrt(stdDev)
              print ("Standard deviation of accuracy: %0.4f" % stdDev)
              #print

        sys.exit(1)
    

if __name__ == '__main__':
  "********************************************************************************************************************"
  # Read input
  args, options = readCommand( sys.argv[1:] ) #sys.argv[1:] slices the command line. this excludes sys.argv[0] which is the program name by itself.
"********************************************************************************************************************"
  # Run classifier

runClassifier(args, options) #calling runClassifier.
