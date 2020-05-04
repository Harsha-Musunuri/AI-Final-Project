# naiveBayes.py
# -------------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# For more info, see http://inst.eecs.berkeley.edu/~cs188/sp09/pacman.html

import util
import classificationMethod
import math

class NaiveBayesClassifier(classificationMethod.ClassificationMethod):
  """
  See the project description for the specifications of the Naive Bayes classifier.

  Note that the variable 'datum' in this code refers to a counter of features
  (not to a raw samples.Datum).
  """
  def __init__(self, legalLabels):
    self.legalLabels = legalLabels
    self.type = "naivebayes"
    self.k = 1 # this is the smoothing parameter, ** use it in your train method **
    self.automaticTuning = False # Look at this flag to decide whether to choose k automatically ** use this in your train method **
    self.probs = {}
    self.extra = False

  def setSmoothing(self, k):
    """
    This is used by the main method to change the smoothing parameter before training.
    Do not modify this method.
    """
    self.k = k

  def train(self, trainingData, trainingLabels, validationData, validationLabels):
    """
    Outside shell to call your method. Do not modify this method.
    """

    # might be useful in your code later...
    # this is a list of all features in the training set.
    self.features = list(set([ f for datum in trainingData for f in datum.keys() ])); #all the pixel locations of 60 x 70 image
    # print(len(self.features)) 
    '''
    >>> a=[{(1,2):1,(2,3):4},{(1,2):1,(2,3):4},{(1,2):1,(2,3):4}]
    >>> features = list(set([ f for datum in a for f in datum.keys() ]))
    >>> features
    [(1, 2), (2, 3)]
    >>> a=[{(14,2):1,(65,0):4},{(1,8):1,(2,0):4},{(9,7):1,(6,7):4}]
    >>> features = list(set([ f for datum in a for f in datum.keys() ]))
    >>> features
    [(6, 7), (2, 0), (1, 8), (65, 0), (14, 2), (9, 7)]
    '''

    if (self.automaticTuning):
        kgrid = [0.001, 0.01, 0.05, 0.1, 0.5, 1, 5, 10, 20, 50]
    else:
        kgrid = [self.k] #by default it is 2.0

    self.trainAndTune(trainingData, trainingLabels, validationData, validationLabels, kgrid)

  def trainAndTune(self, trainingData, trainingLabels, validationData, validationLabels, kgrid):
    """
    Trains the classifier by collecting counts over the training data, and
    stores the Laplace smoothed estimates so that they can be used to classify.
    Evaluate each value of k in kgrid to choose the smoothing parameter
    that gives the best accuracy on the held-out validationData.

    trainingData and validationData are lists of feature Counters.  The corresponding
    label lists contain the correct label for each datum.

    To get the list of all possible features or labels, use self.features and
    self.legalLabels.
    """

    countTotal = util.Counter()
    priorD = util.Counter()
    # print("count printing util counter")
    # print(count[2])

    for label in trainingLabels:
        countTotal[label] += 1 #counts total number of faces and maps to key 1 and not faces to key 0
        priorD[label] += 1 
    #Debug Lines
    # print('total no.of images with label as 1 and 0 '+str(priorD[1])+ ' ' + str(priorD[0]))
    # print('total no.of images with label as 1 '+str(count[1]))
    # print('total no.of images with label as 0 '+str(count[0]))
    # print('number of images in trainingLabels '+str(len(trainingLabels)))


    count = {}
    # print("count printing")
    # print(count[2])

    for feat in self.features:

        count[feat] = {}
        # >>> count
        # {(6, 7): {}, (2, 0): {}, (1, 8): {}, (65, 0): {}, (14, 2): {}, (9, 7): {}}

        for label in self.legalLabels:
            count[feat][label] = {
                0: 0,
                1: 0
            } #count[feat] is a dictionary. with 2 keys - 0,1; assigning dictionary {0:0,1:0} for each key
            #{(3, 4): {0: {0: 0, 1: 0}, 1: {0: 0, 1: 0}}, (1, 2): {0: {0: 0, 1: 0}, 1: {0: 0, 1: 0}}}
            #(3,4) is a pixel which has an assigned dictionary as a value and that dictionary has 0, 1 as keys with value as another dictionary {0:0,1:0}

    for i in range(len(trainingData)): #going through all images

        datum = trainingData[i]
        label = trainingLabels[i]
        # k=1
        for (feat, val) in datum.items(): #for each pixel (x,y) and its value val update count. (.items() give list of tuples where first element of tuple here is a tuple that represents cordinates of pixel)
            count[feat][label][val] += 1
            ##{(3, 4): {0 (# this denotes that the pixel belongs to image of type not face): {0: 0 (#num of times this pixel has 0 as score), 1: 0(#num of times this pixel has 0 as score)},
             # 1(# this denotes that the pixel belongs to image of type face): {0: 0 (#num of times this pixel has 0 as score), 1: 0(#num of times this pixel has 0 as score)}
            
            # print('Count update for image'+str(k)) 
            # print(count)
            # k+=1 
    # print('count after the updates \n')        
    # print(count) 

    # print('Below is priorD number of face labelled images and non face labelled images: \n')
    # print(priorD)
    priorD.normalize()
    # print('normalized priorD \n')
    # print(priorD)
    self.priorD = priorD #saving probability of face and probability of not face.
    # print ('printinng count.items()[1] \n')
    # print(count.items()[1])
    # for (feat,label) in [((31, 6), {0: {0: 19, 1: 1}, 1: {0: 20, 1: 5}})]: #debug lines
    #     print('feat \n')
    #     print (feat)
    #     print('\n')
    #     print('label \n')
    #     print (label)
    # Using Laplace smoothing to tune the data and find the best k value that gives the highest accuracy

    # bestK = -1
    # bestAcc = -1
    for k in kgrid:
        tempProb = {}
        # print('temp prob in the beginning: ')
        # print(tempProb)
        for (feat, labels) in count.items(): #count.items -> tuple of pixel - feat => (cordinates) & lables => {0:{1:?,0:?},1:{1:?,0:?}}
            tempProb[feat] = {}
            # print('temp prob after adding keys: ')
            # print(tempProb)
            for (label, vals) in labels.items(): #labels.items => [(0,{1:?,0:?}),(1,{1:?,0:?})]
                # print('labels and its associated dictionary as list of tuples for each pixel: ')
                # print(labels.items())
                tempProb[feat][label] = {}
                # print(count[feat][label])
                # print('temp prob after adding labels: ')
                # print(tempProb)
                # total = sum(count[feat][label].values()) #sum of number of times this pixel had value 0 and number of times this pixel had value 1 on a label face or non face accordingly
                # print(total)
                # total += 2*k
                # print('total is \n')
                # print(total)
                # print('val.items: ')
                # print(vals.items())
                for (val, c) in vals.items():

                    # print('val is:')
                    # print(val)
                    # print('c is: ')
                    # print(c)
                    #Normalizing the probability
                    # print('current number of times the feature had a value of: ')
                    # print(count[feat][label][val])
                    # print('total is ')
                    # print(total)
                    tempProb[feat][label][val] = (count[feat][label][val]+k) / countTotal[label]

        self.probs = tempProb

        predictions = self.classify(validationData)

        # Count number of correct predictions
        acc = 0
        for i in range(len(predictions)):
            if predictions[i] == validationLabels[i]:
                acc += 1


  def classify(self, testData):
    """
    Classify the data based on the posterior distribution over labels.

    You shouldn't modify this method.
    """
    guesses = []
    self.posteriors = [] # Log posteriors are stored for later data analysis (autograder).
    for datum in testData:
      posterior = self.calculateLogJointProbabilities(datum)
      guesses.append(posterior.argMax())
      self.posteriors.append(posterior)
    return guesses

  def calculateLogJointProbabilities(self, datum):
    """
    Returns the log-joint distribution over legal labels and the datum.
    Each log-probability should be stored in the log-joint counter, e.g.
    logJoint[3] = <Estimate of log( P(Label = 3, datum) )>

    To get the list of all possible features or labels, use self.features and
    self.legalLabels.
    """
    logJoint = util.Counter()
    #print self.priorD

    for label in self.legalLabels:
        logJoint[label] = math.log(self.priorD[label])
        #print str(logJoint[label])

        for (feat, val) in datum.items():
            #print "inside of datum for loop"
            p = self.probs[feat][label][val];
            # print('P')
            # print(p)
            logJoint[label] += math.log(p)#final value = log(P(phi_1))+log(P(phi_2))+log(P(phi_3))+log(P(phi_4))+.........+log(P(Face)) when label = face and vice versa
            #all we have to get is the max val b/w logJoin[0], logJoin[1] -> the corresponding label is the prediction.

    return logJoint

  def findHighOddsFeatures(self, label1, label2):
    """
    Returns the 100 best features for the odds ratio:
            P(feature=1 | label1)/P(feature=1 | label2)

    Note: you may find 'self.features' a useful way to loop through all possible features
    """
    featuresOdds = []

    "*** YOUR CODE HERE ***"

    for feat in self.features:
        #Doing what is defined above in the comment P(feature=1 | label1)/P(feature=1 | label2)
        featuresOdds.append((self.probs[feat][label1][1] / self.probs[feat][label2][1], feat))

    #First we sort the featuresOdds list and then reverse it to get the last 100 of the list
    featuresOdds.sort()
    fOdds = list(map(lambda x: x[1], featuresOdds[-100:]))

    return fOdds
