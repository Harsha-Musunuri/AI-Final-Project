# perceptron.py
# -------------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# For more info, see http://inst.eecs.berkeley.edu/~cs188/sp09/pacman.html

# Perceptron implementation
import util
PRINT = True

class PerceptronClassifier:
  """
  Perceptron classifier.

  Note that the variable 'datum' in this code refers to a counter of features
  (not to a raw samples.Datum).
  """
  def __init__( self, legalLabels, max_iterations): #constructor for Perceptron class. #number of times to train on each percent of data is defaulted to 3 - check in the options menu
    self.legalLabels = legalLabels
    self.type = "perceptron"
    self.max_iterations = max_iterations
    self.bestWeights={}
    for label in legalLabels: 
      self.bestWeights[label] = util.Counter()
    self.weights = {}
    for label in legalLabels: #legallabels are face or not face for face and 0 to 9 for numbers
      self.weights[label] = util.Counter() # this is the data-structure you should use 
      #counter() creates a dictionary with all keys mapped to value = 0; weights[0], weights[1] are two keys of the dictionary self.weights
       # for which 2 dictonary are mapped as values (nested dictionaries) 
    

    self.extra = True

  def setWeights(self, weights):
    assert len(weights) == len(self.legalLabels);
     #returns an assertionError, this is used as a debugging method.
    self.weights == weights;

  def train( self, trainingData, trainingLabels, validationData, validationLabels ):

    """
    The training loop for the perceptron passes through the training data several
    times and updates the weight vector for each label based on classification errors.
    See the project description for details.

    Use the provided self.weights[label] data structure so that
    the classify method works correctly. Also, recall that a
    datum is a counter from features to values for those features
    (and thus represents a vector a values).
    """
    # print('before zig wig begins')
    # print(self.weights)
    self.features = trainingData[0].keys() # could be useful later #self.features is a list of keys i:e all the x,y tuples possible for each datum. x = coloumn and y is row.
    #these x,y tuples are the features and their value is feature value.
    # DO NOT ZERO OUT YOUR WEIGHTS BEFORE STARTING TRAINING, OR
    # THE AUTOGRADER WILL LIKELY DEDUCT POINTS.
    acc = 0
    for iteration in range(self.max_iterations): #range always start from 0 and extend till one less than max value. 
    #range(5) = [0,1,2,3,4]
        print ("Executing iteration " + str(iteration + 1) + "...")
        # self.weights[0][(54,22)] = 1 #checkpoint
        # print (self.weights) #weights are empty dictionaries at this point.
        for i in range(len(trainingData)): #for each datum in trainingData
            "******************* YOUR CODE HERE *******************"

            target = None #None is to define a null variable
            label = None

            for x in self.legalLabels: #0,1 for faces and 0-9 for digits

                count = 0 #this is f(n)=weights * feature values associated with each weight.
                for f, num in trainingData[i].items(): #.items() will create a iterable (here it is created from a dictionary of key value pairs; 
                        # each dictionary corresponds to an image and its 'keys' corresponding to pixels & 'values' tell the pixel strenth - gray,black,empty  as list of key-value tuples )

                        #Example:
                        # >>> for x,y in [('one',1),('two',2),('three',3),('four',4)]:
                        # print(x, y)
                        # ... 
                        # one 1
                        # two 2
                        # three 3
                        # four 4
                        # >>> dictionary = {'one':1,'two':2,'three':3,'four':4}
                        # >>> dictionary.items()
                        # dict_items([('one', 1), ('two', 2), ('three', 3), ('four', 4)])
                        # >>> for x,y in dictionary.items():
                        # ...     print(x, y)
                        # ... 
                        # one 1
                        # two 2
                        # three 3
                        # four 4
                        # >>>   
                        count += num * self.weights[x][f]
                        # print('weights \n')
                        # if(self.weights[x][f] == 1):
                        #     print(x)
                        #     print('\n')
                        #     print(f)
                        # else:
                        #     print('zeros all weights')
                        
                        # print('f is: \n')
                        # print(f)
                        # print('***********')

                        # print count




                #output the scoring
                if target is None:
                    # print('I am in target is none')
                    target = count
                    # print('x is \n ')
                    # print(x)
                    # print('i is \n')
                    # print(i)
                    # print('Target is: ')
                    # print (target)

                    label = x
                    # print('Label is: ')
                    # print (label)
                    # print('If block is done \n')




                elif count > target:
                    # print('am in elif')
                    target = count
                    # print('x is \n ')
                    # print(x)
                    # print('i is \n')
                    # print(i)
                    # print('Target is: ')
                    # print (target)

                    label = x
                #     print('label is : \n')
                #     print (label)
                #     print('ElIf block is done \n')
                # else:
                #         print('I am in x = 1 loop iteration')
                #         print('else block is done \n')





            # print("I came back to outer for loop that is iterating through images")
            t = trainingLabels[i]
            #if predicted value == actual value then do nothing, otherwise update the weights accordingly
            if label == t:
                tempp = "hit"
                # print ('Do Nothing')
            else:
                self.weights[t] = self.weights[t] + trainingData[i] #overloaded addition operation to update all the weights with the sum of current weight+training data's pixel value.
                #print "working"
                self.weights[label] = self.weights[label] - trainingData[i]
                #print "working"
            # print('done wiht outer loop part will enter into new iteration for outer loop')


        val = self.classify(validationData)
        #print val

        len1 = len(validationLabels)
        ran = range(len1)
        bestFunctionVal = 0
        for j in ran:
            if val[j] == validationLabels[j]:
                bestFunctionVal = bestFunctionVal +1
        #print superCount
        print ("i:", iteration, " accuracy:", (100.0 * bestFunctionVal / len1))

        #switch accordingly
        if bestFunctionVal > acc:
            newW = self.weights.copy()
            acc = bestFunctionVal

    self.bestWeights = newW
    # print(self.bestWeights)
    # print('after all zig wig')
    # print(self.weights)


  def classify(self, data ):
    """
    Classifies each datum as the label that most closely matches the prototype vector
    for that label.  See the project description for details.

    Recall that a datum is a util.counter...
    """
    # print('Inside Classify method , printing Weights')
    # print(self.weights)
    guesses = []
    for datum in data:
      vectors = util.Counter()
      for l in self.legalLabels:
        vectors[l] = self.weights[l] * datum
      guesses.append(vectors.argMax())
    return guesses

  def classifyTestData(self, data ):
    """
    Classifies each datum as the label that most closely matches the prototype vector
    for that label.  See the project description for details.

    Recall that a datum is a util.counter...
    """
    # print('Inside ClassifyTestData class, printing bestWeights')
    # print(self.bestWeights)
    guesses = []
    for datum in data:
      vectors = util.Counter()
      for l in self.legalLabels:
        vectors[l] = self.bestWeights[l] * datum
      guesses.append(vectors.argMax())
    return guesses   



  def findHighWeightFeatures(self, label):
    """
    Returns a list of the 100 features with the greatest weight for some label
    """
    featuresWeights = []

    temp = self.weights[label]
    #print temp

    featuresWeights = temp.sortedKeys()[:100]
    #print featuresWeights

    return featuresWeights
