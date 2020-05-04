# samples.py
# ----------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# For more info, see http://inst.eecs.berkeley.edu/~cs188/sp09/pacman.html

import util

## Constants
DATUM_WIDTH = 0 # in pixels
DATUM_HEIGHT = 0 # in pixels

## Module Classes

class Datum:
  """
  A datum is a pixel-level encoding of digits or face/non-face edge maps.

  Digits are from the MNIST dataset and face images are from the
  easy-faces and background categories of the Caltech 101 dataset.


  Each digit is 28x28 pixels, and each face/non-face image is 60x74
  pixels, each pixel can take the following values:
    0: no edge (blank)
    1: gray pixel (+) [used for digits only]
    2: edge [for face] or black pixel [for digit] (#)

  Pixel data is stored in the 2-dimensional array pixels, which
  maps to pixels on a plane according to standard euclidean axes
  with the first dimension denoting the horizontal and the second
  the vertical coordinate:

    28 # # # #      #  #
    27 # # # #      #  #
     .
     .
     .
     3 # # + #      #  #
     2 # # # #      #  #
     1 # # # #      #  #
     0 # # # #      #  #
       0 1 2 3 ... 27 28

  For example, the + in the above diagram is stored in pixels[2][3], or
  more generally pixels[column][row].

  The contents of the representation can be accessed directly
  via the getPixel and getPixels methods.
  """
  def __init__(self,data,width,height): #__init__ is the constructor in python & used to initialize the attributes of a datum class to the object created.
    #self is the object created by the class datum. this is same as 'This' is oops.
    """
    Create a new datum from file input (standard MNIST encoding).
    """
    DATUM_HEIGHT = height
    DATUM_WIDTH=width
    self.height = DATUM_HEIGHT 
    self.width = DATUM_WIDTH
    if data == None:
      data = [[' ' for i in range(DATUM_WIDTH)] for j in range(DATUM_HEIGHT)] #creates a list of lists (is a 2d matrix) 
      #of ' ' of no. of rows = datum_height; no. of columns = datum_width 
      #below executed from python terminal.
      # data = [['*' for i in range(2)] for j in range(3)]
      # print(data)
      # [['*', '*'], ['*', '*'], ['*', '*']]
      # EX:- a = [[1,2,3],[4,5,6],[7,8,9],[10,11,12]]
      # len(a)
      # 4 - number of rows
      # len(a[0])
      # 3 - number of columns
      #it is a 4 row,3 column matrix
    self.pixels = util.arrayInvert(convertToInteger(data)) #after this, pixels values are 0 ,1, 2 for ' ', '+', '*' and in euclidean axes format.

  def getPixel(self, column, row):
    """
    Returns the value of the pixel at column, row as 0, 1 or 2.
    """
    return self.pixels[column][row]

  def getPixels(self):
    """
    Returns all pixels as a list of lists in the euclidean axes format.
    """
    return self.pixels

  def getAsciiString(self):
    """
    Renders the data item as an ascii image.
    """
    rows = []
    data = util.arrayInvert(self.pixels)
    for row in data:
      ascii = map(asciiGrayscaleConversionFunction, row)
      rows.append( "".join(ascii) )
    return "\n".join(rows)

  def __str__(self):
    return self.getAsciiString()



# Data processing, cleanup and display functions

def loadDataFile(filename, n,width,height):
  """
  Reads n data images from a file and returns a list of lines read from the given file..(Datum objects.)

  (Return less then n items if the end of file is encountered).
  """
  DATUM_WIDTH=width
  DATUM_HEIGHT=height
  fin = readlines(filename)
  fin.reverse() #inbuilt funtion to reverse a list in place, because we pop the last element of the list and the last element to be the first element of the file, we need to reverse.
  #for example if the file has 
  #harsha@myBeast Image Classification % python readImages.py
  # original
  # no. of elements = 6
  # ['Harsha', 'Rutgers', 'Oneplus', 'Iphone', 'Alright', '      ']
  # reversed
  # ['      ', 'Alright', 'Iphone', 'Oneplus', 'Rutgers', 'Harsha']
  # popping the last element of list (which is the first element of original list): 
  # Harsha
  # final list
  # ['      ', 'Alright', 'Iphone', 'Oneplus', 'Rutgers']
  items = []
  for i in range(n):
    data = []
    for j in range(height): #for number of rows = 70 push each element i:e list (of all the characters in the line read from the file) into data => i:e data is a list of lists.
    #(the reason we do this to forumulate the data as a 2d matrix, cuz 2d matrix is list of lists)
      data.append(list(fin.pop())) #pop the last element of the list as a list of characters. for ex, if the line as harsha => outputs a list ['h','a','r','s','h','a']
      #data will be a list of lists, where each list is a row of the image.
    if len(data[0]) < DATUM_WIDTH-1: #at some point when we are at the end of the file, we want to exit the outer for loop. this may happen when n>the images available
      # we encountered end of file...
      print ("Truncating at %d examples (maximum)" % i)
      break
    items.append(Datum(data,DATUM_WIDTH,DATUM_HEIGHT)) #each image is convereted to datum object and pushed into items. items is a list of Datum objects where each Datum is a list of lists i:e euclidean matrix of pixels.
  return items

import zipfile
import os
def readlines(filename):
  "Opens a file or reads it from the zip archive data.zip"
  if(os.path.exists(filename)):
    return [l[:-1] for l in open(filename).readlines()] #gives a list of strings - where each string is a line of the file.

  else:
    z = zipfile.ZipFile('data.zip')
    return z.read(filename).split('\n')

def loadLabelsFile(filename, n):
  """
  Reads n labels from a file and returns a list of integers.
  """
  fin = readlines(filename)
  labels = []
  for line in fin[:min(n, len(fin))]:
    if line == '':
        break
    labels.append(int(line))
  return labels

def asciiGrayscaleConversionFunction(value):
  """
  Helper function for display purposes.
  """
  if(value == 0):
    return ' '
  elif(value == 1):
    return '+'
  elif(value == 2):
    return '#'

def IntegerConversionFunction(character):
  """
  Helper function for file reading.
  """
  if(character == ' '):
    return 0
  elif(character == '+'):
    return 1
  elif(character == '#'):
    return 2

def convertToInteger(data):
  """
  Helper function for file reading.
  """
  if type(data) != type([]):
    return IntegerConversionFunction(data)
  else:
    return map(convertToInteger, data)

# Testing

def _test():
  import doctest
  doctest.testmod() # Test the interactive sessions in function comments
  n = 1
#  items = loadDataFile("facedata/facedatatrain", n,60,70)
#  labels = loadLabelsFile("facedata/facedatatrainlabels", n)
  items = loadDataFile("digitdata/trainingimages", n,28,28)
  labels = loadLabelsFile("digitdata/traininglabels", n)
  for i in range(1):
    print (items[i])
    print (items[i])
    print (items[i].height)
    print (items[i].width)
    print (dir(items[i]))
    print (items[i].getPixels())

if __name__ == "__main__":
  _test()
