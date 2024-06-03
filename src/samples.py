# samples.py
# ----------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# For more info, see http://inst.eecs.berkeley.edu/~cs188/sp09/pacman.html

import util
import zipfile
import os

## Constants
DATUM_WIDTH = 0 # in pixels
DATUM_HEIGHT = 0 # in pixels

## Module Classes

class Datum:
    def __init__(self, data, width, height):
        self.width = width
        self.height = height
        self.pixels = [[' ' for _ in range(width)] for _ in range(height)]
        
        # Populate the pixels array with provided data
        for i in range(min(len(data), height)):
            for j in range(min(len(data[i]), width)):
                self.pixels[i][j] = data[i][j]

    def getPixel(self, column, row):
        if 0 <= row < self.height and 0 <= column < self.width:
            return self.pixels[row][column]
        return ' '  # Default value if out of bounds

    def getPixels(self):
        return self.pixels

def loadDataFile(filename, n, width, height):
    """
    Reads n data images from a file and returns a list of Datum objects.

    (Return less then n items if the end of file is encountered).
    """
    fin = readlines(filename)
    items = []
    for i in range(n):
        data = []
        for j in range(height):
            data.append(list(map(convertToInteger, fin.pop(0))))
        if len(data[0]) < width-1:
            print("Truncating at %d examples (maximum)" % i)
            break
        items.append(Datum(data, width, height))
    return items

def readlines(filename):
    "Opens a file or reads it from the zip archive data.zip"
    if(os.path.exists(filename)):
        with open(filename, 'r') as f:
            return [l.strip() for l in f.readlines()]
    else:
        with zipfile.ZipFile('data.zip') as z:
            return z.read(filename).decode('utf-8').split('\n')

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
    if isinstance(data, list):
        return list(map(convertToInteger, data))
    else:
        return IntegerConversionFunction(data)

# Testing

def _test():
    import doctest
    doctest.testmod() # Test the interactive sessions in function comments
    n = 1
    items = loadDataFile("digitdata/trainingimages", n,28,28)
    labels = loadLabelsFile("digitdata/traininglabels", n)
    for i in range(1):
        print(items[i])
        print(items[i].height)
        print(items[i].width)
        print(dir(items[i]))
        print(items[i].getPixels())

if __name__ == "__main__":
    _test()
