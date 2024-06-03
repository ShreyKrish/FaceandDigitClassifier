# dataClassifier.py
# -----------------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# For more info, see http://inst.eecs.berkeley.edu/~cs188/sp09/pacman.html


import perceptron
import samples
import sys
import util
import neuralNet
import math
import numpy as np
from scipy.ndimage import label, generate_binary_structure


TEST_SET_SIZE = 100
DIGIT_DATUM_WIDTH = 28
DIGIT_DATUM_HEIGHT = 28
FACE_DATUM_WIDTH = 60
FACE_DATUM_HEIGHT = 70


def basicFeatureExtractorDigit(datum):
   features = util.Counter()
   for x in range(DIGIT_DATUM_WIDTH):
       for y in range(DIGIT_DATUM_HEIGHT):
           features[(x, y)] = datum.getPixel(x, y) > 0
   return features


def basicFeatureExtractorFace(datum):
   features = util.Counter()
   for x in range(FACE_DATUM_WIDTH):
       for y in range(FACE_DATUM_HEIGHT):
           features[(x, y)] = datum.getPixel(x, y) > 0
   return features


def calculate_bounding_box(binary_image):
   """Calculate the bounding box ratio of the digit."""
   rows = np.any(binary_image, axis=1)
   cols = np.any(binary_image, axis=0)
   rmin, rmax = np.where(rows)[0][[0, -1]]
   cmin, cmax = np.where(cols)[0][[0, -1]]
   height = rmax - rmin + 1
   width = cmax - cmin + 1
   return height / width if width > 0 else 0


def calculate_center_of_mass(binary_image):
   """Calculate the center of mass of the digit."""
   total = binary_image.sum()
   x, y = np.indices(binary_image.shape)
   x_center = (x * binary_image).sum() / total if total > 0 else 0
   y_center = (y * binary_image).sum() / total if total > 0 else 0
   return x_center, y_center


def count_components_and_holes(binary_image):
   """Count connected components and holes within those components."""
   structure = generate_binary_structure(2, 2)
   labeled, ncomponents = label(binary_image, structure)
   # Invert image to find holes
   holes_image = np.invert(binary_image)
   holes_image[labeled == 0] = False  # Exclude the background
   _, nholes = label(holes_image, structure)
   return ncomponents, nholes


def enhancedFeatureExtractorDigit(datum):
   features = basicFeatureExtractorDigit(datum)
  
   # Convert datum to binary image array
   binary_image = np.array([[datum.getPixel(x, y) > 0 for y in range(DIGIT_DATUM_HEIGHT)] for x in range(DIGIT_DATUM_WIDTH)])
  
   # Compute features using helper functions
   features['bounding_box_ratio'] = calculate_bounding_box(binary_image)
   x_center, y_center = calculate_center_of_mass(binary_image)
   features['center_of_mass_x'] = x_center
   features['center_of_mass_y'] = y_center
   components, holes = count_components_and_holes(binary_image)
   features['connectivity'] = components
   features['holes_count'] = holes
  
   return features


def enhancedFeatureExtractorFace(datum):
   features = basicFeatureExtractorFace(datum)
   # Example: add a feature for edge detection (Sobel)
   for x in range(1, FACE_DATUM_WIDTH - 1):
       for y in range(1, FACE_DATUM_HEIGHT - 1):
           Gx = (datum.getPixel(x+1, y-1) + 2*datum.getPixel(x+1, y) + datum.getPixel(x+1, y+1)) - (datum.getPixel(x-1, y-1) + 2*datum.getPixel(x-1, y) + datum.getPixel(x-1, y+1))
           Gy = (datum.getPixel(x-1, y+1) + 2*datum.getPixel(x, y+1) + datum.getPixel(x+1, y+1)) - (datum.getPixel(x-1, y-1) + 2*datum.getPixel(x, y-1) + datum.getPixel(x+1, y-1))
           edge_magnitude = np.sqrt(Gx**2 + Gy**2)
           features[(x, y, 'edge_magnitude')] = edge_magnitude
   return features


def readCommand(argv):
   from argparse import ArgumentParser
   parser = ArgumentParser(description='Run classifiers on digit and face data using various extraction methods.')


   parser.add_argument('-c', '--classifier', help='The type of classifier [Default: %(default)s]', choices=['perceptron', 'neuralNet'], default='perceptron')
   parser.add_argument('-d', '--data', help='Dataset to use [Default: %(default)s]', choices=['digits', 'faces'], default='digits')
   parser.add_argument('-t', '--training', help='The size of the training set [Default: %(default)s]', default=100, type=int)
   parser.add_argument('-e', '--enhanced', help='Use enhanced features [Default: %(default)s]', action='store_true', default=False)


   options = parser.parse_args(argv)
   args = {}


   print("Doing classification")
   print("--------------------")
   print("Data:\t\t" + options.data)
   print("Classifier:\t\t" + options.classifier)
   print("Training set size:\t" + str(options.training))
   print("Using enhanced features:\t" + str(options.enhanced))


   if options.data == "digits":
       featureFunction = enhancedFeatureExtractorDigit if options.enhanced else basicFeatureExtractorDigit
   elif options.data == "faces":
       featureFunction = enhancedFeatureExtractorFace if options.enhanced else basicFeatureExtractorFace
   else:
       print("Unknown dataset", options.data)
       sys.exit(2)


   if options.data == "digits":
       legalLabels = list(range(10))
   else:
       legalLabels = list(range(2))


   if options.training <= 0:
       print("Training set size should be a positive integer (you provided: %d)" % options.training)
       sys.exit(2)


   if options.classifier == "perceptron":
       classifier = perceptron.PerceptronClassifier(legalLabels, 3)  # Assume default 3 iterations
   elif options.classifier == "neuralNet":
       classifier = neuralNet.NeuralNetworkClassifier(legalLabels)
   args['classifier'] = classifier
   args['featureFunction'] = featureFunction
   args['printImage'] = lambda image: image  # Replace this with actual function to print image if needed


   return args, options


def analysis(classifier, guesses, testLabels, testData, rawTestData, printImage):
   print("Analysis of the results")
   errors = []


   for i in range(len(guesses)):
       predicted = guesses[i]
       truth = testLabels[i]


       if predicted == truth:
           if len(errors) < 5:  # Only print a few correct classifications
               print("===================================")
               print("Correctly classified example #%d" % i)
               print("Predicted: %d; Truth: %d" % (predicted, truth))
               printImage(rawTestData[i].getPixels())
       else:
           errors.append((i, predicted, truth))
           if len(errors) <= 5:  # Limit to first few errors
               print("===================================")
               print("Misclassified example #%d" % i)
               print("Predicted: %d; Truth: %d" % (predicted, truth))
               printImage(rawTestData[i].getPixels())


   # Example: Calculate and print overall accuracy
   accuracy = float(sum(1 for i in range(len(guesses)) if guesses[i] == testLabels[i])) / len(guesses)
   print("Overall accuracy: %.2f%%" % (accuracy * 100))


def runClassifier(args, options):
    featureFunction = args['featureFunction']
    classifier = args['classifier']
    printImage = args['printImage']

    # Load data  
    numTraining = options.training
    numTest = TEST_SET_SIZE

    if(options.data == "faces"):
        rawTrainingData = samples.loadDataFile("facedata/facedatatrain", numTraining, FACE_DATUM_WIDTH, FACE_DATUM_HEIGHT)
        trainingLabels = samples.loadLabelsFile("facedata/facedatatrainlabels", numTraining)
        rawTestData = samples.loadDataFile("facedata/facedatatest", numTest, FACE_DATUM_WIDTH, FACE_DATUM_HEIGHT)
        testLabels = samples.loadLabelsFile("facedata/facedatatestlabels", numTest)
    else:
        rawTrainingData = samples.loadDataFile("digitdata/trainingimages", numTraining, DIGIT_DATUM_WIDTH, DIGIT_DATUM_HEIGHT)
        trainingLabels = samples.loadLabelsFile("digitdata/traininglabels", numTraining)
        rawTestData = samples.loadDataFile("digitdata/testimages", numTest, DIGIT_DATUM_WIDTH, DIGIT_DATUM_HEIGHT)
        testLabels = samples.loadLabelsFile("digitdata/testlabels", numTest)

    # Extract features
    print("Extracting features...")
    trainingData = list(map(featureFunction, rawTrainingData))
    testData = list(map(featureFunction, rawTestData))

    # Conduct training and testing
    print("Training...")
    classifier.train(trainingData, trainingLabels, None, None)
    print("Testing...")
    guesses = classifier.classify(testData)
    correct = sum(guesses[i] == testLabels[i] for i in range(len(testLabels)))
    print("%d correct out of %d (%.1f%%)." % (correct, len(testLabels), 100.0 * correct / len(testLabels)))
    analysis(classifier, guesses, testLabels, testData, rawTestData, printImage)

if __name__ == '__main__':
    args, options = readCommand(sys.argv[1:])  # Get game components based on input
    runClassifier(args, options)