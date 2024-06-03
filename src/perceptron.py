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
    def __init__(self, legalLabels, max_iterations):
        self.legalLabels = legalLabels
        self.type = "perceptron"
        self.max_iterations = max_iterations
        self.weights = {}
        for label in legalLabels:
            self.weights[label] = util.Counter()  # Initialize weights for each label

    def setWeights(self, weights):
        assert len(weights) == len(self.legalLabels)
        self.weights = weights

    def train(self, trainingData, trainingLabels, validationData, validationLabels):
        """
        This method trains the perceptron. For each iteration, the method loops over each data point,
        predicts a label, and updates the model if the prediction is wrong.
        """
        for iteration in range(self.max_iterations):
            print("Starting iteration", iteration, "...")
            for i in range(len(trainingData)):
                current_features = trainingData[i]
                actual_label = trainingLabels[i]
                predicted_label = None
                highest_score = float('-inf')

                # Predict the label for the current datum
                for label in self.legalLabels:
                    score = current_features * self.weights[label]
                    if score > highest_score:
                        highest_score = score
                        predicted_label = label

                # Update weights if prediction is wrong
                if predicted_label != actual_label:
                    self.weights[actual_label] += current_features
                    self.weights[predicted_label] -= current_features

    def classify(self, data):
        """
        This method classifies each datum as the label that most closely matches the prototype vector
        for that label.
        """
        guesses = []
        for datum in data:
            vectors = util.Counter()
            for l in self.legalLabels:
                vectors[l] = self.weights[l] * datum
            guesses.append(vectors.argMax())
        return guesses

    def findHighWeightFeatures(self, label):
        """
        Returns a list of the 100 features with the greatest weight for the given label.
        """
        featuresWeights = self.weights[label]
        # Sort the features by weight in descending order and select the top 100
        topFeatures = featuresWeights.sortedKeys()[:100]
        return topFeatures