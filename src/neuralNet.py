import numpy as np
from classificationMethod import ClassificationMethod
from util import Counter

class NeuralNetworkClassifier(ClassificationMethod):
    def __init__(self, legalLabels, hidden_units=500, learning_rate=0.1, epochs=100):
        super(NeuralNetworkClassifier, self).__init__(legalLabels)
        self.hidden_units = hidden_units
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.weights1 = None
        self.weights2 = None
        self.output_units = len(legalLabels)

    def initialize_weights(self, input_size):
        # He initialization for weights of ReLU activation
        self.weights1 = np.random.randn(input_size, self.hidden_units) * np.sqrt(2. / input_size)
        self.weights2 = np.random.randn(self.hidden_units, self.output_units) * np.sqrt(2. / self.hidden_units)

    def forward_propagation(self, data):
        # Check and print data shape for debugging
        # print("Data shape entering forward_propagation:", data.shape)
        # print("Weights1 shape:", self.weights1.shape)
        self.z1 = np.dot(data, self.weights1)
        self.a1 = np.maximum(0, self.z1)  # ReLU
        self.z2 = np.dot(self.a1, self.weights2)
        self.a2 = np.exp(self.z2) / np.sum(np.exp(self.z2), axis=1, keepdims=True)
        return self.a2

    def compute_cost(self, predictions, labels):
        # Cross-entropy loss
        correct_log_probs = -np.log(predictions[range(len(labels)), labels])
        loss = np.sum(correct_log_probs) / len(labels)
        return loss

    def backward_propagation(self, data, labels):
        num_examples = data.shape[0]
        delta3 = self.a2
        delta3[range(num_examples), labels] -= 1
        dW2 = (self.a1.T).dot(delta3)
        dW1 = (data.T).dot((delta3.dot(self.weights2.T)) * (self.z1 > 0))  # Only backpropagate errors where ReLU is active

        # Gradient descent parameter update
        self.weights1 -= self.learning_rate * dW1 / num_examples
        self.weights2 -= self.learning_rate * dW2 / num_examples

    def train(self, trainingData, trainingLabels, validationData, validationLabels):
        trainingData = np.array([list(d.values()) for d in trainingData])
        trainingLabels = np.array(trainingLabels)
        self.initialize_weights(trainingData.shape[1])
        for epoch in range(self.epochs):
            predictions = self.forward_propagation(trainingData)
            loss = self.compute_cost(predictions, trainingLabels)
            self.backward_propagation(trainingData, trainingLabels)
            # print("Epoch %d: Loss %f" % (epoch, loss))

    def classify(self, data):
        # Ensure data is in numpy array form and correctly shaped
        data = np.array([list(d.values()) for d in data])
        if data.ndim == 1:  # This is a check to reshape the data if it's incorrectly flattened
            data = data.reshape(1, -1)  # Reshape to 2D array with one instance per row
        output_probs = self.forward_propagation(data)
        predictions = np.argmax(output_probs, axis=1)
        return predictions