from models.random_forest import RFMnistClassifier
from models.neural_network import NNMnistClassifier
from models.cnn import CNNMnistClassifier

"""
The MnistClassifier class abstracts the model selection process, allowing users to specify the algorithm and use a unified interface for training and prediction.
"""
class MnistClassifier:
    def __init__(self, algorithm, **kwargs):
        self.algorithm = algorithm
        if algorithm == 'rf':
            self.model = RFMnistClassifier(**kwargs)
        elif algorithm == 'nn':
            self.model = NNMnistClassifier(**kwargs)
        elif algorithm == 'cnn':
            self.model = CNNMnistClassifier(**kwargs)
        else:
            raise ValueError("Invalid algorithm. Choose 'rf', 'nn' or 'cnn'.")
        
    def train(self, X_train, y_train):
        self.model.train(X_train, y_train)

    def predict(self, X_test):
        return self.model.predict(X_test)