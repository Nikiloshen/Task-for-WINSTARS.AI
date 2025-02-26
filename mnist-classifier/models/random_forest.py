from sklearn.ensemble import RandomForestClassifier
from models.interface import MnistClassifierInterface

"""
Random Forest Classifier for MNIST dataset using scikit-learn.
"""
class RFMnistClassifier(MnistClassifierInterface):
    def __init__(self, n_estimators=100):
        self.model = RandomForestClassifier(n_estimators=n_estimators, random_state=42, n_jobs=-1)

    def train(self, X_train, y_train):
        X_flat = X_train.reshape((X_train.shape[0], -1))
        self.model.fit(X_flat, y_train)

    def predict(self, X_test):
        X_flat = X_test.reshape((X_test.shape[0], -1))
        return self.model.predict(X_flat)