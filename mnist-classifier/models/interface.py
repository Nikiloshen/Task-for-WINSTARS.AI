from abc import ABC, abstractmethod # ABC --> Abstract Base Class

"""
Interface Definition. 
We use Abstract Base Class (ABC) to define the interface and a contract for all models. 
It specifies that any class implementing this interface must provide implementations
for the train and predict methods.
"""
class MnistClassifierInterface(ABC):
    @abstractmethod
    def train(self, X_train, y_train):
        pass

    @abstractmethod
    def predict(self, X_test):
        pass