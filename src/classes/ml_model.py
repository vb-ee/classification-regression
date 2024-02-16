# Import the Abstract Base Class package
from abc import ABC, abstractmethod


class MLModel(ABC):
    # Abstract class for machine learning model classes

    @abstractmethod
    def train(self, X_train, y_train):
        pass

    @abstractmethod
    def predict(self, X_test):
        pass

    @abstractmethod
    def evaluate(self, y_test, prediction):
        pass
