# Import the Abstract Base Class package
from abc import ABC, abstractmethod


class MLModel(ABC):
    # Abstract class for machine learning model classes

    @abstractmethod
    def split_data(self, test_size):
        pass

    @abstractmethod
    def train(self, X_train, y_train):
        pass

    @abstractmethod
    def predict(self, X_test):
        pass

    @abstractmethod
    def evaluate(self, y_test, prediction):
        pass

    @abstractmethod
    def save_model(self):
        pass
