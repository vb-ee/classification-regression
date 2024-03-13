# Import the Abstract Base Class package
from abc import ABC, abstractmethod


class MLModel(ABC):
    # Abstract class for machine learning model classes

    @abstractmethod
    def split_data(self, test_size):
        pass

    @abstractmethod
    def train(self):
        pass

    @abstractmethod
    def predict(self):
        pass

    @abstractmethod
    def evaluate(self):
        pass
