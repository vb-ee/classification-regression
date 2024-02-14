from abc import ABC, abstractmethod

# Abstract class for machine learning model classes
class MLModel(ABC):
    @abstractmethod
    def train(self, X_train, y_train):
        pass
    
    @abstractmethod
    def predict(self, X_test):
        pass
    
    @abstractmethod
    def evaluate(self, y_test, prediction):
        pass