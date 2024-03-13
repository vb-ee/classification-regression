# built-in imports
from os.path import join, dirname

# third-party imports
import pandas as pd

# custom imports
from .ml_model import MLModel
from ..utils import DATA_PATH


class Classification(MLModel):
    def __init__(self):
        self.data = pd.read_csv(
            join(dirname(dirname(dirname(__file__))), *DATA_PATH.CLASSIFICATION_RAW.value))

    def split_data(self, test_size):
        pass

    def train(self, X_train, y_train):
        pass

    def predict(self, X_test):
        pass

    def evaluate(self, y_test, prediction):
        pass

    def save_model(self):
        pass
