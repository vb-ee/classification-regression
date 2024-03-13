# built-in imports
from os.path import join, dirname
import sys

# third-party imports
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix, roc_curve, auc

# custom imports
from .ml_model import MLModel
from ..utils.constants import MODEL_FEATURE, DATA_PATH
from ..utils.data_pre_processing import standard_scaling

# Create a Classification class that inherits from MLModel


class Classification(MLModel):

    def __init__(self):
        '''
        This class use SVC model for calssification tasks.

        ex: Classification_model = Classification()
        '''
        self.data = pd.read_csv(
            join(dirname(dirname(dirname(__file__))), *DATA_PATH.CLASSIFICATION_RAW.value))
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.model = None
        self.prediction = None

    def class_model(self, kernel: str, C: float, gamma: float):
        '''
        user define the usage kernel 

        ex: Classification_model.class_model(kernel = 'rbf', C = 1.0, gamma = 0.1)
        '''
        self.model = SVC(kernel=kernel, C=C, gamma=gamma)

    def split_data(self, test_size: float):
        '''
        split the data by assigned test size, ranging from 0.05 to 0.3

        ex: Classification_model.data_split(test_size = 0.2)
        '''
        X = standard_scaling(self.data[MODEL_FEATURE.CLASSIFICATION_INPUT.value],
                             MODEL_FEATURE.CLASSIFICATION_INPUT.value).values
        y = self.data[MODEL_FEATURE.CLASSIFICATION_OUTPUT.value].values

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=test_size, random_state=42)

    def train(self):
        '''
        Train the Classification model on the provided training data.

        ex: Classification_model.train()
        '''
        # .ravel change the shape of y_train to 1-d array
        self.model.fit(self.X_train, self.y_train.ravel())

    def predict(self):
        '''
        Make predictions using the trained model.

        ex: prediction = Classification_model.predict()[0]
        '''
        self.prediction = [self.model.predict(
            self.X_train), self.model.predict(self.X_test)]

        return self.prediction

    def evaluate(self):
        '''
        Evaluate the performance of the model.

        :return: dictionary

        ex: train_accuracy = Classification_model.predict()['accuracy_train']
        '''
        accuracy_train = accuracy_score(
            self.y_train.ravel(), self.prediction[0])
        accuracy_test = accuracy_score(self.y_test.ravel(), self.prediction[1])

        precision_train = precision_score(
            self.y_train.ravel(), self.prediction[0])
        precision_test = precision_score(
            self.y_test.ravel(), self.prediction[1])

        recall_train = recall_score(self.y_train.ravel(), self.prediction[0])
        recall_test = recall_score(self.y_test.ravel(), self.prediction[1])

        f1_train = f1_score(self.y_train.ravel(), self.prediction[0])
        f1_test = f1_score(self.y_test.ravel(), self.prediction[1])

        return dict(accuracy_train=accuracy_train, accuracy_test=accuracy_test,
                    precision_train=precision_train, precision_test=precision_test,
                    recall_train=recall_train, recall_test=recall_test,
                    f1_train=f1_train, f1_test=f1_test)

    def save_model(self):
        '''
        Save the trained model to a file.

        ex: Classification_model.save_model()
        '''
        filepath = join(dirname(dirname(dirname(__file__))),
                        *DATA_PATH.CLASSIFICATION_TRAINED.value)
        joblib.dump(self.model, filepath)

    def get_confusion_matrix(self):
        '''
        Plot the confusion matrix for the classification model.

        :return: np.ndarray

        ex: cm = Classification_model.get_confusion_matrix()[0]
        '''
        return [confusion_matrix(self.y_train, self.prediction[0]),
                confusion_matrix(self.y_test, self.prediction[1])]
