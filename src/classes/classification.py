# built-in imports
from os.path import join, dirname

# third-party imports
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix

# custom imports
from .ml_model import MLModel
from ..utils.constants import MODEL_FEATURE, DATA_PATH, MODEL_RESULT_MODE


class Classification(MLModel):

    def __init__(self):
        """
        This class use SVC model for classification tasks.

        ex: Classification_model = Classification()
        """
        self.data = pd.read_csv(
            join(dirname(dirname(dirname(__file__))), *DATA_PATH.CLASSIFICATION_RAW.value))
        self.X_train = None
        self.X_test = None
        self.Y_train = None
        self.Y_test = None
        self.model = None
        self.prediction = None
        self.evaluation = None

    def split_data(self, test_size: float):
        """
        split the data by assigned test size, ranging from 0.05 to 0.3

        ex: Classification_model.data_split(test_size = 0.2)
        """
        X = self.data[MODEL_FEATURE.CLASSIFICATION_INPUT.value]
        y = self.data[MODEL_FEATURE.CLASSIFICATION_OUTPUT.value]

        self.X_train, self.X_test, self.Y_train, self.Y_test = train_test_split(
            X, y, test_size=test_size, random_state=42)

    def _get_scalar(self):
        """
        return scalar which is trained by X_train
        move it here because I want to keep the raw X_train and X_test and use them in GUI
        """
        scaler = StandardScaler()
        scaler.fit(self.X_train)
        return scaler

    def train(self, kernel: str = 'linear'):
        """
        Train the Classification model on the scaled X_train

        param: kernel: string, the parameter obtained from GUI

        ex: Classification_model.train()
        """
        # .ravel change the shape of Y_train to 1-d array
        kernel_params = {
            'poly': {'gamma': 0.01, 'C': 100},
            'rbf': {'gamma': 0.1, 'C': 1000},
            'sigmoid': {'gamma': 0.01 , 'C': 10}
            }

        # Set kernel parameters based on input kernel type
        if kernel in kernel_params:
            params = kernel_params[kernel]
        self.model = SVC(kernel=kernel, **params)

        X_train_scaled = self._get_scalar().transform(self.X_train)
        self.model.fit(X_train_scaled, self.Y_train.values.ravel())

    def predict(self):
        """
        Make predictions using the trained model and scaled X_test
        prediction is a dictionary with two keys: train and test

        ex: Classification_model.predict()
        """
        X_train_scaled = self._get_scalar().transform(self.X_train)
        X_test_scaled = self._get_scalar().transform(self.X_test)
        self.prediction = dict(train=self.model.predict(X_train_scaled),
                               test=self.model.predict(X_test_scaled))

    def evaluate(self):
        """
        Evaluate the performance of the model.

        evaluation is a dictionary with two keys: train and test, the values are Dataframe
        each Dataframe contains 4 columns: accuracy, precision, recall, f1

        ex:  Classification_model.predict()
        """
        train = {
            'accuracy': [accuracy_score(self.Y_train.values.ravel(), self.prediction[MODEL_RESULT_MODE.TRAIN.value])],
            'precision': [precision_score(
                self.Y_train.values.ravel(), self.prediction[MODEL_RESULT_MODE.TRAIN.value], zero_division=0.0)],
            'recall': [recall_score(self.Y_train.values.ravel(), self.prediction[MODEL_RESULT_MODE.TRAIN.value])],
            'f1': [f1_score(self.Y_train.values.ravel(), self.prediction[MODEL_RESULT_MODE.TRAIN.value])]
        }
        test = {
            'accuracy': [accuracy_score(self.Y_test.values.ravel(), self.prediction[MODEL_RESULT_MODE.TEST.value])],
            'precision': [precision_score(
                self.Y_test.values.ravel(), self.prediction[MODEL_RESULT_MODE.TEST.value], zero_division=0.0)],
            'recall': [recall_score(self.Y_test.values.ravel(), self.prediction[MODEL_RESULT_MODE.TEST.value])],
            'f1': [f1_score(self.Y_test.values.ravel(), self.prediction[MODEL_RESULT_MODE.TEST.value])]
        }

        self.evaluation = dict(train=pd.DataFrame(train), test=pd.DataFrame(test))

    def get_confusion_matrix(self):
        """
        Plot the confusion matrix for the classification model.

        :return: np.ndarray

        ex: cm = Classification_model.get_confusion_matrix()
        """
        return dict(train=confusion_matrix(self.Y_train, self.prediction[MODEL_RESULT_MODE.TRAIN.value]),
                    test=confusion_matrix(self.Y_test, self.prediction[MODEL_RESULT_MODE.TEST.value]))
