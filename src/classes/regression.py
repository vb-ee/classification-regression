# built-in imports
from os.path import join, dirname

# third-party imports
import pandas as pd
from sklearn.metrics import mean_squared_error, root_mean_squared_error, r2_score, explained_variance_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

# custom imports
from .ml_model import MLModel
from ..utils import DATA_PATH, MODEL_FEATURE, MODEL_RESULT_MODE


class Regression(MLModel):
    """
    class for regression, inherit MLModel

    pass test_size, data_pre_process, window_size when create an object
    by default use raw data for regression, if set data_pre_process=True, then use filtered data

    ex: re = Regression(0.2, True, 80)

    """

    def __init__(self):
        self.data = pd.read_csv(
            join(dirname(dirname(dirname(__file__))), *DATA_PATH.REGRESSION_RAW.value))
        self.model = LinearRegression()
        self.poly_features = None
        self.prediction = None
        self.Y_test = None
        self.Y_train = None
        self.X_test = None
        self.X_train = None
        self.X_test_poly = None
        self.X_train_poly = None
        self.evaluation = []

    def split_data(self, test_size: float = 0.2):
        """
        split the data according to the test_size obtained from GUI
        """
        X = self.data[MODEL_FEATURE.REGRESSION_INPUT.value]
        Y = self.data[MODEL_FEATURE.REGRESSION_OUTPUT.value]

        self.X_train, self.X_test, self.Y_train, self.Y_test = train_test_split(
            X, Y, test_size=test_size, random_state=42)

    def train(self, degree: int = 2):
        """
        obtain the order of the polynomial regression and train the model
        """
        self.poly_features = PolynomialFeatures(degree=degree)
        self.X_train_poly = self.poly_features.fit_transform(self.X_train)
        self.model.fit(self.X_train_poly, self.Y_train)

    def predict(self):
        """
        predict the value of the output, prediction contains two keys: train and test
        """
        self.X_test_poly = self.poly_features.fit_transform(self.X_test)
        self.prediction = dict(train=self.model.predict(self.X_train_poly),
                               test=self.model.predict(self.X_test_poly))

    def evaluate(self):
        """
        evaluate the mean_squared_error and root_mean_squared_error for both train and test

        evaluation is a List of two dictionaries, the first is for biogas1, the second is for biogas2
            every dictionary includes 2 keys: train and test,
            values are Dataframes containing 4 columns: mean_squared_error, root_mean_squared_error,
            r2_score, explained_variance_score

        ex: evaluate = re.evaluate(predict)
        """
        for i, col in enumerate(self.Y_test.columns.values):
            train = {
                'mean_squared_error': [
                    int(mean_squared_error(self.Y_train[col], self.prediction[MODEL_RESULT_MODE.TRAIN.value][:, i]))],
                'root_mean_squared_error': [
                    int(root_mean_squared_error(
                        self.Y_train[col], self.prediction[MODEL_RESULT_MODE.TRAIN.value][:, i]))],
                'r2_score': [r2_score(self.Y_train[col], self.prediction[MODEL_RESULT_MODE.TRAIN.value][:, i])],
                'explained_variance_score': [
                    explained_variance_score(self.Y_train[col], self.prediction[MODEL_RESULT_MODE.TRAIN.value][:, i])]
            }
            test = {
                'mean_squared_error': [
                    int(mean_squared_error(self.Y_test[col], self.prediction[MODEL_RESULT_MODE.TEST.value][:, i]))],
                'root_mean_squared_error': [
                    int(root_mean_squared_error(
                        self.Y_test[col], self.prediction[MODEL_RESULT_MODE.TEST.value][:, i]))],
                'r2_score': [r2_score(self.Y_test[col], self.prediction[MODEL_RESULT_MODE.TEST.value][:, i])],
                'explained_variance_score': [
                    explained_variance_score(self.Y_test[col], self.prediction[MODEL_RESULT_MODE.TEST.value][:, i])]
            }

            self.evaluation.append(
                dict(train=pd.DataFrame(train), test=pd.DataFrame(test)))
