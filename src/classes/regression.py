# import built-in packages
from os.path import dirname, join

# import third party packages
from sklearn.metrics import mean_squared_error, root_mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import joblib

from ml_model import MLModel
from src.utils import DATA_PATH, MODEL_FEATURE
from src.utils.data_pre_processing import DataProcess


class Regression(MLModel):
    """
    class for regression, inherit MLModel

    pass test_size, data_pre_process, window_size when create an object
    by default use raw data for regression, if set data_pre_process=True, then use filtered data

    ex: re = Regression(0.2, True, 80)

    """

    def __init__(self, test_size=0.2, data_pre_process=False, window_size=85):
        self.model = LinearRegression()
        self.dp = DataProcess(DATA_PATH.REGRESSION_RAW.value)
        self.X = self.dp.data[MODEL_FEATURE.REGRESSION_INPUT.value]
        self.Y = self.dp.data[MODEL_FEATURE.REGRESSION_OUTPUT.value]
        if data_pre_process:
            self.X = self.dp.centered_moving_average(
                MODEL_FEATURE.REGRESSION_INPUT.value, window_size)
        self.X_train, self.X_test, self.Y_train, self.Y_test = train_test_split(
            self.X, self.Y, test_size=test_size, random_state=42)

    def train(self):
        """
        train the model
        """
        self.model.fit(self.X_train, self.Y_train)

    def predict(self):
        """
        predict the value of the output

        :return: Dictionary

        ex: predict = re.predict()
        """
        prediction = {
            'train': self.model.predict(
                self.X_train), 'test': self.model.predict(
                self.X_test)}
        return prediction

    def evaluate(self, prediction):
        """
        evaluate the mean_squared_error and root_mean_squared_error for both train and test

        :return: Dictionary

        ex: evaluate = re.evaluate(predict)
        """
        evaluation = {
            'train': {
                'mse': mean_squared_error(
                    self.Y_train, prediction['train']), 'rmse': root_mean_squared_error(
                    self.Y_train, prediction['train'])}, 'test': {
                'mse': mean_squared_error(
                    self.Y_test, prediction['test']), 'rmse': root_mean_squared_error(
                    self.Y_test, prediction['test'])}}
        return evaluation

    def save_model(self):
        """
        save the model in the dictionary: data/model/regression, filename is regression.mo

        ex: re.save_model()
        """
        filepath = join(dirname(dirname(dirname(__file__))),
                        *DATA_PATH.REGRESSION_PROCESSED.value)
        joblib.dump(self.model, filepath)
