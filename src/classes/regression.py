from os.path import dirname, join

from sklearn.metrics import mean_squared_error, root_mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

from ml_model import MLModel
from src.utils import DATA_PATH, MODEL_FEATURE
from src.utils.data_pre_processing import DataProcess
import joblib


class Regression(MLModel):
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
        self.model.fit(self.X_train, self.Y_train)

    def predict(self):
        return self.model.predict(self.X_test)

    def evaluate(self, prediction):
        mse = mean_squared_error(self.Y_test, prediction)
        rmse = root_mean_squared_error(self.Y_test, prediction)
        return mse, rmse

    def save_model(self):
        filepath = join(dirname(dirname(dirname(__file__))),
                        *DATA_PATH.REGRESSION_PROCESSED.value)
        print(filepath)
        joblib.dump(self.model, filepath)
