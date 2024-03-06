# built-in imports
from os.path import join, dirname

# third-party imports
import joblib
import pandas as pd
from sklearn.metrics import mean_squared_error, root_mean_squared_error, r2_score, explained_variance_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

# custom imports
from src.classes.ml_model import MLModel
from src.utils import DATA_PATH, MODEL_FEATURE, MODEL_RESULT_MODE


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
        self.X = self.data[MODEL_FEATURE.REGRESSION_INPUT.value]
        self.Y = self.data[MODEL_FEATURE.REGRESSION_OUTPUT.value]
        self.Y_test = None
        self.Y_train = None
        self.X_test = None
        self.X_train = None
        self.X_test_poly = None
        self.X_train_poly = None

    def get_polynomial_order(self, degree: int = 4):
        self.poly_features = PolynomialFeatures(degree=degree)

    def split_data(self, test_size: float = 0.2):
        self.X_train, self.X_test, self.Y_train, self.Y_test = train_test_split(
            self.X, self.Y, test_size=test_size, random_state=42)

    def train(self):
        """
        train the model
        """
        self.X_train_poly = self.poly_features.fit_transform(self.X_train)
        self.model.fit(self.X_train_poly, self.Y_train)

    def predict(self):
        """
        predict the value of the output

        :return: Dictionary

        ex: predict = re.predict()
        """
        self.X_test_poly = self.poly_features.fit_transform(self.X_test)
        self.prediction = dict(train=self.model.predict(self.X_train_poly),
                               test=self.model.predict(self.X_test_poly))

    def evaluate(self):
        """
        evaluate the mean_squared_error and root_mean_squared_error for both train and test

        :return: Dictionary

        ex: evaluate = re.evaluate(predict)
        """
        return dict(train=dict(mse=mean_squared_error(self.Y_train, self.prediction[MODEL_RESULT_MODE.TRAIN.value]),
                               rmse=root_mean_squared_error(self.Y_train, self.prediction[MODEL_RESULT_MODE.TRAIN.value]),
                               r2=r2_score(self.Y_train, self.prediction[MODEL_RESULT_MODE.TRAIN.value]),
                               evs=explained_variance_score(self.Y_train, self.prediction[MODEL_RESULT_MODE.TRAIN.value])),
                    test=dict(mse=mean_squared_error(self.Y_test, self.prediction[MODEL_RESULT_MODE.TEST.value]),
                              rmse=root_mean_squared_error(self.Y_test, self.prediction[MODEL_RESULT_MODE.TEST.value]),
                              r2=r2_score(self.Y_test, self.prediction[MODEL_RESULT_MODE.TEST.value]),
                              evs=explained_variance_score(self.Y_test, self.prediction[MODEL_RESULT_MODE.TEST.value])))

    def save_model(self):
        """
        save the model in the dictionary: data/model/regression, filename is regression.mo

        ex: re.save_model()
        """
        filepath = join(dirname(dirname(dirname(__file__))),
                        *DATA_PATH.REGRESSION_TRAINED.value)
        joblib.dump(self.model, filepath)
