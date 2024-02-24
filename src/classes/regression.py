import joblib

from .ml_model import MLModel
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt


class Regression(MLModel):
     def __init__(self, file_path):
        self.data = pd.read_csv(file_path)
        self.X = self.data[['Methangehalt CH4', 'TS-Wert']]
        self.y = self.data[['BHKW2_Biogas']]
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=0.2, random_state=42)
        self.model = LinearRegression()

    def train(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def predict(self, X_test):
        predictions = self.model.predict(X_test)
        return predictions

    def evaluate(self, y_test, prediction):
        mse = mean_squared_error(y_test, prediction)
        return mse

    def save_model(self, file_path):
        # Save to file in the current working directory
        joblib_file = f"{file_path}.pkl"
        joblib.dump(self.model, joblib_file)

    def visualize_training_result(self):
        predictions_train = self.model.predict(self.X_train)
        plt.scatter(self.X_train['TS-Wert'], self.y_train, color='black', label='Actual Training Values')
        plt.scatter(self.X_train['TS-Wert'], predictions_train, color='blue', label='Predicted training values')
        plt.xlabel('TS-Wert')
        plt.ylabel('BHKW2_Biogas')
        plt.legend()
        plt.show()

    def visualize_test_result(self):
        predictions = self.model.predict(self.X_test)
        plt.scatter(self.X_test['TS-Wert'], self.y_test, color='black', label='Actual Test Values')
        plt.scatter(self.X_test['TS-Wert'], predictions, color='blue', label='Predicted test values')
        plt.xlabel('TS-Wert')
        plt.ylabel('BHKW2_Biogas')
        plt.legend()
        plt.show()

