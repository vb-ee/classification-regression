import pandas as pd
import joblib
from src.utils import MODEL_FEATURE, DataProcess, DATA_PATH
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt


class Regression:
    def __init__(self, file_path, test_size, window_size):
        dataprocess = DataProcess(file_path)
        self.data = dataprocess.centered_moving_average(MODEL_FEATURE.REGRESSION_INPUT, window_size)
        self.X = self.data[MODEL_FEATURE.REGRESSION_INPUT]
        self.y = self.data[MODEL_FEATURE.REGRESSION_OUTPUT]
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=test_size, random_state=42)
        self.model = LinearRegression()

    def train_model(self):
        self.model.fit(self.X_train, self.y_train)

    def evaluate_model(self):
        predictions = self.model.predict(self.X_test)
        mse = mean_squared_error(self.y_test, predictions)
        return mse
    
    def save_model(self, file_path):
        # Save to file in the current working directory
        joblib_file = f"{file_path}.pkl"
        joblib.dump(self.model, joblib_file)

    def visualize_training_result(self):
        predictions_train = self.model.predict(self.X_train)
        #BHKW1_Biogas = predictions_train.iloc[:, 0]
        #BHKW2_Biogas = predictions_train.iloc[:, 1]
        plt.scatter(self.X_train['TS-Wert'], self.y_train, color='black', label='Actual Training Values')
        plt.scatter(self.X_train['TS-Wert'], predictions_train.iloc[:, 0], color='blue', label='Predicted training values')
        plt.xlabel('TS-Wert')
        plt.ylabel('BHKW1_Biogas')
        plt.legend()
        plt.show()
        
        predictions_train = self.model.predict(self.X_train)
        plt.scatter(self.X_train['Methangehalt CH4'], self.y_train, color='black', label='Actual Training Values')
        plt.scatter(self.X_train['Methangehalt CH4'], predictions_train.iloc[:, 0], color='blue', label='Predicted training values')
        plt.xlabel('Methangehalt CH4')
        plt.ylabel('BHKW1_Biogas')
        plt.legend()
        plt.show()
        
        predictions_train = self.model.predict(self.X_train)
        plt.scatter(self.X_train['pH-Wert'], self.y_train, color='black', label='Actual Training Values')
        plt.scatter(self.X_train['pH-Wert'], predictions_train.iloc[:, 0], color='blue', label='Predicted training values')
        plt.xlabel('pH-Wert')
        plt.ylabel('BHKW1_Biogas')
        plt.legend()
        plt.show()
        
        predictions_train = self.model.predict(self.X_train)
        plt.scatter(self.X_train['TS-Wert'], self.y_train, color='black', label='Actual Training Values')
        plt.scatter(self.X_train['TS-Wert'], predictions_train.iloc[:, 1], color='blue', label='Predicted training values')
        plt.xlabel('TS-Wert')
        plt.ylabel('BHKW2_Biogas')
        plt.legend()
        plt.show()
        
        predictions_train = self.model.predict(self.X_train)
        plt.scatter(self.X_train['Methangehalt CH4'], self.y_train, color='black', label='Actual Training Values')
        plt.scatter(self.X_train['Methangehalt CH4'], predictions_train.iloc[:, 1], color='blue', label='Predicted training values')
        plt.xlabel('Methangehalt CH4')
        plt.ylabel('BHKW2_Biogas')
        plt.legend()
        plt.show()
        
        predictions_train = self.model.predict(self.X_train)
        plt.scatter(self.X_train['pH-Wert'], self.y_train, color='black', label='Actual Training Values')
        plt.scatter(self.X_train['pH-Wert'], predictions_train.iloc[:, 1], color='blue', label='Predicted training values')
        plt.xlabel('pH-Wert')
        plt.ylabel('BHKW2_Biogas')
        plt.legend()
        plt.show()

    def visualize_test_result(self):
        predictions = self.model.predict(self.X_test)
        plt.scatter(self.X_test['TS-Wert'], self.y_test, color='black', label='Actual Test Values')
        plt.scatter(self.X_test['TS-Wert'], predictions.iloc[:, 0], color='blue', label='Predicted test values')
        plt.xlabel('TS-Wert')
        plt.ylabel('BHKW1_Biogas')
        plt.legend()
        plt.show()
        
        predictions = self.model.predict(self.X_test)
        plt.scatter(self.X_test['Methangehalt CH4'], self.y_test, color='black', label='Actual Test Values')
        plt.scatter(self.X_test['Methangehalt CH4'], predictions.iloc[:, 0], color='blue', label='Predicted test values')
        plt.xlabel('Methangehalt CH4')
        plt.ylabel('BHKW1_Biogas')
        plt.legend()
        plt.show()
        
        predictions = self.model.predict(self.X_test)
        plt.scatter(self.X_test['pH-Wert'], self.y_test, color='black', label='Actual Test Values')
        plt.scatter(self.X_test['pH-Wert'], predictions.iloc[:, 0], color='blue', label='Predicted test values')
        plt.xlabel('pH-Wert')
        plt.ylabel('BHKW1_Biogas')
        plt.legend()
        plt.show()
        
        predictions = self.model.predict(self.X_test)
        plt.scatter(self.X_test['TS-Wert'], self.y_test, color='black', label='Actual Test Values')
        plt.scatter(self.X_test['TS-Wert'], predictions.iloc[:, 1], color='blue', label='Predicted test values')
        plt.xlabel('TS-Wert')
        plt.ylabel('BHKW2_Biogas')
        plt.legend()
        plt.show()
        
        predictions = self.model.predict(self.X_test)
        plt.scatter(self.X_test['Methangehalt CH4'], self.y_test, color='black', label='Actual Test Values')
        plt.scatter(self.X_test['Methangehalt CH4'], predictions.iloc[:, 1], color='blue', label='Predicted test values')
        plt.xlabel('Methangehalt CH4')
        plt.ylabel('BHKW2_Biogas')
        plt.legend()
        plt.show()
        
        predictions = self.model.predict(self.X_test)
        plt.scatter(self.X_test['pH-Wert'], self.y_test, color='black', label='Actual Test Values')
        plt.scatter(self.X_test['pH-Wert'], predictions.iloc[:, 1], color='blue', label='Predicted test values')
        plt.xlabel('pH-Wert')
        plt.ylabel('BHKW2_Biogas')
        plt.legend()
        plt.show()

# Example of class and do operations
biogas_model = Regression(DATA_PATH.REGRESSION_RAW, test_size=0.2, window_size=20)
biogas_model.train_model()
mse=biogas_model.evaluate_model()
print(mse)
biogas_model.visualize_training_result()
biogas_model.visualize_test_result()
