import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt


class Regression:
    def __init__(self, file_path):
        self.data = pd.read_csv(file_path)
        self.X = self.data[['Methangehalt CH4', 'TS-Wert', 'pH-Wert']]
        self.y = self.data[['BHKW2_Biogas']]
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=0.2, random_state=42)
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
        plt.scatter(self.X_train['TS-Wert'], self.y_train, color='black', label='Actual Training Values')
        plt.scatter(self.X_train['TS-Wert'], predictions_train, color='blue', label='Predicted training values')
        plt.xlabel('TS-Wert')
        plt.ylabel('BHKW1_Biogas')
        plt.legend()
        plt.show()
        
        predictions_train = self.model.predict(self.X_train)
        plt.scatter(self.X_train['Methangehalt CH4'], self.y_train, color='black', label='Actual Training Values')
        plt.scatter(self.X_train['Methangehalt CH4'], predictions_train, color='blue', label='Predicted training values')
        plt.xlabel('Methangehalt CH4')
        plt.ylabel('BHKW1_Biogas')
        plt.legend()
        plt.show()
        
        predictions_train = self.model.predict(self.X_train)
        plt.scatter(self.X_train['pH-Wert'], self.y_train, color='black', label='Actual Training Values')
        plt.scatter(self.X_train['pH-Wert'], predictions_train, color='blue', label='Predicted training values')
        plt.xlabel('pH-Wert')
        plt.ylabel('BHKW1_Biogas')
        plt.legend()
        plt.show()
        
        predictions_train = self.model.predict(self.X_train)
        plt.scatter(self.X_train['TS-Wert'], self.y_train, color='black', label='Actual Training Values')
        plt.scatter(self.X_train['TS-Wert'], predictions_train, color='blue', label='Predicted training values')
        plt.xlabel('TS-Wert')
        plt.ylabel('BHKW2_Biogas')
        plt.legend()
        plt.show()
        
        predictions_train = self.model.predict(self.X_train)
        plt.scatter(self.X_train['Methangehalt CH4'], self.y_train, color='black', label='Actual Training Values')
        plt.scatter(self.X_train['Methangehalt CH4'], predictions_train, color='blue', label='Predicted training values')
        plt.xlabel('Methangehalt CH4')
        plt.ylabel('BHKW2_Biogas')
        plt.legend()
        plt.show()
        
        predictions_train = self.model.predict(self.X_train)
        plt.scatter(self.X_train['pH-Wert'], self.y_train, color='black', label='Actual Training Values')
        plt.scatter(self.X_train['pH-Wert'], predictions_train, color='blue', label='Predicted training values')
        plt.xlabel('pH-Wert')
        plt.ylabel('BHKW2_Biogas')
        plt.legend()
        plt.show()

    def visualize_test_result(self):
        predictions = self.model.predict(self.X_test)
        plt.scatter(self.X_test['TS-Wert'], self.y_test, color='black', label='Actual Test Values')
        plt.scatter(self.X_test['TS-Wert'], predictions, color='blue', label='Predicted test values')
        plt.xlabel('TS-Wert')
        plt.ylabel('BHKW1_Biogas')
        plt.legend()
        plt.show()
        
        predictions = self.model.predict(self.X_test)
        plt.scatter(self.X_test['Methangehalt CH4'], self.y_test, color='black', label='Actual Test Values')
        plt.scatter(self.X_test['Methangehalt CH4'], predictions, color='blue', label='Predicted test values')
        plt.xlabel('Methangehalt CH4')
        plt.ylabel('BHKW1_Biogas')
        plt.legend()
        plt.show()
        
        predictions = self.model.predict(self.X_test)
        plt.scatter(self.X_test['pH-Wert'], self.y_test, color='black', label='Actual Test Values')
        plt.scatter(self.X_test['pH-Wert'], predictions, color='blue', label='Predicted test values')
        plt.xlabel('pH-Wert')
        plt.ylabel('BHKW1_Biogas')
        plt.legend()
        plt.show()
        
        predictions = self.model.predict(self.X_test)
        plt.scatter(self.X_test['TS-Wert'], self.y_test, color='black', label='Actual Test Values')
        plt.scatter(self.X_test['TS-Wert'], predictions, color='blue', label='Predicted test values')
        plt.xlabel('TS-Wert')
        plt.ylabel('BHKW2_Biogas')
        plt.legend()
        plt.show()
        
        predictions = self.model.predict(self.X_test)
        plt.scatter(self.X_test['Methangehalt CH4'], self.y_test, color='black', label='Actual Test Values')
        plt.scatter(self.X_test['Methangehalt CH4'], predictions, color='blue', label='Predicted test values')
        plt.xlabel('Methangehalt CH4')
        plt.ylabel('BHKW2_Biogas')
        plt.legend()
        plt.show()
        
        predictions = self.model.predict(self.X_test)
        plt.scatter(self.X_test['pH-Wert'], self.y_test, color='black', label='Actual Test Values')
        plt.scatter(self.X_test['pH-Wert'], predictions, color='blue', label='Predicted test values')
        plt.xlabel('pH-Wert')
        plt.ylabel('BHKW2_Biogas')
        plt.legend()
        plt.show()

# Example of class and do operations
biogas_model = Regression('C:/Users/user/Downloads/Data.csv')
biogas_model.train_model()
mse=biogas_model.evaluate_model()
print(mse)
biogas_model.visualize_training_result()
biogas_model.visualize_test_result()
