# import built-in packages
from os.path import dirname, join

# import third party packages
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

# custom imports
from constants import MODEL_FEATURE, DATA_PATH


class DataProcess:
    '''
    class for data processing

    pass DATA_PATH enum as an argument when create an object

    ex: data_process = DataProcess(DATA_PATH.REGRESSION_RAW)
    '''

    def __init__(self, filepath: DATA_PATH):
        self.filepath = join(dirname(dirname(__file__)), *filepath.value)
        self.data = pd.read_csv(self.filepath)
        self.output_column = pd.DataFrame()
        self.filtered_data = pd.DataFrame()
        self.scaled_data = np.ndarray

    def get_columns(self, model_features: MODEL_FEATURE):
        '''
        pass MODEL_FEATURE enum as an argument when choose the columns to return
        :return: Dataframe type

        ex: regression_original = regression_data.get_columns(MODEL_FEATURE.REGRESSION_OUTPUT)
        '''
        self.output_column = self.data[model_features.value]

        return self.output_column

    def centered_moving_average(self, window_size):
        '''
        use moving filter to smooth the data and return the filtered data
        It's recommended only used for regression model.

        :param window_size: Integer, recommend range (20-50)
        :return: Dataframe type

        ex: filtered_data = regression_data.centered_moving_average(20)
        '''
        if window_size not in range(20, 51):
            raise ValueError('Window size should in range 20-50')

        half_window = window_size // 2
        for column in self.output_column.columns:
            temp = []
            for i in range(0, len(self.output_column)):
                if i <= half_window + 1:
                    temp.append(np.mean(self.output_column[column][0:i + half_window]))
                else:
                    temp.append(np.mean(self.output_column[column][i - half_window:i + half_window]))
            self.filtered_data.loc[:, column] = temp

        return self.filtered_data

    def standard_scaling(self):
        '''
        It's recommended only used for classification model.

        :return: scaled_data: ndarray, which can be used for coming classification
        ex: scaling_data = classification_data.standard_scaling()
        '''

        scaler = StandardScaler()
        self.scaled_data = scaler.fit_transform(self.output_column)

        return self.scaled_data
