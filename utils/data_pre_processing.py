# import built-in packages
from os.path import dirname, join

import numpy as np
# import third party packages
import pandas as pd

# custom imports
from constants import MODEL_FEATURE, DATA_PATH
from sklearn.preprocessing import StandardScaler


class DataProcess:
    """
    class for data processing

    pass DATA_PATH enum as an argument when create an object

    ex: data_process = DataProcess(DATA_PATH.REGRESSION_RAW)
    """

    def __init__(self, filepath: DATA_PATH):
        self.filepath = join(dirname(dirname(__file__)), *filepath.value)
        self.data = pd.read_csv(self.filepath)
        self.output_column = pd.DataFrame()
        self.filtered_data = pd.DataFrame()
        self.scaled_data = np.ndarray

    def get_columns(self, model_features: MODEL_FEATURE):
        """
        :param model_features: Enum, choose the columns you want
        :return: X: Dataframe
        ex: df_cols = get_cols(data, MODEL_FEATURE.REGRESSION_INPUT)
        """
        self.output_column = self.data[model_features.value]

        return self.output_column

    def centered_moving_average(self, window_size):
        """
        It's recommended only used for regression model.

        :param window_size: Integer, recommend range (20-50)
        :return: data_moving_filtering Dataframe, using moving filter to smooth the data,
            having the same structure as the input data
        ex: data2 = centered_moving_average(data, 50)
        """
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
        """
        It's recommended only used for classification model.
        Example: data_scaled = classification_data.standard_scaling(dataframe)

        :return: data_scaled ndarray, which can be used for coming classification
        """
        scaler = StandardScaler()
        self.scaled_data = scaler.fit_transform(self.output_column)

        return self.scaled_data
