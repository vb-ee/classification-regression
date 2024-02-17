# import built-in packages
from os.path import dirname, join

import numpy as np
# import third party packages
import pandas as pd

# custom imports
from constants import MODEL_FEATURE, DATA_PATH


# import matplotlib.pyplot as plt


class DataProcess:
    """
    class for data processing

    pass DATA_PATH enum as an argument when create an object

    ex: data_process = DataProcess(DATA_PATH.REGRESSION_RAW)
    """

    def __init__(self, filepath):
        self.filepath = join(dirname(dirname(__file__)), *filepath.value)
        self.data = pd.read_csv(self.filepath)

    def get_columns(self, model_features: MODEL_FEATURE):
        """
        Example: df_cols = get_cols(data, MODEL_FEATURE.REGRESSION_INPUT)

        :param model_features: Enum, choose the columns u want to get
        :return: X: Dataframe
        """

        return self.data[model_features.value]

    def centered_moving_average(self, data, window_size):
        """
        It's only used for regression model.
        It shouldn't be used for classification model.
        Example: data2 = centered_moving_average(data, 50)

        :param data: Dataframe, with multiple columns
        :param window_size: Integer, recommend range (20-50)
        :return: data_moving_filtering Dataframe, using moving filter to smooth the data,
            having the same structure as the input data
        """
        if window_size not in range(20, 51):
            raise ValueError('Window size should in range 20-50')

        half_window = window_size // 2
        for column in data.columns:
            temp = []
            for i in range(0, len(data)):
                if i <= half_window + 1:
                    temp.append(np.mean(data[column][0:i + half_window]))
                else:
                    temp.append(np.mean(data[column][i - half_window:i + half_window]))
            data.loc[:, column] = temp

        return data


data_process = DataProcess(DATA_PATH.REGRESSION_RAW)
X = data_process.get_columns(MODEL_FEATURE.REGRESSION_INPUT)
X1 = data_process.centered_moving_average(X, 20)
print(X1)
