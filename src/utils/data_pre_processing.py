# import built-in packages
from os.path import dirname, join

# import third party packages
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


class DataProcess:
    """
    class for data processing

    pass DATA_PATH enum as an argument when create an object

    ex: data_process = DataProcess(DATA_PATH.REGRESSION_RAW.value)
    """

    def __init__(self, filepath: list[str]):
        self.filepath = join(
            dirname(dirname(dirname(__file__))), *filepath)
        self.data = pd.read_csv(self.filepath)

    def centered_moving_average(self, model_features: list[str], window_size: int = 20):
        """
        use moving filter to smooth the data and return the filtered data
        It's recommended only used for regression model.

        :param window_size: Integer, recommend range (20-50)
        :param model_features: List of strings, which are the columns' names of the data
        :return: Dataframe

        ex: filtered_data = regression_data.centered_moving_average(20)
        """

        if window_size not in range(20, 51):
            raise ValueError('Window size should in range 20-50')

        output = self.data[model_features]
        filtered_data = pd.DataFrame()
        half_window = window_size // 2
        for column in output:
            temp = []
            for i in range(0, len(output)):
                if i <= half_window + 1:
                    temp.append(np.mean(output[column][0:i + half_window]))
                else:
                    temp.append(
                        np.mean(output[column][i - half_window:i + half_window]))
            filtered_data[column] = temp

        return filtered_data

    def standard_scaling(self, model_features: list[str]):
        """
        It's recommended only used for classification model.

        :return: scaled_data: Dataframe, which can be used for coming classification

        ex: scaled_data = classification_data.standard_scaling()
        """
        output = self.data[model_features]
        scaler = StandardScaler()
        scaled_data = pd.DataFrame(scaler.fit_transform(output))
        scaled_data.columns = output.columns

        return scaled_data
