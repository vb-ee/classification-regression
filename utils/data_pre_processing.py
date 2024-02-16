# import built-in packages
from os.path import dirname, join

# import third party packages
import pandas as pd
import numpy as np
# import matplotlib.pyplot as plt


# custom imports
from constants import MODEL_FEATURE


class DataProcess:
    '''
    class for data processing

    pass DATA_PATH enum as an argument when creatin an object

    ex: data_process = DataProcess(DATA_PATH.REGRESSION_RAW)
    '''

    def __init__(self, filepath):
        self.filepath = join(dirname(dirname(__file__)), *filepath.value)
        self.data = pd.read_csv(self.filepath)

    def get_columns(data, model_features: MODEL_FEATURE):
        '''
        Example: df_cols = get_cols(data, MODEL_FEATURE.REGRESSION_INPUT)

        :param model_features: Enum, choose the columns u want to get
        :param data: Dataframe, get the data from pandas read_csv function
        :return: X: Dataframe
        '''
        return data[model_features.value]

    # TODO: refactor the function, avoid using reindex and deal with hardcoded variables
    def centered_moving_average(data, window_size):
        '''
        Example: data2 = centered_moving_average(data, 50)

        :param data: Dataframe, with multiple columns
        :param window_size: Integer, recommend range (20-50)
        :return: data_moving_filtering Dataframe, using moving filter to smooth the data,
            having the same structure as the input data
        '''
        # data2 is an empty Dataframe with the same structure as the input data
        data_moving_filtering = data.reindex(columns=data.columns).iloc[0:0]
        data_moving_filtering['Datum'] = data['Datum']
        half_window = window_size // 2
        smoothed_data = np.zeros(len(data))

        for column in data:
            if column == 'Datum':
                pass
            else:
                for i in range(0, len(smoothed_data)):
                    if i <= half_window + 1:
                        smoothed_data[i] = np.mean(
                            data[column][0:i + half_window])
                    else:
                        smoothed_data[i] = np.mean(
                            data[column][i - half_window:i + half_window])

                data_moving_filtering[column] = smoothed_data

        return data_moving_filtering
