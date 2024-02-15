# import built in packages
from os.path import dirname, join

# import third party packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# custom imports
from constants import DATA_PATH


def load_csv(filepath: DATA_PATH):
    """
    Example: data = load_csv(DATA_PATH.CLASSIFICATION_RAW)

    :return: Dataframe
    """
    return pd.read_csv(join(dirname(dirname(__file__)), *filepath.value))


def get_regression_cols(df_data):
    """
    Example: Inputs, out_bhkw1, out_bhkw2 = get_df_cols(data)

    :param df_data: Dataframe, get the data from load_csv function
    :return: X: Dataframe, three input columns
            y1: Dataframe, output column BHKW1_Biogas
            y2: Dataframe, output column BHKW2_Biogas
    """
    X = df_data[['Methangehalt CH4', 'TS-Wert', 'pH-Wert']]
    y1 = df_data[['BHKW1_Biogas']]
    y2 = df_data[['BHKW2_Biogas']]

    return X, y1, y2


def get_classification_cols(df_data):
    """
    Example: Inputs, donated_blood_flag = get_classification_cols(data)

    :param df_data: Dataframe, get the data from load_csv function
    :return: X: Dataframe, four input columns
            y: Dataframe, output column whether he/she donated blood in March 2007
    """
    X = df_data[['Recency (months)', 'Frequency (times)',
                 'Monetary (c.c. blood)', 'Time (months)']]
    y = df_data[['whether he/she donated blood in March 2007']]

    return X, y


def centered_moving_average(data, window_size):
    """
    Example: data2 = centered_moving_average(data, 50)

    :param data: Dataframe, with multiple columns
    :param window_size: Integer, recommend range (20-50)
    :return: data_moving_filtering Dataframe, using moving filter to smooth the data,
        having the same structure as the input data
    """
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
                    smoothed_data[i] = np.mean(data[column][0:i + half_window])
                else:
                    smoothed_data[i] = np.mean(
                        data[column][i - half_window:i + half_window])

            data_moving_filtering[column] = smoothed_data

    return data_moving_filtering


# data = load_csv(DATA_PATH.REGRESSION_RAW)
# data2 = centered_moving_average(data, 50)
# print(data.head())
# print(data2.head())
#
# for column in data:
#     plt.plot(data[column], label='Original Data')
#     plt.plot(data2[column], label='Filtering Data')
#     plt.legend()
#     plt.show()
