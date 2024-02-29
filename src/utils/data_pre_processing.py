# import built-in packages
from datetime import datetime, timedelta

# import third party packages
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


def centered_moving_average(data: pd.DataFrame, model_features: list[str], window_size: int = 20):
    '''
    use moving filter to smooth the data and return the filtered data
    It's recommended only used for regression model.

    :param window_size: Integer, recommend range (20-50)
    :param model_features: List of strings, which are the columns' names of the data
    :return: Dataframe

    ex: filtered_data = regression_data.centered_moving_average(20)
    '''

    if window_size not in range(20, 51):
        raise ValueError('Window size should in range 20-50')

    output = data[model_features]
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


def standard_scaling(data: pd.DataFrame, model_features: list[str]):
    '''
    It's recommended only used for classification model.

    :return: scaled_data: Dataframe, which can be used for coming classification

    ex: scaled_data = classification_data.standard_scaling()
    '''
    output = data[model_features]
    scaler = StandardScaler()
    scaled_data = pd.DataFrame(scaler.fit_transform(output))
    scaled_data.columns = output.columns

    return scaled_data


def matlab_time_to_datetime(date_column: pd.DataFrame) -> list[str]:
    '''
    convert matlab date time column to readable time

    param: data Dataframe
    '''

    start_date = datetime(1, 1, 1)
    dates = []

    for date_decimal in date_column:
        # Convert decimal to datetime object
        delta = timedelta(days=date_decimal)

        date = start_date + delta

        # Output: 2003-01-01
        dates.append(date.strftime('%Y-%m-%d %H'))

    return dates
