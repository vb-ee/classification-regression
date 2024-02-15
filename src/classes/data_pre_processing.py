import pandas as pd
import os
import pathlib


def load_csv(method):
    """
    Example: data = load_csv('regression')

    :param method: String, regression or classification
    :return: Dataframe
    """
    dir_path = pathlib.Path(__file__).parent.parent.parent.resolve()
    if method == 'regression':
        filepath = os.path.join(dir_path, 'data', 'raw', 'regression', 'regression.csv')
        data = pd.read_csv(filepath, skiprows=1)
    elif method == 'classification':
        filepath = os.path.join(dir_path, 'data', 'raw', 'classification', 'transfusion.csv')
        data = pd.read_csv(filepath)

    return data


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
    X = df_data[['Recency (months)', 'Frequency (times)', 'Monetary (c.c. blood)', 'Time (months)']]
    y = df_data[['whether he/she donated blood in March 2007']]

    return X, y
