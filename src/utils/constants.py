from enum import Enum


class DATA_PATH(Enum):
    # Define Enum class for data paths to avoid hardcoded in code

    REGRESSION_RAW = ['data', 'raw', 'regression', 'regression.csv']
    CLASSIFICATION_RAW = ['data', 'raw', 'classification', 'transfusion.csv']
    REGRESSION_PROCESSED = ['data', 'processed',
                            'regression', 'regression.csv']
    CLASSIFICATION_PROCESSED = ['data', 'processed',
                                'classification', 'transfusion.csv']


class MODEL_FEATURE(Enum):
    # Define Enum class for model features to avoid hardcoded values

    REGRESSION_INPUT = ['Methangehalt CH4', 'TS-Wert', 'pH-Wert']
    REGRESSION_OUTPUT = ['BHKW1_Biogas', 'BHKW2_Biogas']
    REGRESSION = ['Datum', 'BHKW1_Biogas', 'BHKW2_Biogas', 'Methangehalt CH4', 'TS-Wert', 'pH-Wert']
    CLASSIFICATION_INPUT = [
        'Recency (months)', 'Frequency (times)', 'Monetary (c.c. blood)', 'Time (months)']
    CLASSIFICATION_OUTPUT = ['whether he/she donated blood in March 2007']
    CLASSIFICATION = ['Recency (months)', 'Frequency (times)', 'Monetary (c.c. blood)',
                      'Time (months)', 'whether he/she donated blood in March 2007']
