from enum import Enum


# Defined Enum class for data paths to avoid hardcoded variables
class DATA_PATH(Enum):
    REGRESSION_RAW = ['data', 'raw', 'regression', 'regression.csv']
    CLASSIFICATION_RAW = ['data', 'raw', 'classification', 'transfusion.csv']
    REGRESSION_PROCESSED = ['data', 'processed', 'regression', 'regression.csv']
    CLASSIFICATION_PROCESSED = ['data', 'processed', 'classification', 'transfusion.csv']


class METHOD(Enum):
    REGRESSION_INPUT = ['Methangehalt CH4', 'TS-Wert', 'pH-Wert']
    REGRESSION_OUTPUT = ['BHKW1_Biogas', 'BHKW2_Biogas']
    CLASSIFICATION_INPUT = ['Recency (months)', 'Frequency (times)', 'Monetary (c.c. blood)', 'Time (months)']
    CLASSIFICATION_OUTPUT = ['whether he/she donated blood in March 2007']

