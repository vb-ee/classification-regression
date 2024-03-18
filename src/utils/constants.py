from enum import Enum


class MODEL(Enum):
    # Define Enum class for model to be used to avoid hardcoded in code

    REGRESSION = 'Regression'
    CLASSIFICATION = 'Classification'


class MODEL_RESULT_MODE(Enum):
    # Define Enum class for model to be used to avoid hardcoded in code

    TRAIN = 'train'
    TEST = 'test'


class DATA_PATH(Enum):
    # Define Enum class for data paths to avoid hardcoded in code

    REGRESSION_RAW = ['data', 'raw', 'regression', 'regression.csv']
    CLASSIFICATION_RAW = ['data', 'raw', 'classification', 'transfusion.csv']


class MODEL_FEATURE(Enum):
    # Define Enum class for model features to avoid hardcoded values

    REGRESSION_INPUT = ['Methangehalt CH4', 'TS-Wert', 'pH-Wert']
    REGRESSION_OUTPUT = ['BHKW1_Biogas', 'BHKW2_Biogas']
    REGRESSION = ['Datum', 'BHKW1_Biogas', 'BHKW2_Biogas',
                  'Methangehalt CH4', 'TS-Wert', 'pH-Wert']
    CLASSIFICATION_INPUT = [
        'Recency (months)', 'Frequency (times)', 'Monetary (blood)', 'Time (months)']
    CLASSIFICATION_OUTPUT = ['whether he/she donated blood in March 2007']
    CLASSIFICATION = ['Recency (months)', 'Frequency (times)', 'Monetary (blood)',
                      'Time (months)', 'whether he/she donated blood in March 2007']


# Define array for classification kernels to avoid hardcoded values. Add more kernels here if needed
REGRESSION_DEGREE = [1, 2, 3, 4]
CLASSIFICATION_KERNELS = ['rbf', 'sigmoid', 'poly']
