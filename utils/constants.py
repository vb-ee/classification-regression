from enum import Enum

# Defined Enum class for data paths to avoid hardcoded variables
class DATA_PATH(Enum):
    REGRESSION_RAW = ['data', 'raw', 'regression', 'regression.csv']
    CLASSIFICATION_RAW = ['data', 'raw', 'classification', 'transfusion.csv']
    REGRESSION_PROCESSED = ['data', 'processed', 'regression', 'regression.csv']
    CLASSIFICATION_PROCESSED = ['data', 'processed', 'classification', 'transfusion.csv']
