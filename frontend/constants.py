from enum import Enum


class MODEL(Enum):
    # Define Enum class for model to be used to avoid hardcoded in code

    REGRESSION = 'Regression'
    CLASSIFICATION = 'Classification'


CLASSIFICATION_KERNELS = ['Linear', 'SVM']
