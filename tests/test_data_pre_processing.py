# built in imports
import unittest

# third-party imports
import pandas as pd
import numpy as np

# custom imports
from src.utils import centered_moving_average, MODEL_FEATURE, replace_outlier
from src.classes import Regression, Classification


class TestDataProcess(unittest.TestCase):

    def setUp(self):
        self.regression_data = Regression().data
        self.classification_data = Classification().data

    def test_centered_moving_average(self):
        '''
        check the datatype and the shape of data returned by get_columns
        '''
        filtered_data = centered_moving_average(self.regression_data,
                                                MODEL_FEATURE.REGRESSION_INPUT.value, 20)
        self.assertIsInstance(filtered_data, pd.DataFrame)
        self.assertEqual(filtered_data.shape, (1621, 3))

        filtered_data = centered_moving_average(self.regression_data,
                                                MODEL_FEATURE.REGRESSION_OUTPUT.value, 20)
        self.assertIsInstance(filtered_data, pd.DataFrame)
        self.assertEqual(filtered_data.shape, (1621, 2))

    def test_replace_outlier(self):
        data_without_outliers = replace_outlier(
            self.regression_data, MODEL_FEATURE.REGRESSION_INPUT.value)
        for col in data_without_outliers[MODEL_FEATURE.REGRESSION_INPUT.value]:
            data = data_without_outliers[col]
            quartile_1, quartile_3 = np.percentile(data, [25, 75])
            iqr = quartile_3 - quartile_1
            threshold = 1.5
            outliers = [x for x in data if (
                x < quartile_1 - threshold * iqr) or (x > quartile_3 + threshold * iqr)]
            self.assertEqual(len(outliers), 0)
