# built in imports
import unittest

# third-party imports
import pandas as pd

# custom imports
from src.utils import DATA_PATH, MODEL_FEATURE, DataProcess


class TestDataProcess(unittest.TestCase):

    def setUp(self):
        self.regression = DataProcess(DATA_PATH.REGRESSION_RAW.value)
        self.classification = DataProcess(DATA_PATH.CLASSIFICATION_RAW.value)

    def test_init(self):
        '''
        check the columns' names of data read from csv file
        '''
        self.assertIsInstance(self.regression.data, pd.DataFrame)
        self.assertEqual(self.regression.data.shape, (1621, 6))
        for i in range(len(self.regression.data.columns)):
            self.assertEqual(
                self.regression.data.columns.values[i], MODEL_FEATURE.REGRESSION.value[i])

        self.assertIsInstance(self.classification.data, pd.DataFrame)
        self.assertEqual(self.classification.data.shape, (748, 5))
        for i in range(len(self.classification.data.columns)):
            self.assertEqual(
                self.classification.data.columns.values[i], MODEL_FEATURE.CLASSIFICATION.value[i])

    def test_centered_moving_average(self):
        '''
        check the datatype and the shape of data returned by get_columns
        '''
        filtered_data = self.regression.centered_moving_average(
            MODEL_FEATURE.REGRESSION_INPUT.value, 20)
        self.assertIsInstance(filtered_data, pd.DataFrame)
        self.assertEqual(filtered_data.shape, (1621, 3))

        filtered_data = self.regression.centered_moving_average(
            MODEL_FEATURE.REGRESSION_OUTPUT.value, 20)
        self.assertIsInstance(filtered_data, pd.DataFrame)
        self.assertEqual(filtered_data.shape, (1621, 2))

    def test_standard_scaling(self):
        '''
        check the datatype and the shape of the result returned by standard_scaling
        check the values whether between -10 and 10
        '''
        scaled_data = self.classification.standard_scaling(
            MODEL_FEATURE.CLASSIFICATION_INPUT.value)
        self.assertIsInstance(scaled_data, pd.DataFrame)
        self.assertEqual(scaled_data.shape, (748, 4))
        for row in scaled_data.values:
            for value in row:
                self.assertGreaterEqual(value, -10)
                self.assertLessEqual(value, 10)

        scaled_data = self.classification.standard_scaling(
            MODEL_FEATURE.CLASSIFICATION_OUTPUT.value)
        self.assertIsInstance(scaled_data, pd.DataFrame)
        self.assertEqual(scaled_data.shape, (748, 1))
        for row in scaled_data.values:
            for value in row:
                self.assertGreaterEqual(value, -10)
                self.assertLessEqual(value, 10)
