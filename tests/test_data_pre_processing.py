import unittest

import pandas as pd
from utils.constants import DATA_PATH, MODEL_FEATURE
from utils.data_pre_processing import DataProcess


class TestDataProcess(unittest.TestCase):

    def setUp(self):
        self.dp_regression = DataProcess(DATA_PATH.REGRESSION_RAW)
        self.dp_classification = DataProcess(DATA_PATH.CLASSIFICATION_RAW)

    def test_init(self):
        """
        check the columns' names of data read from csv file
        """
        self.assertIsInstance(self.dp_regression.data, pd.DataFrame)
        self.assertEqual(self.dp_regression.data.shape, (1621, 6))
        for i in range(len(self.dp_regression.data.columns)):
            self.assertEqual(self.dp_regression.data.columns.values[i], MODEL_FEATURE.REGRESSION.value[i])

        self.assertIsInstance(self.dp_classification.data, pd.DataFrame)
        self.assertEqual(self.dp_classification.data.shape, (748, 5))
        for i in range(len(self.dp_classification.data.columns)):
            self.assertEqual(self.dp_classification.data.columns.values[i], MODEL_FEATURE.CLASSIFICATION.value[i])

    def test_get_columns(self):
        """
        check the datatype and the shape of data returned by get_columns
        """
        self.assertIsInstance(self.dp_regression.get_columns(MODEL_FEATURE.REGRESSION_INPUT), pd.DataFrame)
        self.assertEqual(self.dp_regression.get_columns(MODEL_FEATURE.REGRESSION_INPUT).shape, (1621, 3))
        self.assertIsInstance(self.dp_regression.get_columns(MODEL_FEATURE.REGRESSION_OUTPUT), pd.DataFrame)
        self.assertEqual(self.dp_regression.get_columns(MODEL_FEATURE.REGRESSION_OUTPUT).shape, (1621, 2))
        self.assertIsInstance(self.dp_classification.get_columns(MODEL_FEATURE.CLASSIFICATION_INPUT), pd.DataFrame)
        self.assertEqual(self.dp_classification.get_columns(MODEL_FEATURE.CLASSIFICATION_INPUT).shape, (748, 4))
        self.assertIsInstance(self.dp_classification.get_columns(MODEL_FEATURE.CLASSIFICATION_OUTPUT), pd.DataFrame)
        self.assertEqual(self.dp_classification.get_columns(MODEL_FEATURE.CLASSIFICATION_OUTPUT).shape, (748, 1))

    def test_centered_moving_average(self):
        """
        check the datatype and the shape of data returned by get_columns
        """
        filtered_data = self.dp_regression.centered_moving_average(MODEL_FEATURE.REGRESSION_INPUT, 20)
        self.assertIsInstance(filtered_data, pd.DataFrame)
        self.assertEqual(filtered_data.shape, (1621, 3))

        filtered_data = self.dp_regression.centered_moving_average(MODEL_FEATURE.REGRESSION_OUTPUT, 20)
        self.assertIsInstance(filtered_data, pd.DataFrame)
        self.assertEqual(filtered_data.shape, (1621, 2))

    def test_standard_scaling(self):
        """
        check the datatype and the shape of the result returned by standard_scaling
        check the values whether between -10 and 10
        """
        scaled_data = self.dp_classification.standard_scaling(MODEL_FEATURE.CLASSIFICATION_INPUT)
        self.assertIsInstance(scaled_data, pd.DataFrame)
        self.assertEqual(scaled_data.shape, (748, 4))
        for row in scaled_data.values:
            for value in row:
                self.assertGreaterEqual(value, -10)
                self.assertLessEqual(value, 10)

        scaled_data = self.dp_classification.standard_scaling(MODEL_FEATURE.CLASSIFICATION_OUTPUT)
        self.assertIsInstance(scaled_data, pd.DataFrame)
        self.assertEqual(scaled_data.shape, (748, 1))
        for row in scaled_data.values:
            for value in row:
                self.assertGreaterEqual(value, -10)
                self.assertLessEqual(value, 10)


if __name__ == '__main__':
    unittest.main()
