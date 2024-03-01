# built in imports
import unittest
from math import sqrt

# custom imports
from src.classes import Regression
from src.utils import MODEL_FEATURE


class TestRegression(unittest.TestCase):

    def setUp(self):
        self.re_raw = Regression()
        self.re_raw.split_data(0.2)
        self.re_raw.train()
        self.prediction_raw = self.re_raw.predict()
        self.evaluation_raw = self.re_raw.evaluate(self.prediction_raw)

    def test_init(self):
        self.assertEqual(self.re_raw.X.shape, (1621, 3))
        i = 0
        for col in self.re_raw.X.columns.values:
            self.assertEqual(col, MODEL_FEATURE.REGRESSION_INPUT.value[i])
            i += 1
        self.assertEqual(self.re_raw.Y.shape, (1621, 2))
        i = 0
        for col in self.re_raw.Y.columns.values:
            self.assertEqual(col, MODEL_FEATURE.REGRESSION_OUTPUT.value[i])
            i += 1

    def test_split_data(self):
        # how many line in test and train for raw data
        self.assertAlmostEqual(len(self.re_raw.X_test), 324, delta=1)
        self.assertEqual(len(self.re_raw.X_train),
                         1621 - len(self.re_raw.X_test))
        self.assertEqual(len(self.re_raw.Y_test), len(self.re_raw.X_test))
        self.assertEqual(len(self.re_raw.Y_train), len(self.re_raw.X_train))
        self.assertEqual(len(self.re_raw.Y_train),
                         1621 - len(self.re_raw.Y_test))

    def test_prediction(self):
        # check the shape of prediction
        self.assertIsInstance(self.prediction_raw, dict)
        self.assertEqual(
            self.prediction_raw['train'].shape, (len(
                self.re_raw.Y_train), 2))
        self.assertEqual(
            self.prediction_raw['test'].shape, (len(
                self.re_raw.Y_test), 2))

    def test_evaluate(self):
        self.assertIsInstance(self.evaluation_raw, dict)
        self.assertIsInstance(self.evaluation_raw['train']['mse'], float)
        self.assertIsInstance(self.evaluation_raw['train']['rmse'], float)
        self.assertAlmostEqual(
            self.evaluation_raw['train']['rmse'], sqrt(
                self.evaluation_raw['train']['mse']), delta=0.5)
