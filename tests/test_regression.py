# built in imports
import unittest
from math import sqrt

# third-party imports

# custom imports
from src.classes import Regression


class TestRegression(unittest.TestCase):

    def setUp(self):
        self.re_raw = Regression(0.2, False)
        self.re_processed = Regression(0.2, True, 50)
        self.re_raw.train()
        self.re_processed.train()
        self.prediction_raw = self.re_raw.predict()
        self.prediction_processed = self.re_processed.predict()
        self.evaluation_raw = self.re_raw.evaluate(self.prediction_raw)
        self.evaluation_processed = self.re_processed.evaluate(
            self.prediction_raw)

    def test_init(self):
        # how many line in test and train for raw data
        self.assertAlmostEqual(len(self.re_raw.X_test), 324, delta=1)
        self.assertEqual(len(self.re_raw.X_train),
                         1621 - len(self.re_raw.X_test))
        self.assertEqual(len(self.re_raw.Y_test), len(self.re_raw.X_test))
        self.assertEqual(len(self.re_raw.Y_train), len(self.re_raw.X_train))
        self.assertEqual(len(self.re_raw.Y_train),
                         1621 - len(self.re_raw.Y_test))

        # how many line in test and train for processed data
        self.assertAlmostEqual(len(self.re_processed.X_test), 324, delta=1)
        self.assertEqual(len(self.re_processed.X_train),
                         1621 - len(self.re_processed.X_test))
        self.assertEqual(len(self.re_processed.Y_test),
                         len(self.re_processed.X_test))
        self.assertEqual(len(self.re_processed.Y_train),
                         len(self.re_processed.X_train))
        self.assertEqual(len(self.re_processed.Y_train),
                         1621 - len(self.re_processed.Y_test))

    def test_prediction(self):
        # check the shape of prediction
        self.assertIsInstance(self.prediction_raw, dict)
        self.assertEqual(
            self.prediction_raw['train'].shape, (len(
                self.re_raw.Y_train), 2))
        self.assertEqual(
            self.prediction_raw['test'].shape, (len(
                self.re_raw.Y_test), 2))

        self.assertIsInstance(self.prediction_processed, dict)
        self.assertEqual(
            self.prediction_processed['train'].shape, (len(
                self.re_processed.Y_train), 2))
        self.assertEqual(
            self.prediction_processed['test'].shape, (len(
                self.re_processed.Y_test), 2))

    def test_evaluate(self):
        self.assertIsInstance(self.evaluation_raw, dict)
        self.assertIsInstance(self.evaluation_raw['train']['mse'], float)
        self.assertIsInstance(self.evaluation_raw['train']['rmse'], float)
        self.assertAlmostEqual(
            self.evaluation_raw['train']['rmse'], sqrt(
                self.evaluation_raw['train']['mse']), delta=0.5)

        self.assertIsInstance(self.evaluation_processed, dict)
        self.assertIsInstance(self.evaluation_processed['train']['mse'], float)
        self.assertIsInstance(
            self.evaluation_processed['train']['rmse'], float)
        self.assertAlmostEqual(
            self.evaluation_processed['train']['rmse'], sqrt(
                self.evaluation_processed['train']['mse']), delta=0.5)
