# built in imports
import unittest
from math import sqrt

# custom imports
from src.classes import Regression
from src.utils import MODEL_FEATURE


class TestRegression(unittest.TestCase):

    def setUp(self):
        self.regression = Regression()
        self.regression.set_polynomial_order(2)
        self.regression.split_data(0.2)
        self.regression.train()
        self.regression.predict()
        self.evaluation = self.regression.evaluate()

    def test_init(self):
        self.assertEqual(self.regression.X.shape, (1621, 3))
        i = 0
        for col in self.regression.X.columns.values:
            self.assertEqual(col, MODEL_FEATURE.REGRESSION_INPUT.value[i])
            i += 1
        self.assertEqual(self.regression.Y.shape, (1621, 2))
        i = 0
        for col in self.regression.Y.columns.values:
            self.assertEqual(col, MODEL_FEATURE.REGRESSION_OUTPUT.value[i])
            i += 1

    def test_split_data(self):
        # how many line in test and train for raw data
        self.assertAlmostEqual(len(self.regression.X_test), 324, delta=1)
        self.assertEqual(len(self.regression.X_train),
                         1621 - len(self.regression.X_test))
        self.assertEqual(len(self.regression.Y_test),
                         len(self.regression.X_test))
        self.assertEqual(len(self.regression.Y_train),
                         len(self.regression.X_train))
        self.assertEqual(len(self.regression.Y_train),
                         1621 - len(self.regression.Y_test))

    def test_prediction(self):
        # check the shape of prediction
        self.assertIsInstance(self.regression.prediction, dict)
        self.assertEqual(
            self.regression.prediction['train'].shape, (len(
                self.regression.Y_train), 2))
        self.assertEqual(
            self.regression.prediction['test'].shape, (len(
                self.regression.Y_test), 2))

    def test_evaluate(self):
        self.assertIsInstance(self.evaluation, dict)
        self.assertIsInstance(self.evaluation['train']['mean_squared_error'], float)
        self.assertIsInstance(self.evaluation['train']['root_mean_squared_error'], float)
        self.assertIsInstance(self.evaluation['train']['r2_score'], float)
        self.assertIsInstance(self.evaluation['train']['explained_variance_score'], float)
        self.assertIsInstance(self.evaluation['test']['mean_squared_error'], float)
        self.assertIsInstance(self.evaluation['test']['root_mean_squared_error'], float)
        self.assertIsInstance(self.evaluation['test']['r2_score'], float)
        self.assertIsInstance(self.evaluation['test']['explained_variance_score'], float)
        self.assertAlmostEqual(
            self.evaluation['train']['root_mean_squared_error'], sqrt(
                self.evaluation['train']['mean_squared_error']), delta=0.5)
        self.assertLessEqual(self.evaluation['test']['r2_score'], 1)
        self.assertLessEqual(self.evaluation['test']['explained_variance_score'], 1)
        self.assertGreaterEqual(self.evaluation['test']['r2_score'], 0)
        self.assertGreaterEqual(self.evaluation['test']['explained_variance_score'], 0)
