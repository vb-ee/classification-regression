# third-party imports
import unittest
import numpy as np
import pandas as pd

# custom imports
from src.classes import Classification


class TestClassification(unittest.TestCase):

    def setUp(self):
        self.classification_model = Classification()
        self.classification_model.split_data(test_size=0.2)
        self.classification_model.train(kernel='rbf')
        self.classification_model.predict()
        self.classification_model.evaluate()

    def test_split_data(self):
        test_size = 0.2
        self.classification_model.split_data(test_size)

        # Check if data is split properly
        self.assertEqual(len(self.classification_model.X_train),
                         int(len(self.classification_model.data) * (1 - test_size)))
        self.assertAlmostEqual(len(self.classification_model.X_test),
                               int(len(self.classification_model.data) * test_size), delta=1)

        # Check if split data length match
        self.assertEqual(len(self.classification_model.X_test),
                         len(self.classification_model.Y_test))
        self.assertEqual(len(self.classification_model.X_train),
                         len(self.classification_model.Y_train))

    def test_data_types(self):
        # Check data type of X_train_scaled and X_test_scaled
        self.assertIsInstance(self.classification_model._get_scalar().transform(self.classification_model.X_train),
                              np.ndarray)
        self.assertIsInstance(self.classification_model._get_scalar().transform(self.classification_model.X_test),
                              np.ndarray)

        # Check if Y_train and Y_test are 1-d array
        self.assertIsInstance(self.classification_model.Y_train.values.ravel(), np.ndarray)
        self.assertEqual(self.classification_model.Y_train.values.ravel().ndim, 1)

        self.assertIsInstance(self.classification_model.Y_test.values.ravel(), np.ndarray)
        self.assertEqual(self.classification_model.Y_test.values.ravel().ndim, 1)

    def test_prediction_shape(self):
        # Check if the prediction has the correct shape. shape[0] return the data row number
        self.assertEqual(self.classification_model.prediction['train'].shape[0], len(self.classification_model.X_train))
        self.assertEqual(self.classification_model.prediction['test'].shape[0], len(self.classification_model.X_test))

    def test_evaluation(self):
        # Check if evaluation train and test results are dataframe
        self.assertIsInstance(self.classification_model.evaluation['train'], pd.DataFrame)
        self.assertIsInstance(self.classification_model.evaluation['test'], pd.DataFrame)

        # Check the range of scores for training data. 
        # .values return 2-d array, using .ravel to flatten the data
        for score in self.classification_model.evaluation['train'].values.ravel():
            self.assertGreaterEqual(score, 0)
            self.assertLessEqual(score, 1)

        # Check the range of scores for testing data
        for score in self.classification_model.evaluation['test'].values.ravel():
            self.assertGreaterEqual(score, 0)
            self.assertLessEqual(score, 1)

    def test_confusion_matrix_shape(self):
        # Check if confusion matrices shape
        confusion_matrices = self.classification_model.get_confusion_matrix()
        self.assertEqual(confusion_matrices['train'].shape, (2, 2))
        self.assertEqual(confusion_matrices['test'].shape, (2, 2))
