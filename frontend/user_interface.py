# third-party imports
import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

# custom imports
from src.utils import (MODEL_FEATURE, MODEL,
                       CLASSIFICATION_KERNELS,
                       REGRESSION_DEGREE,
                       matlab_time_to_datetime,
                       MODEL_RESULT_MODE)
from src.classes import Regression, Classification


def _get_selector(label: str, options: list[str], help: str | None = None):
    """
    Returns a selector widget with the given label and options.

    Args:
        label (str): The label for the selector widget.
        options (list[str]): The list of options for the selector widget.
        help (str | None, optional): The help text for the selector widget. Defaults to None.

    Returns:
        The selected option from the selector widget.
    """
    return st.sidebar.selectbox(label, options, help=help)


class UserInterface:
    """
    This class represents the user interface for the classification-regression application.
    It provides methods for setting up the UI components, visualizing data relationships, and
    handling user inputs.

    Attributes:
        selected_input_feature (str): The selected input feature.
        selected_output_feature (str): The selected output feature.
        data_pre_process (bool): Flag indicating whether the data is pre-processed.
        test_size (int): The size of the test data in percentage.
        regression_degree (int): The degree of the regression model.
        classification_kernel (str): The kernel for the classification model.
        date_relationship_visual (bool): Flag indicating whether to visualize the date relationship.
        relationship_visual (bool): Flag indicating whether to visualize the feature relationship.
        test_visual (bool): Flag indicating whether to visualize the test results.
        train_visual (bool): Flag indicating whether to visualize the train results.
        model (object): The regression or classification model.
    """

    def __init__(self, model_name: str):
        self.selected_input_feature = None
        self.selected_output_feature = None
        self.data_pre_process = False
        self.test_size = None
        self.regression_degree = 1
        self.classification_kernel = None
        self.date_relationship_visual = False
        self.relationship_visual = False
        self.test_visual = False
        self.train_visual = False

        if model_name == MODEL.REGRESSION.value:
            self.model = Regression()
        elif model_name == MODEL.CLASSIFICATION.value:
            self.model = Classification()

    def set_components(self):
        """
        This function sets up the components for the user interface.
        It creates the necessary buttons, sliders, and selectors for the user to interact with.
        """

        st.sidebar.markdown('---')

        self.data_pre_process = True if st.sidebar.radio(
            'Select state of the data', ('Raw', 'Pre-Processed'),
            horizontal=True) == 'Pre-Processed' else False

        st.sidebar.markdown('---')

        if isinstance(self.model, Regression):
            self.date_relationship_visual = st.sidebar.button('Visualize Data', help='''Visualize 
                                                                  all features relative to the date and 
                                                                  show the correlation heatmap''')

        self.relationship_visual = st.sidebar.button('Visualize Relationship', help='''Visualize 
                                                                  output features relative to the 
                                                                  input features''')

        st.sidebar.markdown('---')

        self.test_size = st.sidebar.slider('Test Size (%)', 5, 30, 20)

        if isinstance(self.model, Classification):
            self.classification_kernel = _get_selector('Kernel', CLASSIFICATION_KERNELS,
                                                       help='''Select the kernel for the 
                                                                       classification model''')

        if isinstance(self.model, Regression):
            self.regression_degree = _get_selector('Degree', REGRESSION_DEGREE, help='''
                                                                    if degree equals 1 then linear 
                                                                    regression is used, if degree is 
                                                                    bigger than 1 then polynomial 
                                                                    regression is used''')

        col1, col2 = st.sidebar.columns([0.45, 0.55])

        self.train_visual = col1.button('Train Results')

        self.test_visual = col2.button('Test Results')

    def visualize_date_relationship(self):
        """
        Visualizes the relationship between the date feature and other features in the regression data. 
        And shows the correlation heatmap between features.
        """
        if self.date_relationship_visual:
            date_feature_name = 'Datum'
            if self.data_pre_process:
                # process the data
                pass

            self.model.data[date_feature_name] = matlab_time_to_datetime(
                self.model.data[date_feature_name])

            for feature in self.model.data.columns:
                if feature != date_feature_name:
                    st.line_chart(
                        self.model.data[[date_feature_name, feature]], x=date_feature_name, y=feature)

            # show the correlation of each column except the 'Datum'
            if isinstance(self.model, Regression):
                fig, ax = plt.subplots()
                sns.heatmap(self.model.data.drop(columns=date_feature_name).corr(method='pearson'),
                            annot=True, cmap='coolwarm')
                st.pyplot(fig)

    def visualize_feature_relationship(self):
        """
        Visualizes the relationship between input and output features using scatter charts.
        """
        model_params = self._get_model_params()

        if self.relationship_visual:
            if self.data_pre_process:
                # process the data
                pass

            for output in model_params['output_features']:
                for input in model_params['input_features']:
                    st.scatter_chart(
                        self.model.data[[input, output]], x=input, y=output)

    def visualize_train(self):
        """
        Shows the training results based on the type of model.
        """
        if self.train_visual:
            self._initialize_model(MODEL_RESULT_MODE.TRAIN.value)

    def visualize_prediction(self):
        """
        Shows the predictions made by the model.
        """
        if self.test_visual:
            self._initialize_model(MODEL_RESULT_MODE.TEST.value)

    def _show_model_result(self, mode: str, X, Y, prediction):
        """
        Display the model result using scatter charts.

        Parameters:
        mode (str): The mode of the model (e.g., 'train', 'test').
        X: The input features.
        Y: The target variable(s).
        prediction: The predicted values.

        Returns:
        None
        """
        i = 0
        columns = X.columns.values
        df = pd.DataFrame(X, columns=columns)
        for y in Y:
            df[mode] = Y[y]

            # for regression, prediction has two columns: gas1 and gas2
            if isinstance(self.model, Regression):
                df['prediction'] = prediction[:, i]
            # for classification, prediction has only one column: whether donate blood
            elif isinstance(self.model, Classification):
                df['prediction'] = prediction

            for x in columns:
                st.write(y)
                st.scatter_chart(df[[mode, 'prediction', x]], x=x)

            i += 1

    def _show_evaluation(self, mode: str):
        """
        Displays the evaluation and visualization of the given mode.

        Parameters:
        mode (str): The mode for which evaluation is to be displayed (e.g., 'train', 'test').

        Returns:
        None
        """
        # show the evaluation
        st.header('Evaluation of ' + mode + ': ')

        if isinstance(self.model, Regression):
            for i in range(len(self.model.evaluation)):
                st.write(MODEL_FEATURE.REGRESSION_OUTPUT.value[i])
                st.write(self.model.evaluation[i][mode])
        elif isinstance(self.model, Classification):
            st.write(self.model.evaluation[mode])

        st.markdown('---')

        # if the model is classification, show the confusion matrix
        st.header('Visualization of ' + mode + ': ')

        if isinstance(self.model, Classification):
            cm = self.model.get_confusion_matrix()[mode]
            fig, ax = plt.subplots()
            sns.heatmap(cm, annot=True, fmt='d', cmap="Blues")
            plt.title('Confusion Matrix')
            plt.xlabel('Predicted Labels')
            plt.ylabel('True Labels')
            st.pyplot(fig)

        # show the scatter for each input
        if mode == MODEL_RESULT_MODE.TRAIN.value:
            self._show_model_result(
                mode, self.model.X_train, self.model.Y_train, self.model.prediction[mode])

        elif mode == MODEL_RESULT_MODE.TEST.value:
            self._show_model_result(
                mode, self.model.X_test, self.model.Y_test, self.model.prediction[mode])

    def _initialize_model(self, mode: str):
        """
        Initialize the instance according to the model user chooses
        """
        if self.data_pre_process:
            pass

        self.model.split_data(self.test_size / 100)

        if isinstance(self.model, Regression):
            self.model.train(self.regression_degree)
        elif isinstance(self.model, Classification):
            self.model.train(self.classification_kernel)

        self.model.predict()
        self.model.evaluate()
        self._show_evaluation(mode)

    def _get_model_params(self):
        """
        Returns a dictionary containing the input and output features based on the type of model.

        Returns:
            dict: A dictionary containing the input and output features.
        """
        if isinstance(self.model, Regression):
            return dict(input_features=MODEL_FEATURE.REGRESSION_INPUT.value,
                        output_features=MODEL_FEATURE.REGRESSION_OUTPUT.value)

        elif isinstance(self.model, Classification):
            return dict(input_features=MODEL_FEATURE.CLASSIFICATION_INPUT.value,
                        output_features=MODEL_FEATURE.CLASSIFICATION_OUTPUT.value)
