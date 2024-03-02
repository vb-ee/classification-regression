# third-party imports
import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

# custom imports
from src.utils import MODEL_FEATURE, MODEL, CLASSIFICATION_KERNELS, matlab_time_to_datetime, MODEL_RESULT_MODE
from src.classes import Regression, Classification


class UserInterface:
    def __init__(self, model_name: str):
        self.selected_input_feature = None
        self.selected_output_feature = None
        self.data_pre_process = False
        self.test_size = None
        self.classification_kernel = None
        self.date_relationship_visual = False
        self.relationship_visual = False
        self.test_visual = False
        self.train_visual = False

        if model_name == MODEL.REGRESSION.value:
            self.model = Regression()
        elif model_name == MODEL.CLASSIFICATION.value:
            self.model = Classification()

    def _get_selector(self, label: str, options: list[str], help: str | None = None):
        return st.sidebar.selectbox(label, options, help=help)

    def _get_model_params(self):
        if isinstance(self.model, Regression):
            return dict(input_features=MODEL_FEATURE.REGRESSION_INPUT.value,
                        output_features=MODEL_FEATURE.REGRESSION_OUTPUT.value)

        elif isinstance(self.model, Classification):
            return dict(input_features=MODEL_FEATURE.CLASSIFICATION_INPUT.value,
                        output_features=MODEL_FEATURE.CLASSIFICATION_OUTPUT.value)

    def set_components(self):
        model_params = self._get_model_params()

        st.sidebar.markdown('---')

        self.data_pre_process = True if st.sidebar.radio(
            'Select state of the data', ('Raw', 'Pre-Processed'),
            horizontal=True) == 'Pre-Processed' else False

        st.sidebar.markdown('---')

        if isinstance(self.model, Regression):
            self.date_relationship_visual = st.sidebar.button('Visualize Data', help='''Visualize 
                                                              all features relative to the date and 
                                                              show the correlation heatmap''')

            st.sidebar.markdown('---')

        self.selected_input_feature = self._get_selector('Select Input Feature',
                                                         model_params['input_features'],
                                                         help='''Select the input feature you want 
                                                                to visualize with relationship to 
                                                                the output feature''')

        self.selected_output_feature = self._get_selector(
            'Select Output Feature', model_params['output_features'],)

        self.relationship_visual = st.sidebar.button('Visualize Relationship')

        st.sidebar.markdown('---')

        self.test_size = st.sidebar.slider('Test Size (%)', 5, 30, 20)

        if isinstance(self.model, Classification):
            self.classification_kernel = self._get_selector('Kernel', CLASSIFICATION_KERNELS,
                                                            help='''Select the kernel for the 
                                                                   classification model''')
        col1, col2 = st.sidebar.columns([0.45, 0.55])

        self.train_visual = col1.button('Train Results')

        self.test_visual = col2.button('Prediction Results')

    def visualize_date_relationship(self):
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

            fig, ax = plt.subplots()
            sns.heatmap(self.model.data.drop(columns=date_feature_name).corr(method='pearson'),
                        annot=True, cmap='coolwarm')
            st.pyplot(fig)

    def visualize_feature_relationship(self):
        if self.relationship_visual:
            if (self.data_pre_process):
                # process the data
                pass

            st.scatter_chart(
                self.model.data[[self.selected_input_feature,
                                 self.selected_output_feature]],
                x=self.selected_input_feature, y=self.selected_output_feature)

    def _regression_result(self, mode: str):
        if self.data_pre_process:
            pass

        self.model.split_data(self.test_size / 100)
        self.model.train()
        self.model.predict()

        # TODO: Refactor this block
        prediction = pd.DataFrame(
            self.model.prediction[mode], columns=MODEL_FEATURE.REGRESSION_OUTPUT.value)

        for column in prediction:
            for input_column in MODEL_FEATURE.REGRESSION_INPUT.value:
                data = pd.DataFrame()
                if mode == MODEL_RESULT_MODE.TEST.value:
                    data[input_column] = pd.DataFrame(
                        self.model.X_test.values, columns=MODEL_FEATURE.REGRESSION_INPUT.value)[input_column]
                    data[mode] = pd.DataFrame(
                        self.model.Y_test.values, columns=MODEL_FEATURE.REGRESSION_OUTPUT.value)[column]
                else:
                    data[input_column] = pd.DataFrame(
                        self.model.X_train.values, columns=MODEL_FEATURE.REGRESSION_INPUT.value)[input_column]
                    data[mode] = pd.DataFrame(
                        self.model.Y_train.values, columns=MODEL_FEATURE.REGRESSION_OUTPUT.value)[column]
                data["prediction"] = prediction[column]
                st.write(column)
                st.scatter_chart(
                    data, x=input_column
                )

    def _classification_result(self, mode: str):
        if self.data_pre_process:
            pass

    def visualize_train(self):
        if self.train_visual:
            if isinstance(self.model, Regression):
                self._regression_result(MODEL_RESULT_MODE.TRAIN.value)
            elif isinstance(self.model, Classification):
                pass

    def visualize_prediction(self):
        if self.test_visual:
            if isinstance(self.model, Regression):
                self._regression_result(MODEL_RESULT_MODE.TEST.value)
            elif isinstance(self.model, Classification):
                pass
