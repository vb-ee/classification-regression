# third-party imports
import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

# custom imports
from src.utils import MODEL_FEATURE, MODEL, CLASSIFICATION_KERNELS, matlab_time_to_datetime, centered_moving_average
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
        self.prediction_visual = False
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

        self.prediction_visual = col2.button('Prediction Results')

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

    def visualize_prediction(self):
        if self.prediction_visual:
            if isinstance(self.model, Regression):
                point = int(len(self.model.X) * self.test_size / 100)
                if self.data_pre_process:
                    self.model.X = centered_moving_average(self.model.X, MODEL_FEATURE.REGRESSION_INPUT.value, 80)

                self.model.split_data(point)
                self.model.train()
                data = pd.DataFrame()
                prediction = pd.DataFrame(self.model.predict()["test"], columns=MODEL_FEATURE.REGRESSION_OUTPUT.value)

                for column in prediction:
                    data["Test"] = pd.DataFrame(
                        self.model.Y_test.values, columns=MODEL_FEATURE.REGRESSION_OUTPUT.value)[column]
                    data["Prediction"] = prediction[column]
                    st.write(column)
                    st.scatter_chart(
                        data
                    )

    def visualize_train(self):
        if self.train_visual:
            if isinstance(self.model, Regression):
                point = int(len(self.model.X) * self.test_size / 100)
                if self.data_pre_process:
                    self.model.X = centered_moving_average(self.model.X, MODEL_FEATURE.REGRESSION_INPUT.value, 80)

                self.model.split_data(point)
                self.model.train()
                data = pd.DataFrame()
                prediction = pd.DataFrame(self.model.predict()["train"], columns=MODEL_FEATURE.REGRESSION_OUTPUT.value)

                for column in prediction:
                    data["Train"] = pd.DataFrame(
                        self.model.Y_train.values, columns=MODEL_FEATURE.REGRESSION_OUTPUT.value)[column]
                    data["Prediction"] = prediction[column]
                    st.write(column)
                    st.scatter_chart(
                        data
                    )
