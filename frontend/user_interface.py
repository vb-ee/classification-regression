# third-party imports
import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

# custom imports
from src.utils import MODEL_FEATURE, MODEL, CLASSIFICATION_KERNELS, REGRESSION_ORDERS, matlab_time_to_datetime, \
    MODEL_RESULT_MODE
from src.classes import Regression, Classification


class UserInterface:
    def __init__(self, model_name: str):
        self.selected_input_feature = None
        self.selected_output_feature = None
        self.data_pre_process = False
        self.test_size = None
        self.regression_order = 1
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
            'Select Output Feature', model_params['output_features'], )

        self.relationship_visual = st.sidebar.button('Visualize Relationship')

        st.sidebar.markdown('---')

        self.test_size = st.sidebar.slider('Test Size (%)', 5, 30, 20)

        if isinstance(self.model, Classification):
            self.classification_kernel = self._get_selector('Kernel', CLASSIFICATION_KERNELS,
                                                            help='''Select the kernel for the 
                                                                   classification model''')

        if isinstance(self.model, Regression):
            self.regression_order = self._get_selector('Order', REGRESSION_ORDERS,
                                                       help='''if order equals 1 then linear regressor is used, 
                                                            if order is bigger than 1 then polynomial regression is 
                                                            used''')

        col1, col2 = st.sidebar.columns([0.45, 0.55])

        self.train_visual = col1.button('Train Results')

        self.test_visual = col2.button('Test Results')

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

    def _get_selector(self, label: str, options: list[str], help: str | None = None):
        return st.sidebar.selectbox(label, options, help=help)

    def _get_model_params(self):
        if isinstance(self.model, Regression):
            return dict(input_features=MODEL_FEATURE.REGRESSION_INPUT.value,
                        output_features=MODEL_FEATURE.REGRESSION_OUTPUT.value)

        elif isinstance(self.model, Classification):
            return dict(input_features=MODEL_FEATURE.CLASSIFICATION_INPUT.value,
                        output_features=MODEL_FEATURE.CLASSIFICATION_OUTPUT.value)

    def _show_model_result(self, mode: str, X, Y, prediction):
        i = 0
        columns = X.columns.values
        df = pd.DataFrame(X, columns=columns)
        for y in Y:
            df[mode] = Y[y]
            df['prediction'] = prediction[:, i]
            for x in columns:
                st.write(y)
                st.scatter_chart(df[[mode, 'prediction', x]], x=x)

            i += 1

    def _regression_result(self, mode: str):
        if self.data_pre_process:
            pass

        self.model.split_data(self.test_size / 100)
        self.model.train(self.regression_order)
        self.model.predict()
        self.model.evaluate()

        # show the evaluation
        st.header('Evaluation of ' + mode + ': ')
        for i in range(len(self.model.evaluation)):
            st.write(MODEL_FEATURE.REGRESSION_OUTPUT.value[i])
            st.write(self.model.evaluation[i][mode])
        st.markdown('---')
        st.header('Visualization of ' + mode + ': ')

        # show the scatter
        if mode == MODEL_RESULT_MODE.TRAIN.value:
            self._show_model_result(
                mode, self.model.X_train, self.model.Y_train, self.model.prediction[mode])

        elif mode == MODEL_RESULT_MODE.TEST.value:
            self._show_model_result(
                mode, self.model.X_test, self.model.Y_test, self.model.prediction[mode])

    def _classification_result(self, mode: str):
        if self.data_pre_process:
            pass
