# third-party imports
import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt

# custom imports
from constants import MODEL, CLASSIFICATION_KERNELS
from src.utils import MODEL_FEATURE, DataProcess, DATA_PATH


class UserInterface:
    def __init__(self, model: str):
        self.model = model
        self.selected_input_feature = None
        self.selected_output_feature = None
        self.data_pre_process = False
        self.test_size = None
        self.classification_kernel = None
        self.date_relationship_visual = False
        self.relationship_visual = False
        self.prediction_visual = False
        self.train_visual = False

        if self.model == MODEL.REGRESSION.value:
            self.data = DataProcess(DATA_PATH.REGRESSION_RAW.value).data
        elif self.model == MODEL.CLASSIFICATION.value:
            self.data = DataProcess(DATA_PATH.CLASSIFICATION_RAW.value).data

    def _get_selector(self, label: str, options: list[str], help: str | None = None):
        return st.sidebar.selectbox(label, options, help=help)

    def _get_model_params(self, model: str):
        if model == MODEL.REGRESSION.value:
            return dict(input_features=MODEL_FEATURE.REGRESSION_INPUT.value,
                        output_features=MODEL_FEATURE.REGRESSION_OUTPUT.value)

        elif model == MODEL.CLASSIFICATION.value:
            return dict(input_features=MODEL_FEATURE.CLASSIFICATION_INPUT.value,
                        output_features=MODEL_FEATURE.CLASSIFICATION_OUTPUT.value)

    def set_components(self):
        model_params = self._get_model_params(self.model)

        st.sidebar.markdown("---")

        self.data_pre_process = True if st.sidebar.radio(
            "Select state of the data", ("Raw", "Pre-Processed"),
            horizontal=True) == "Pre-Processed" else False

        st.sidebar.markdown("---")

        if self.model == MODEL.REGRESSION.value:
            self.date_relationship_visual = st.sidebar.button("Visualize Data", help="""Visualize 
                                                              all features relative to the date and 
                                                              show the correlation heatmap""")

            st.sidebar.markdown("---")

        self.selected_input_feature = self._get_selector("Select Input Feature",
                                                         model_params["input_features"],
                                                         help="""Select the input feature you want 
                                                                to visualize with relationship to 
                                                                the output feature""")

        self.selected_output_feature = self._get_selector(
            "Select Output Feature", model_params["output_features"],)

        self.relationship_visual = st.sidebar.button("Visualize Relationship")

        st.sidebar.markdown("---")

        self.test_size = st.sidebar.slider("Test Size (%)", 5, 30, 20)

        if self.model == MODEL.CLASSIFICATION.value:
            self.classification_kernel = self._get_selector("Kernel", CLASSIFICATION_KERNELS,
                                                            help="""Select the kernel for the 
                                                                   classification model""")
        col1, col2 = st.sidebar.columns([0.45, 0.55])

        self.train_visual = col1.button("Train Results")

        self.prediction_visual = col2.button("Prediction Results")

    def visualize_date_relationship(self):
        if self.date_relationship_visual:
            data = self.data.drop(columns=["Datum"])
            if self.data_pre_process:
                # process the data
                pass

            for feature in data.columns:
                st.line_chart(
                    self.data[["Datum", feature]], x="Datum", y=feature)

            fig, ax = plt.subplots()
            sns.heatmap(data.corr(method='pearson'),
                        annot=True, cmap="coolwarm")
            st.pyplot(fig)

    def visualize_feature_relationship(self):
        if self.relationship_visual:
            if (self.data_pre_process):
                # process the data
                pass

            st.scatter_chart(
                self.data[[self.selected_input_feature,
                           self.selected_output_feature]],
                x=self.selected_input_feature, y=self.selected_output_feature)

    def visualize_prediction(self):
        if self.prediction_visual:
            pass

    def visualize_train(self):
        if self.train_visual:
            pass
