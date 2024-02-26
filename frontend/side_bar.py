# third-party imports
import streamlit as st

# custom imports
from constants import MODEL, CLASSIFICATION_KERNELS
from src.utils import MODEL_FEATURE


class SideBar:
    def __init__(self, model: str):
        self.model = model
        self.selected_input_feature = None
        self.selected_output_feature = None
        self.data_is_pre_processed = False
        self.test_size = None
        self.classification_kernel = None

    def get_selector(self, label: str, options: list[str], help: str | None = None):
        return st.sidebar.selectbox(label, options, help=help)

    def get_model_params(self, model: str):
        if model == MODEL.REGRESSION.value:
            return dict(input_features=MODEL_FEATURE.REGRESSION_INPUT.value,
                        ouput_features=MODEL_FEATURE.REGRESSION_OUTPUT.value)

        elif model == MODEL.CLASSIFICATION.value:
            return dict(input_features=MODEL_FEATURE.CLASSIFICATION_INPUT.value,
                        ouput_features=MODEL_FEATURE.CLASSIFICATION_OUTPUT.value)

    def set_components(self):
        model_params = self.get_model_params(self.model)

        self.selected_input_feature = self.get_selector("Select Input Feature",
                                                        model_params["input_features"],
                                                        help="""Select the input feature you want 
                                                                to visualize with relationship to 
                                                                the output feature""")

        self.selected_ouput_feature = self.get_selector(
            "Select Output Feature", model_params["ouput_features"],)

        self.data_is_pre_processed = True if st.sidebar.radio(
            "Select kind of the data", ("Raw", "Pre-Processed")) == "Pre-Processed" else False

        st.sidebar.markdown("---")

        self.test_size = st.sidebar.slider("Test Size (%)", 5, 30, 20)

        if self.model == MODEL.CLASSIFICATION.value:
            self.classification_kernel = self.get_selector("Kernel", CLASSIFICATION_KERNELS,
                                                           help="""Select the kernel for the 
                                                                   classification model""")
