# built-in imports
import sys
from os.path import dirname

# add project directory to the path
sys.path.append(dirname(dirname(__file__)))  # nopep8

# third-party imports
import streamlit as st

# custom imports
from user_interface import UserInterface
from src.utils import MODEL


def main():
    # Model selection
    selected_model = st.sidebar.selectbox(
        'Select Model', [m.value for m in MODEL])

    user_interface = UserInterface(selected_model)

    user_interface.set_components()

    user_interface.visualize_feature_relationship()

    user_interface.visualize_date_relationship()
    # visualize_prediction()


if __name__ == '__main__':
    main()
