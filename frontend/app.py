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
    selected_model = st.selectbox(
        'Select Model', [m.value for m in MODEL])

    user_interface = UserInterface(selected_model)

    user_interface.set_components()

    user_interface.visualize_feature_relationship()

    user_interface.visualize_date_relationship()

    user_interface.visualize_train()

    user_interface.visualize_prediction()


if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        print(f"An error occurred: {e}")
        st.error(f"Unknown error occurred. Please update the page and try again.")
