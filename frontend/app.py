# buildt-in imports
import sys
from os.path import dirname
from constants import MODEL

# add project directory to the path
sys.path.append(dirname(dirname(__file__)))  # nopep8

# third-party imports
import streamlit as st

# custom imports
from side_bar import SideBar
from constants import MODEL


def main():
    # Model selection sidebar
    selected_model = st.sidebar.selectbox(
        "Select Model", [m.value for m in MODEL])

    side_bar = SideBar(selected_model)

    side_bar.set_components()
    # visualize_relationship()
    # visualize_prediction()


if __name__ == "__main__":
    main()
