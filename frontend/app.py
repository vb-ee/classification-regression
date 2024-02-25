import streamlit as st

# Model selection sidebar
selected_model = st.sidebar.selectbox(
    "Select Model", ("Default", "Regression", "Classification"))

# Depending on the selected model, show relevant options
if selected_model == "Regression":
    # Regression specific options
    st.sidebar.selectbox("Select Input Feature", ("Feature 1", "Feature 2", "Feature 3", "All Features Combined"),
                         help="Select the input feature you want to visualize with relationship to the output feature")
    output_feature = st.sidebar.selectbox(
        "Select Output Feature", ("Feature 1", "Feature 2"))
    st.sidebar.radio("Select kind of the data", ("Pre-Processed", "Raw"))
    st.sidebar.button("Visualize Relationship")
    st.sidebar.markdown("---")
    st.sidebar.slider("Test Size (%)", 5, 30, 10)
    st.sidebar.button("Prediction Results")
elif selected_model == "Classification":
    # Classification specific options
    st.sidebar.selectbox("Select Input Feature", ("Feature 1", "Feature 2", "Feature 3", "All Features Combined"),
                         help="Select the input feature you want to visualize with relationship to the output feature")
    st.sidebar.radio("Select kind of the data", ("Pre-Processed", "Raw"))
    st.sidebar.button("Visualize Relationship")
    st.sidebar.markdown("---")
    st.sidebar.slider("Test Size (%)", 5, 30, 10)
    st.sidebar.selectbox("Kernel", ("linear", "poly", "rbf", "sigmoid"),
                         help="Select the kernel for the classification model")
    st.sidebar.button("Prediction Results")
