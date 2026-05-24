import streamlit as st
import pandas as pd
import numpy as np

from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Train Model

@st.cache_resource
def train_model():

    # Load Iris Dataset
    iris = load_iris()

    X = iris.data
    y = iris.target

    feature_names = iris.feature_names
    target_names = iris.target_names

    # Train Test Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42
    )

    # Decision Tree Model
    model = DecisionTreeClassifier(
        criterion="gini",
        max_depth=3,
        random_state=42
    )

    model.fit(X_train, y_train)

    # Accuracy
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    return model, accuracy, feature_names, target_names


model, accuracy, feature_names, target_names = train_model()

# Streamlit UI
st.title("Iris Flower Prediction App")

st.write(
    "Enter flower measurements below to predict the Iris flower species."
)

# Sidebar
st.sidebar.header("Flower Measurements")

# Input Fields
sepal_length = st.sidebar.number_input(
    "Sepal Length (cm)",
    min_value=0.0,
    value=5.1,
    step=0.1
)

sepal_width = st.sidebar.number_input(
    "Sepal Width (cm)",
    min_value=0.0,
    value=3.5,
    step=0.1
)

petal_length = st.sidebar.number_input(
    "Petal Length (cm)",
    min_value=0.0,
    value=1.4,
    step=0.1
)

petal_width = st.sidebar.number_input(
    "Petal Width (cm)",
    min_value=0.0,
    value=0.2,
    step=0.1
)

# Create Input Array
input_data = np.array([
    [
        sepal_length,
        sepal_width,
        petal_length,
        petal_width
    ]
])

# Predict Button
if st.button("Predict Species"):

    prediction = model.predict(input_data)[0]
    prediction_name = target_names[prediction]

    st.success(f"Predicted Flower Species: {prediction_name}")


# Show Accuracy

st.subheader("Model Accuracy")

st.write(f"Accuracy: {accuracy * 100:.2f}%")


# Show Input Data

st.subheader("Input Values")

input_df = pd.DataFrame(
    input_data,
    columns=feature_names
)

st.dataframe(input_df)
