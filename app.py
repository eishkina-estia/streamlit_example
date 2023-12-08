import pandas as pd
import streamlit as st
from PIL import Image
from model import load_data, preprocess_data, load_model_and_predict


translation_Sex = {
    "Male": "male",
    "Female": "female"
}

translation_Embarked = {
    "Cherbourg": "C",
    "Queenstown": "Q",
    "Southampton": "S"
}

translation_Pclass = {
    "1st": 1,
    "2nd": 2,
    "3rd": 3
}

def process_main_page():
    show_main_page()
    process_side_bar_inputs()


def show_main_page():
    image = Image.open('img/titanic.jpg')

    st.set_page_config(
        layout="wide",
        initial_sidebar_state="auto",
        page_title="Demo Titanic",
        page_icon=image,

    )

    st.write(
        """
        # Machine Learning from Disaster
        Use machine learning to create a model that predicts which passengers survived the Titanic shipwreck.
        """
    )

    st.image(image)


def write_user_data(df):
    st.write("## Passenger's data")
    st.write(df)


def write_prediction(prediction, prediction_probas):
    st.write("## Prediction")
    st.write(prediction)

    st.write("## Probability")
    st.write(prediction_probas)


def preprocess_input(user_input):

    data = {
        "Pclass": translation_Pclass[user_input['pclass']],
        "Sex": translation_Sex[user_input['sex']],
        "Age": user_input['age'],
        "SibSp": user_input['sibsp'],
        "Parch": user_input['parch'],
        "Embarked": translation_Embarked[user_input['embarked']]
    }

    df = pd.DataFrame(data, index=[0])

    return df


def process_side_bar_inputs():

    st.sidebar.header('Passenger\'s data')
    user_input = sidebar_input_features()
    user_input_df = preprocess_input(user_input)
    write_user_data(user_input_df)

    X_user_input = preprocess_data(user_input_df)

    prediction, prediction_probas = load_model_and_predict(X_user_input)
    write_prediction(prediction, prediction_probas)


def sidebar_input_features():

    pclass = st.sidebar.selectbox(
        "Ticket class",
        ("1st", "2nd", "3rd"))

    sex = st.sidebar.selectbox("Sex", ("Male", "Female"))

    age = st.sidebar.slider(
        "Age (in years)",
        min_value=1, max_value=80, value=20, step=1)

    sibsp = st.sidebar.slider(
        "Number of siblings / spouses aboard the Titanic",
        min_value=0, max_value=10, value=0, step=1)

    parch = st.sidebar.slider(
        "Number of parents / children aboard the Titanic",
        min_value=0, max_value=10, value=0, step=1)

    embarked = st.sidebar.selectbox(
        "Port of Embarkation",
        ("Cherbourg", "Queenstown", "Southampton"))


    return {
        'pclass': pclass,
        'sex': sex,
        'age': age,
        'sibsp': sibsp,
        'parch': parch,
        'embarked': embarked
    }


if __name__ == "__main__":
    process_main_page()
