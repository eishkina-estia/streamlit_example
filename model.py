from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from pickle import dump, load
import pandas as pd


FEATURES = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Embarked']
FEATURES_OHE = ['Sex', 'Embarked']
TARGET = 'Survived'
TARGET_LABELS = {
    0: "Not survived",
    1: "Survived"
}
PATH_DATA = "data/train/titanic.csv"
PATH_PREPROCESSOR = "models/preprocessor.pkl"
PATH_MODEL = "models/clf_randomforest.pkl"

def load_data(path_data=PATH_DATA):

    df = pd.read_csv(path_data)
    X = df[FEATURES]
    y = df[TARGET]

    return X, y


def fit_and_persist_preprocessor(X: pd.DataFrame, path_preprocessor=PATH_PREPROCESSOR):

    column_transformer = ColumnTransformer([
        ('ohe', OneHotEncoder(), FEATURES_OHE)
    ], remainder='passthrough')

    column_transformer.fit(X)
    feature_names_out = column_transformer.get_feature_names_out()

    with open(path_preprocessor, "wb") as file:
        dump((column_transformer, feature_names_out), file)

    print(f"Preprocessor was saved to {path_preprocessor}")


def preprocess_data(X: pd.DataFrame, y: pd.Series = None, path_preprocessor=PATH_PREPROCESSOR):

    with open(path_preprocessor, "rb") as file:
        column_transformer, feature_names_transformed = load(file)

    if y is not None:
        X = X.dropna()
        y = y.iloc[X.index]

    X = column_transformer.transform(X)
    X = pd.DataFrame(X, columns=feature_names_transformed)

    if y is not None:
        return X, y
    else:
        return X


def fit_and_persist_model(X, y, path_model=PATH_MODEL):

    model = RandomForestClassifier()
    model.fit(X, y)

    y_pred_train = model.predict(X)
    accuracy = accuracy_score(y, y_pred_train)
    print(f"Model accuracy on train data is {accuracy:.3f}")

    with open(path_model, "wb") as file:
        dump(model, file)

    print(f"Model was saved to {path_model}")


def predict(X: pd.DataFrame, path_preprocessor=PATH_PREPROCESSOR, path_model=PATH_MODEL):

    X = preprocess_data(X)

    with open(path_model, "rb") as file:
        model = load(file)

    prediction = model.predict(X)[0]
    prediction_probas = model.predict_proba(X)[0]

    return prediction, prediction_probas


if __name__ == "__main__":

    X, y = load_data()
    fit_and_persist_preprocessor(X)
    X, y = preprocess_data(X, y)
    fit_and_persist_model(X, y)
