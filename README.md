# Streamlit Demo

Inspired by [Streamlit demo](https://github.com/evgpat/streamlit_demo)

This project demonstrates how to present machine learning solution as a web application using [Streamlit](https://www.streamlit.io/) framework. The data used in this repo is the [Titanic dataset](https://www.kaggle.com/c/titanic) from Kaggle.

## Files

- `data/train/titanic.csv`: train data
- `models/titanic.model`: model fitted on train data
- `app.py`: streamlit app file
- `model.py`: script for fitting and saving a model
- `requirements.txt`: package requirements file
- `Dockerfile`: for docker deployment

## Run Demo Locally 

### Shell

To run streamlit app locally, in the repo root folder do the following:

```shell
$ python -m venv venv
$ source venv/bin/activate
$ pip install -r requirements.txt
$ streamlit run app.py
```
Open http://localhost:8501 to view the app UI.

### Docker

To build and run the docker image named `st-demo`:

```
$ docker build -t st-demo .
$ docker run -it --rm -p '8501:8501' st-demo
```

`-it` keeps the terminal interactive

`--rm` removes the image once the command is stopped (e.g. using Ctrl+C)

Open http://localhost:8501/ to view the app.

## Deploy on Streamlit Cloud
 
1. Put your app on GitHub: make sure it's in a public repo and that you have a `requirements.txt` file.
 
2. Sign in on [Streamlit Cloud](https://share.streamlit.io/signup) using your GitHub or Google account, or create a new account using an email (see [documentation](https://docs.streamlit.io/streamlit-community-cloud/get-started/quickstart)).
 
3. Deploy and share (see [documentation](https://docs.streamlit.io/streamlit-community-cloud/deploy-your-app)): click "New app", then fill in your repo, branch, and file path, choose a Python version (3.9 for this demo) and click "Deploy", then you should be able to see your app.
