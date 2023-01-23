from fastapi import FastAPI
import uvicorn
from backend.config import conf
from pydantic import BaseModel
from backend.ml.utility import load_exp_config
from backend.ml.preprocessing import get_features, get_train_dataset
from backend.config import Features
from backend.ml.experiment import predict_experiment
import pandas as pd
from backend.ml.preprocessing import apply_preprocessing
import requests

app = FastAPI()
JSON_FORMAT = "records"

# TODO:
#  1. Receive experiment config from frontend and trigger training
#  2. Receive example row from frontend, apply preprocessing (clean, enrich, transform), predict and supply result
#  3. Receive explain request (no data) and supply explanation data for models


def start_api():
    """
    Starts fastapi on host and port specified in config.
    @returns None
    """

    uvicorn.run(__name__ + ":app", reload=conf.reload, host=conf.host, port=conf.port, log_level="info")


@app.get("/")
async def root():
    return {"message": "Hello World"}


@app.get("/{exp_name}/example_data")
async def api_exp_example_data(exp_name: str) -> str:
    """
    Returns example data for given experment name.
    @param exp_name: experiment name
    @return first 10 rows of dataframe
    """
    
    exp_config = load_exp_config(exp_name)
    X, y = get_train_dataset(exp_config)
    df_json = X.head(10).to_json(orient=JSON_FORMAT)
    
    return df_json


@app.get("/{exp_name}/features")
async def api_exp_features(exp_name: str) -> list[str]:
    """
    Returns feature names for given experiment name.
    @param exp_name: experiment name
    @return list of features
    """
    exp_config = load_exp_config(exp_name)
    return get_features(exp_config)


@app.post("/{exp_name}/predict")
async def api_exp_predict(exp_name: str, df_json: str) -> str:
    exp_config = load_exp_config(exp_name)

    df = apply_preprocessing(pd.read_json(df_json, orient=JSON_FORMAT))
    
    preds = predict_experiment(exp_config, df)
    
    return preds.to_json(orient=JSON_FORMAT)


