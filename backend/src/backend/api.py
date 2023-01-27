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
from enum import Enum

app = FastAPI()
JSON_FORMAT = "records"
DICT_FORMAT = "list"

# TODO:
#  1. Receive experiment config from frontend and trigger training
#  2. Receive example row from frontend, apply preprocessing (clean, enrich, transform), predict and supply result
#  3. Receive explain request (no data) and supply explanation data for models

# NOTES:
# Returned dict objects are automatically converted to json at client side
# Additional parameters in function (not in url) are query parameters. Thesea are naturally strings, but get converted by fastAPI to defined types
# Post: FastAPi reads BODY of request as json and converts types to defined Datamodel (from type hint in function declaration)

class ExpName(str, Enum):
    exp_no_subset = "exp_no_subset"
    exp_subset = "exp_subset"


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
async def api_exp_example_data(exp_name: ExpName):
    """
    Returns example data for given experment name.
    @param exp_name: experiment name
    @return first 10 rows of dataframe
    """
    
    exp_config = load_exp_config(exp_name.value)
    X, y = get_train_dataset(exp_config)
    df_dict = X.head(10).to_dict()
    
    return df_dict


@app.get("/{exp_name}/features")
async def api_exp_features(exp_name: ExpName):
    """
    Returns feature names for given experiment name.
    @param exp_name: experiment name
    @return list of features
    """
    exp_config = load_exp_config(exp_name.value)
    return get_features(exp_config)


@app.post("/{exp_name}/predict")
async def api_exp_predict(exp_name: ExpName, df_dict: dict):
    exp_config = load_exp_config(exp_name.value)

    print(type(df_dict))
    print(df_dict)
    df = pd.DataFrame.from_dict(df_dict)
    print(df.head())
    
    df = apply_preprocessing(df)
    
    
    preds = predict_experiment(exp_config, df)

    preds_dict = preds.to_dict()
    return preds_dict


