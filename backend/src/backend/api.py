from fastapi import FastAPI
import uvicorn
from backend.config import conf
from backend.ml.utility import load_exp_config
from backend.ml.preprocessing import get_exp_features, get_example_dataset
from backend.ml.experiment import predict_experiment, explain_experiment, visualize_experiment
from backend.config import Row, PredRow, ImportanceRow, ExpName
from typing import Dict
import pandas as pd
from pydantic import BaseModel
from typing import List

app = FastAPI()
NUM_EXAMPLES = 10
DF_DICT_FORMAT = "index"
pd.option_context('display.max_rows', None, 'display.max_columns', None)

# NOTES:
# Returned dict objects are automatically converted to json at client side
# Additional parameters in function (not in url) are query parameters. Thesea are naturally strings, but get converted by fastAPI to defined types
# Post: FastAPi reads BODY of request as json and converts types to defined Datamodel (from type hint in function declaration)

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
async def api_exp_example_data(exp_name: ExpName) -> Dict[int, Row]:
    """
    Returns example data for given experment name.
    @param exp_name: experiment name
    @return first 10 rows of dataframe
    """
    
    exp_config = load_exp_config(exp_name.value)
    X, y = get_example_dataset(exp_config)
    df_dict = X.head(NUM_EXAMPLES).to_dict(orient=DF_DICT_FORMAT)
    
    return df_dict


@app.get("/{exp_name}/features")
async def api_exp_features(exp_name: ExpName) -> List[str]:
    """
    Returns feature names for given experiment name.
    @param exp_name: experiment name
    @return list of features
    """

    exp_config = load_exp_config(exp_name.value)
    return get_exp_features(exp_config)


@app.post("/{exp_name}/predict")
async def api_exp_predict(exp_name: ExpName, df_dict: Dict[int, Row]) -> Dict[int, PredRow]:
    """
    Returns predictions for posted DataFrame.
    @param exp_name: experiment name
    @param df_dict: DataFrame parsed as dict - IMPORTANT to define "Dict[int, Row]" annotation, otherwise parsing error
    @return predictions DataFrame parsed as dict
    """

    exp_config = load_exp_config(exp_name.value)

    # map DataModel back to dict
    df_dict = dict(zip(df_dict.keys(), map(BaseModel.dict, df_dict.values())))

    df = pd.DataFrame.from_dict(df_dict, orient=DF_DICT_FORMAT)

    # drop mocked id
    ids = df.loc[:, ["id"]]
    df.drop(columns=["id"])
    
    preds = predict_experiment(exp_config, df)

    # put id in front
    preds["id"] = ids
    df = df.loc[:, ["id"] + list(df.columns)]

    preds_dict = preds.to_dict(orient=DF_DICT_FORMAT)
    return preds_dict


@app.get("/{exp_name}/explain")
async def api_exp_explain(exp_name: ExpName) -> Dict[int, ImportanceRow]:
    """
    Returns feature importance data for all classifiers in given experiment name.
    @param exp_name: experiment name
    @return list of features
    """

    exp_config = load_exp_config(exp_name.value)
    # list pd.DataFrame
    explanation = explain_experiment(exp_config)[0]

    return explanation.to_dict(orient=DF_DICT_FORMAT)


@app.get("/{exp_name}/dt_data")
async def api_exp_explain(exp_name: ExpName):
    """
    Returns dot_data for all DecisionTrees in given experiment name
    @param exp_name: experiment name
    @return dot_data
    """

    exp_config = load_exp_config(exp_name.value)
   
    # string data for graphviz
    dot_data = visualize_experiment(exp_config)[0]


    return dot_data