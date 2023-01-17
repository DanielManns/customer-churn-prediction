from fastapi import FastAPI
import uvicorn
from backend.config import conf
from pydantic import BaseModel
from backend.ml.utility import load_exp_config
from backend.ml.preprocessing import get_features
from backend.config import Features
from backend.ml.experiment import predict_experiment

app = FastAPI()

# TODO:
#  1. Receive experiment config from frontend and trigger training
#  2. Receive example row from frontend, apply preprocessing (clean, enrich, transform), predict and supply result
#  3. Receive explain request (no data) and supply explanation data for models


def start_api():
    uvicorn.run(__name__ + ":app", reload=conf.reload, host=conf.host, port=conf.port, log_level="info")


@app.get("/")
async def root():
    return {"message": "Hello World"}

@app.get("/{exp_name}/features")
async def api_exp_features(exp_name: str):
    exp_config = load_exp_config(exp_name)
    return get_features(exp_config)


@app.post("/{exp_name}/predict")
async def api_exp_predict(modelname: str, data: Features):
    print(type(data))
    return data
