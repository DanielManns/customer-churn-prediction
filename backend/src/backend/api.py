from fastapi import FastAPI
import uvicorn
from config import BackendConfig

app = FastAPI()
conf = BackendConfig.from_yaml()

# TODO:
#  1. Receive experiment config from frontend and trigger training
#  2. Receive example row from frontend, apply preprocessing (clean, enrich, transform), predict and supply result
#  3. Receive explain request (no data) and supply explanation data for models
#  4. Add documentation at get("/") to explain different endpoints


def start_api():
    uvicorn.run(__name__ + ":app", reload=conf.reload, host=conf.host, port=conf.port, log_level="info")


@app.get("/")
async def root():
    return {"message": "Hello World"}



