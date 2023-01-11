from fastapi import FastAPI
import uvicorn
from config import BackendConfig

app = FastAPI()
conf = BackendConfig.from_yaml()


def start_api():
    uvicorn.run(__file__ + ":" + app.__str__(), reload=conf.reload, host=conf.host, port=conf.port, log_level="info")


@app.get("/")
async def root():
    return {"message": "Hello World"}


