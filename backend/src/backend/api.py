from fastapi import FastAPI
import uvicorn

app = FastAPI()
port = 8000
host = "0.0.0.0"
reload = True


def start_api():
    uvicorn.run(__file__ + ":" + app.__str__(), reload=reload, host=host, port=port, log_level="info")


@app.get("/")
async def root():
    return {"message": "Hello World"}


