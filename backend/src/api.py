from fastapi import FastAPI

app = None


def start_api():
    global app
    app = FastAPI()

@app.get("/")
async def root():
    return {"message": "Hello World"}


