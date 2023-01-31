import gradio as gr

from frontend.config import conf
import requests
import pandas as pd
import time

# TODO:
#  - Request example row from backend
#  - (Optionally) Configure example row
#  - Send example row to backend for prediction
#  - Display prediction
#  6. Obtain clf explanation from backend (feature importance)
#  7. Display clf explanation with plots in third tab
#  1. Configure experiment in first tab
#  2. Send experiment data to backend

TIMEOUT = 30
RETRY_INTERVAL = 1
EXP_NAME = "exp_no_subset"
NUM_CLASSIFIERS = 2
ENDPOINT = f"http://{conf.backend_host}:{conf.backend_port}"
HELLO_WORLD_ENDPOINT = f"{ENDPOINT}/"
EXAMPLE_ENDPOINT = f"{ENDPOINT}/{EXP_NAME}/example_data"
PREDICT_ENDPOINT = f"{ENDPOINT}/{EXP_NAME}/predict"

DF_DICT_FORMAT = "index"

def connect():
    t = 0
    st = time.time()
    con = False
    while t < TIMEOUT:
        try:
            requests.get(HELLO_WORLD_ENDPOINT)
            s = requests.session()
            print("backend connection established")
            con = True
            break
        except:
            time.sleep(RETRY_INTERVAL)
            t = time.time() - st
            print("retrying connect...")
            print(f"time elapsed: {round(t, 1)}")
    return con
        

def request_examples() -> pd.DataFrame:
    # response body is automatically json by FASTAPI
    # response.text gives json content
    # response.json() DECODES json content to list object

    response = requests.get(EXAMPLE_ENDPOINT)
    return pd.read_json(response.text, orient=DF_DICT_FORMAT)
    

def request_prediction(input_df: pd.DataFrame) -> pd.DataFrame:
    payload = input_df.to_dict(orient=DF_DICT_FORMAT)
    
    response = requests.post(PREDICT_ENDPOINT, json=payload)
    
    df = pd.read_json(response.text, orient=DF_DICT_FORMAT)
    return df


def run_gui():
    example_df = request_examples()
    inputs = [gr.Dataframe(row_count=(1, "dynamic"), col_count=(len(example_df.columns), "fixed"), label="Input Data", interactive=True)]

    outputs = [
        gr.Dataframe(row_count=(1, "dynamic"), col_count=(NUM_CLASSIFIERS, "fixed"), label="Predictions", headers=["Logistic Regression", "Decision Tree"])]


    interface = gr.Interface(fn=request_prediction, inputs=inputs, outputs=outputs, examples=[example_df], examples_per_page=10)
    interface.launch(server_name=conf.frontend_host, server_port=conf.frontend_port)

if __name__ == "__main__":
    connect()
    run_gui()