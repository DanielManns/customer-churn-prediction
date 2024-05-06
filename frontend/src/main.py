import gradio as gr

from frontend.config import conf
import requests
import pandas as pd
import time
from frontend.plotting import plot_feature_importance, plot_feature_importance
from frontend.plotting import plot_dot_dt
from functools import partial
from typing import List
import json

EXP_NAME = "exp_no_subset"
NUM_CLASSIFIERS = 1
CLF_HEADERS = ["id", "DecisionTreeClassifier"]
ENDPOINT = f"http://{conf.backend_host}:{conf.backend_port}"
HELLO_WORLD_ENDPOINT = f"{ENDPOINT}/"
EXAMPLE_ENDPOINT = f"{ENDPOINT}/{EXP_NAME}/example_data"
PREDICT_ENDPOINT = f"{ENDPOINT}/{EXP_NAME}/predict"
FEATURE_IMPORTANCE_ENDPOINT = f"{ENDPOINT}/{EXP_NAME}/explain"
DT_ENDPOINT = f"{ENDPOINT}/{EXP_NAME}/dt_data"

DF_DICT_FORMAT = "index"
CONNECTION = False

def connect() -> bool:
    """
    Establishes connection to backend with static timeout and retry parameters.
    @return True if success, otherwise False
    """

    t = 0
    st = time.time()
    con = False
    while t < conf.timeout:
        try:
            requests.get(HELLO_WORLD_ENDPOINT)
            con = True
            print("backend connection established")
            break
        except:
            time.sleep(conf.retry_interval)
            t = time.time() - st
            print("retrying connect...")
            print(f"time elapsed: {round(t, 1)}")
    return con
        

def request_examples() -> pd.DataFrame:
    """
    Requests example data for experiment from backend.
    @return pd.DataFrame - example data
    """
    # response body is automatically json by FASTAPI
    # response.text gives json content
    # response.json() DECODES json content to list object

    response = requests.get(EXAMPLE_ENDPOINT)
    return pd.read_json(response.text, orient=DF_DICT_FORMAT)


def request_prediction(input_df: pd.DataFrame) -> pd.DataFrame:
    """
    Requests prediction for given dataframe from backend.
    @param input_df: pd.DataFrame - dataframe to obtain prediction for
    @return pd.DataFrame - predictions
    """

    payload = input_df.to_dict(orient=DF_DICT_FORMAT)
    response = requests.post(PREDICT_ENDPOINT, json=payload)
    
    df = pd.read_json(response.text, orient=DF_DICT_FORMAT)
    return df


def request_feature_importance() -> pd.DataFrame:
    """
    Requests explanation for experiment from backend.
    @return pd.DataFrame - explanations
    """

    response = requests.get(FEATURE_IMPORTANCE_ENDPOINT)
    df = pd.read_json(response.text, orient=DF_DICT_FORMAT)
    return df

def request_dts() -> List[str]:
    response = requests.get(DT_ENDPOINT)
    decoded = json.loads(response.text)
    return decoded



def run_gui():
    """
    Starts gradio gui.
    @return None
    """

    CONNECTION = connect()

    if CONNECTION:
        dot_dts = request_dts()
        example_df = request_examples()
        expl = request_feature_importance()
        choices = list(range(len(expl.index)))
        importance_choices = ["Mean"] + list(range(len(expl.index)))
        importance_fn = partial(plot_feature_importance, expl)
        dt_fn = partial(plot_dot_dt, dot_dts)

        # Hierarichal gui definition
        with gr.Blocks() as ui:

            ### Predict tab
            with gr.Tab("Predict"):
                with gr.Row():
                    with gr.Column():
                        input = gr.Dataframe(row_count=(1, "dynamic"), col_count=(len(example_df.columns), "fixed"), label="Input Data", interactive=True)
                        examples = gr.Examples([example_df], input)
                    with gr.Column():
                        output = gr.Dataframe(row_count=(1, "dynamic"), col_count=(NUM_CLASSIFIERS + 1, "fixed"), label="Predictions", headers=CLF_HEADERS)
                button = gr.Button("predict")
                button.click(request_prediction, inputs=[input], outputs=[output])

            ### Feature Importance tab
            with gr.Tab("Feature Importance"):
                    with gr.Row():
                        with gr.Column():
                            importance_input = gr.Dropdown(choices=importance_choices, value=0, label="Tree Number")
                        with gr.Column():
                            importance_plot = gr.Plot()
                        importance_input.change(fn=importance_fn, inputs=importance_input, outputs=importance_plot)

            ### Decision Tree tab
            with gr.Tab("Decision Tree"):
                    with gr.Row():
                        with gr.Column():
                            dt_input = gr.Dropdown(choices=choices, value=0, label="Tree Number")
                        with gr.Column():
                            dt_plot = gr.Plot(label="DecisionTree")
                        dt_input.change(fn=dt_fn, inputs=dt_input, outputs=dt_plot)

            # load data from dropdown input to plot
            ui.load(fn=importance_fn, inputs=importance_input, outputs=importance_plot)
            ui.load(fn=dt_fn, inputs=dt_input, outputs=dt_plot)

                

        ui.launch(server_name=conf.frontend_host, server_port=conf.frontend_port)
    else:
        print("Connection could not be established")

if __name__ == "__main__":
    run_gui()