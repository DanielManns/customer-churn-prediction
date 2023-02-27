import gradio as gr

from frontend.config import conf
import requests
import pandas as pd
import time
from frontend.plotting import plot_feature_importance, plot_feature_importance
from frontend.plotting import plot_dt
from functools import partial

# TODO:
#  1. Obtain image plot from backend? as bytes object / check if model can be send as bytes object
#  Solution: Send graphviz dot data via api and only plot here in frontend
#  -> sklearn.tree export_graphviz
#  Add tree plot
#  Add button for type of graph
#  6. Obtain clf explanation from backend (feature importance)
#  7. Display clf explanation with plots in third tab
#  1. Configure experiment in first tab
#  2. Send experiment data to backend
#  - (Optionally) Display error message when input incorrect


EXP_NAME = "exp_no_subset"
NUM_CLASSIFIERS = 1
CLF_HEADERS = ["id", "Decision Tree"]
ENDPOINT = f"http://{conf.backend_host}:{conf.backend_port}"
HELLO_WORLD_ENDPOINT = f"{ENDPOINT}/"
EXAMPLE_ENDPOINT = f"{ENDPOINT}/{EXP_NAME}/example_data"
PREDICT_ENDPOINT = f"{ENDPOINT}/{EXP_NAME}/predict"
EXPLAIN_ENDPOINT = f"{ENDPOINT}/{EXP_NAME}/explain"

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


def request_explanation() -> pd.DataFrame:
    """
    Requests explanation for experiment from backend.
    @return pd.DataFrame - explanations
    """

    response = requests.get(EXPLAIN_ENDPOINT)
    df = pd.read_json(response.text, orient=DF_DICT_FORMAT)
    return df


def run_gui():
    """
    Starts gradio gui.
    @return None
    """

    CONNECTION = connect()

    if CONNECTION:
        example_df = request_examples()
        expl = request_explanation()
        choices = ["avg"] + list(range(len(expl.index)))
        importance_fn = partial(plot_feature_importance, expl)

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
                            input_data = gr.Dropdown(choices=choices, value=0, label="Tree Number")
                        with gr.Column():
                            exp_plot = gr.Plot(label="DecisionTree")
                        input_data.change(fn=importance_fn, inputs=input_data, outputs=exp_plot)

            # load data from dropdown input to plot
            ui.load(fn=importance_fn, inputs=input_data, outputs=exp_plot)
                

        ui.launch(server_name=conf.frontend_host, server_port=conf.frontend_port)
    else:
        print("Connection could not be established")

if __name__ == "__main__":
    # add id column in left side somehow
    # für demo interessanten example datensatz raussuchen
    # only plot average feature importance of dt with error bars
    # maybe add details tab for separate views
    run_gui()