import gradio as gr

from frontend.config import conf
import requests
import pandas as pd
import time
from frontend.plotting import plot_feature_importance
from frontend.plotting import plot_dt

# TODO:
#  - (Optionally) Configure example row
#  - Display error message when input incorrect
#  6. Obtain clf explanation from backend (feature importance)
#  7. Display clf explanation with plots in third tab
#  1. Configure experiment in first tab
#  2. Send experiment data to backend


EXP_NAME = "exp_no_subset"
NUM_CLASSIFIERS = 1
CLF_HEADERS = ["Decision Tree"]
ENDPOINT = f"http://{conf.backend_host}:{conf.backend_port}"
HELLO_WORLD_ENDPOINT = f"{ENDPOINT}/"
EXAMPLE_ENDPOINT = f"{ENDPOINT}/{EXP_NAME}/example_data"
PREDICT_ENDPOINT = f"{ENDPOINT}/{EXP_NAME}/predict"
EXPLAIN_ENDPOINT = f"{ENDPOINT}/{EXP_NAME}/explain"

DF_DICT_FORMAT = "index"

def connect():
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


def request_explanation() -> pd.DataFrame:
    response = requests.get(EXPLAIN_ENDPOINT)
    df = pd.read_json(response.text, orient=DF_DICT_FORMAT)
    return df

def run_gui():
    example_df = request_examples()
    expl = request_explanation()
    with gr.Blocks() as ui:
        with gr.Tab("predict_tab"):
            with gr.Row():
                with gr.Column():
                    input = gr.Dataframe(row_count=(1, "dynamic"), col_count=(len(example_df.columns), "fixed"), label="Input Data", interactive=True)
                    examples = gr.Examples([example_df], input)
                with gr.Column():
                    output = gr.Dataframe(row_count=(1, "dynamic"), col_count=(NUM_CLASSIFIERS, "fixed"), label="Predictions", headers=CLF_HEADERS)
            button = gr.Button("predict")
            button.click(request_prediction, inputs=[input], outputs=[output])
        with gr.Tab("explain_tab"):
                with gr.Row():
                    with gr.Column():
                        gr.Plot(plot_feature_importance(expl, "DT Feature Importance"))
                    with gr.Column():
                        # gr.plot(plot_dt())
                        pass
            

    ui.launch(server_name=conf.frontend_host, server_port=conf.frontend_port)
    #interface = gr.Interface(fn=request_prediction, inputs=inputs, outputs=outputs, examples=[example_df], examples_per_page=10)
    #interface.launch(server_name=conf.frontend_host, server_port=conf.frontend_port)

if __name__ == "__main__":
    connect()
    run_gui()