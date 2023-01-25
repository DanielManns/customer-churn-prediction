import gradio as gr

from frontend.config import conf
import requests
import pandas as pd

# TODO:
#  - Request example row from backend
#  - (Optionally) Configure example row
#  - Send example row to backend for prediction
#  - Display prediction
#  6. Obtain clf explanation from backend (feature importance)
#  7. Display clf explanation with plots in third tab
#  1. Configure experiment in first tab
#  2. Send experiment data to backend

EXP_NAME = "exp_no_subset"
NUM_CLASSIFIERS = 2
ENDPOINT = f"http://{conf.backend_host}:{conf.backend_port}/{EXP_NAME}"
EXAMPLE_ENDPOINT = f"{ENDPOINT}/example_data"
PREDICT_ENDPOINT = f"/{ENDPOINT}/predict"


def request_examples():
    response = requests.get(EXAMPLE_ENDPOINT)
    return pd.read_json(response.json())
    

def request_inference(input_dataframe: pd.DataFrame) -> pd.DataFrame:
    response = requests.post(PREDICT_ENDPOINT, input_dataframe.to_json())
    return pd.read_json(response.json())


def run_gui():
    example_df = request_examples()
    # df, _ = get_preprocessed_dataset("mixed", exp_config["features"]["is_subset"], mode="test")
    inputs = [gr.Dataframe(row_count=(1, "dynamic"), col_count=(len(example_df.columns), "fixed"), label="Input Data", interactive=True)]

    outputs = [
        gr.Dataframe(row_count=(1, "dynamic"), col_count=(NUM_CLASSIFIERS, "fixed"), label="Predictions", headers=["Logistic Regression", "Decision Tree"])]

    # LR_clfs = load_clfs(exp_config["name"], "LogisticRegression", exp_config["cross_validation"]["n_splits"])
    # DT_clfs = load_clfs(exp_config["name"], "DecisionTreeClassifier", exp_config["cross_validation"]["n_splits"])

    interface = gr.Interface(fn=request_inference, inputs=inputs, outputs=outputs, examples=[example_df], examples_per_page=10)
    interface.launch(server_name=conf.frontend_host, server_port=conf.frontend_port)

if __name__ == "__main__":
    run_gui()