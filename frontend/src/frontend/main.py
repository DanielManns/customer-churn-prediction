import gradio as gr

from frontend.src.frontend.config import FrontendConfig

conf = FrontendConfig.from_yaml()


def run_gui(exp_config):
    # df, _ = get_preprocessed_dataset("mixed", exp_config["features"]["is_subset"], mode="test")
    inputs = [gr.Dataframe(row_count=(1, "dynamic"), col_count=(26, "fixed"), label="Input Data", interactive=True)]

    outputs = [
        gr.Dataframe(row_count=(1, "dynamic"), col_count=(2, "fixed"), label="Predictions", headers=["Logistic Regression Churn Probability", "DT Churn Probability"])]

    # LR_clfs = load_clfs(exp_config["name"], "LogisticRegression", exp_config["cross_validation"]["n_splits"])
    # DT_clfs = load_clfs(exp_config["name"], "DecisionTreeClassifier", exp_config["cross_validation"]["n_splits"])

    # gr.Interface(fn=partial(run_inference, exp_config), inputs=inputs, outputs=outputs, examples=[[df.head(50)]], examples_per_page=10).launch()