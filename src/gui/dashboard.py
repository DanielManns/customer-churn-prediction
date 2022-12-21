import gradio as gr
import pandas as pd
import joblib

from src.models.preprocessing import get_preprocessed_dataset

inputs = [gr.Dataframe(row_count = (2, "dynamic"), col_count=(4,"dynamic"), label="Input Data", interactive=1)]

outputs = [gr.Dataframe(row_count = (2, "dynamic"), col_count=(1, "fixed"), label="Predictions", headers=["Failures"])]

model = joblib.load("model.pkl")

# we will give our dataframe as example
X, y = get_preprocessed_dataset("mixed", False)

def infer(input_dataframe):
  return pd.DataFrame(model.predict(input_dataframe))

gr.Interface(fn = infer, inputs = inputs, outputs = outputs, examples = [[df.head(2)]]).launch()






def greet(name, is_morning, temperature):
    salutation = "Good morning" if is_morning else "Good evening"
    greeting = f"{salutation} {name}. It is {temperature} degrees today"
    celsius = (temperature - 32) * 5 / 9
    return greeting, round(celsius, 2)


if __name__ == "__main__":
    demo = gr.Interface(
        fn=greet,
        inputs=["text", "checkbox", gr.Slider(0, 100)],
        outputs=["text", "number"],
    )
    demo.launch()