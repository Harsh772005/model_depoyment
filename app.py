# import pickle
# import numpy as np
# import gradio as gr

# # Load the trained model and scaler
# with open("model_linear_regression.pkl", "rb") as model_file:
#     model = pickle.load(model_file)

# with open("model_scaler_x.pkl", "rb") as scaler_file:
#     scaler = pickle.load(scaler_file)

# # Prediction function
# def predict_tip(total_bill, sex, smoker, time, size, day):
#     """
#     Predict tip amount based on input features.
#     """
#     # Map inputs
#     sex = 1 if sex.lower() == "male" else 0
#     smoker = 1 if smoker.lower() == "yes" else 0
#     time = 1 if time.lower() == "dinner" else 0
    
#     # Day one-hot encoding
#     days = ["Fri", "Sat", "Sun", "Thur"]
#     day_one_hot = [1.0 if d == day else 0.0 for d in days]
    
#     # Scale the total_bill
#     total_bill_scaled = scaler.transform([[total_bill]])[0][0]
    
#     # Create the feature array
#     features = [total_bill_scaled, sex, smoker, time, size] + day_one_hot
#     features = np.array(features).reshape(1, -1)
    
#     # Predict using the model
#     tip = model.predict(features)[0]
#     return f"Predicted Tip Amount: ${tip:.2f}"

# # Gradio Interface
# with gr.Blocks() as demo:
#     gr.Markdown("## Tips Prediction App")
#     gr.Markdown("Enter the following details to predict the tip amount:")
    
#     with gr.Row():
#         total_bill_input = gr.Number(label="Total Bill ($)")
#         sex_input = gr.Radio(choices=["Male", "Female"], label="Sex")
#         smoker_input = gr.Radio(choices=["Yes", "No"], label="Smoker")
#         time_input = gr.Radio(choices=["Lunch", "Dinner"], label="Time")
#         size_input = gr.Number(label="Table Size", value=2, precision=0)
#         day_input = gr.Radio(choices=["Fri", "Sat", "Sun", "Thur"], label="Day")
    
#     output = gr.Textbox(label="Prediction")
    
#     predict_button = gr.Button("Predict Tip")
#     predict_button.click(
#         predict_tip,
#         inputs=[total_bill_input, sex_input, smoker_input, time_input, size_input, day_input],
#         outputs=output
#     )

# demo.launch()





import pickle
import numpy as np
import gradio as gr

# Load the trained model and scaler
with open("model_linear_regression.pkl", "rb") as model_file:
    model = pickle.load(model_file)

with open("model_scaler_x.pkl", "rb") as scaler_file:
    scaler = pickle.load(scaler_file)

# Prediction function
def predict_tip(total_bill, sex, smoker, time, size, day):
    """
    Predict tip amount based on input features.
    """
    # Map inputs
    sex = 1 if sex.lower() == "male" else 0
    smoker = 1 if smoker.lower() == "yes" else 0
    time = 1 if time.lower() == "dinner" else 0
    
    # Day one-hot encoding
    days = ["Fri", "Sat", "Sun", "Thur"]
    day_one_hot = [1.0 if d == day else 0.0 for d in days]
    
    # Create the full feature array
    features = [total_bill, sex, smoker, time, size] + day_one_hot
    features = np.array(features).reshape(1, -1)
    
    # Scale all features
    scaled_features = scaler.transform(features)
    
    # Predict using the model
    tip = model.predict(scaled_features)[0]
    return f"Predicted Tip Amount: ${tip:.2f}"

# Gradio Interface
with gr.Blocks() as demo:
    gr.Markdown("## Tips Prediction App")
    gr.Markdown("Enter the following details to predict the tip amount:")
    
    with gr.Row():
        total_bill_input = gr.Number(label="Total Bill ($)")
        sex_input = gr.Radio(choices=["Male", "Female"], label="Sex")
        smoker_input = gr.Radio(choices=["Yes", "No"], label="Smoker")
        time_input = gr.Radio(choices=["Lunch", "Dinner"], label="Time")
        size_input = gr.Number(label="Table Size", value=2, precision=0)
        day_input = gr.Radio(choices=["Fri", "Sat", "Sun", "Thur"], label="Day")
    
    output = gr.Textbox(label="Prediction")
    
    predict_button = gr.Button("Predict Tip")
    predict_button.click(
        predict_tip,
        inputs=[total_bill_input, sex_input, smoker_input, time_input, size_input, day_input],
        outputs=output
    )

demo.launch(share=True)
