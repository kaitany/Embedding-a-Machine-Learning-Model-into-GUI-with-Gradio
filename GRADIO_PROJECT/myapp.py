import gradio as gr
import pandas as pd
import pickle
import os
from sklearn.ensemble import GradientBoostingClassifier


#KEY LISTS
expected_inputs = ['SeniorCitizen', 'Partner', 'Dependents', 'PhoneService','MultipleLines',
                      'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV',
                        'StreamingMovies', 'Contract', 'PaperlessBilling', 'PaymentMethod', 'tenure_category', 'MonthlyCharges', 'TotalCharges']

numerical_cols = ['MonthlyCharges', 'TotalCharges']

categorical_cols = ['SeniorCitizen', 'Partner', 'Dependents', 'PhoneService', 'MultipleLines', 
                        'InternetService', 'OnlineSecurity', 'OnlineBackup',  'DeviceProtection', 'TechSupport',
                          'StreamingTV', 'StreamingMovies', 'Contract', 'PaperlessBilling', 'PaymentMethod', 'tenure_category']

#Define helper functions
#setup
#variables and constants
DIRPATH = os.path.dirname(os.path.realpath(__file__))
ml_core_fp = os.path.join(DIRPATH, "Gradio_toolkit.pkl")
#Function to load dataset

# Load your image file
image_path = os.path.join(DIRPATH, r"Datasets\image\Understanding Customer Churn.png")  
# Create an Image component
app_image = gr.Image(image_path)

#useful functions
def  load_ml_components(fp):
    "load the ml components to re-use in app"
    with open(fp, 'rb') as file:
        obj = pickle.load(file)
        return obj
    
#Execute and instantiate the toolkit components
ml_components_dict = load_ml_components(fp = ml_core_fp)

preprocessor = ml_components_dict["pipeline"]

label_encoder =  ml_components_dict["label_encoder"]

#Import the model
model = GradientBoostingClassifier()

model = ml_components_dict["model"]

pipeline = ml_components_dict["prediction_pipeline"]

#Function to process inputs and return prediction
def return_prediction(app_image, *args, pipeline = pipeline, preprocessor = preprocessor, model = model, label_encoder = label_encoder):

# convert inputs into a dataframe
    #Change the matrix to dataframe
    input_data = pd.DataFrame([args], columns=expected_inputs)

    # Make the prediction using the pipeline containing numerical and categorical transformers
    model_output = pipeline.predict(input_data)

    if model_output == "Yes":
        prediction = 1
    else:
        prediction = 0

    # Return the prediction
    return {
        "Prediction: Customer is likely to LEAVE": prediction,
        "Prediction: Customer is likely to STAY": 1 - prediction
    }
  

#Set app interface
##inputs
SeniorCitizen = gr.Radio(label = "Is the customer a senior citizen?", choices = ["False","True"], value = "False")
Partner = gr.Radio(label = "Does the customer have a partner?", choices =["No", "Yes"], value = "No")
Dependents = gr.Radio(label = "Does the customer have dependents?", choices =["Yes","No"], value = "No")
PhoneService = gr.Radio(label = "Does the customer has phone service?", choices =["Yes","No"], value = "No")
MultipleLines = gr.Radio(label = "Does the customer possesses MultipleLines?", choices =["Yes","No"], value = "No")
InternetService  = gr.Dropdown(label = "What is the customer's internet service provider?", choices =['No Internet Service', 'DSL', 'Fiber optic'], value = "No Internet Service")
OnlineSecurity = gr.Dropdown(label = "Does the customer has online security?", choices =['No internet service', 'No', 'Yes'], value = "No")
OnlineBackup = gr.Dropdown(label = "Does the customer has online backup?", choices =['No Internet', 'Yes', 'No'], value = "No")
DeviceProtection = gr.Dropdown(label = "Does the customer has device protection?", choices =['No internet service', 'No', 'Yes'], value = "No")
TechSupport = gr.Dropdown(label = "Does the customer have tech support?", choices =['No Internet', 'No', 'Yes'], value = "No") 
StreamingTV = gr.Dropdown(label = "Does the customer stream TV?", choices =['No internet service', 'No', 'Yes'], value = "No")  
StreamingMovies = gr.Dropdown(label = "Does the customer stream movies?", choices =['No internet service', 'No', 'Yes'], value = "No")
Contract = gr.Dropdown(label = "What is the contract term of the customer?", choices = ['One year', 'Two year', 'Month-to-month'], value = "Month-to-month")
PaperlessBilling = gr.Radio(label = "Does the customer has paperless billing?", choices =["Yes","No"], value = "No")
PaymentMethod = gr.Radio(label = "What is the customer's payment method?", choices = ['Bank transfer (automatic)', 'Mailed check', 'Electronic check',
       'Credit card (automatic)'], value= "Bank transfer (automatic)")
MonthlyCharges = gr.Slider(label= "What is the monthly amount charged to the customer?", minimum= 15, maximum= 150, value= 20, interactive= True)
TotalCharges = gr.Slider(label="What is the total amount charged to the customer?", minimum=15, maximum=8800, value=150, interactive=True)
tenure_category = gr.Dropdown(label = "How many months has the customer stayed with the company?", choices = ['0-6 months', '7-12 months', '13-24 months', '25-36 months',
                                                                                          '37-48 months', '49-60 months', '61+ months'], value= "0-6 months")


outputs = gr.Label("Awaiting Submission...")
gr.Interface(inputs=[app_image, SeniorCitizen, Partner, Dependents, PhoneService, MultipleLines,
                      InternetService, OnlineSecurity, OnlineBackup, DeviceProtection, TechSupport, StreamingTV,
                        StreamingMovies, Contract, PaperlessBilling, PaymentMethod, tenure_category, MonthlyCharges, TotalCharges],
                          fn=return_prediction,
                          outputs=outputs,
                          title="Telecom Churn Prediction App",
                          description="<b>Customer Attrition Prediction App. Input Customer data to get predictions. </b>", live=True).launch(inbrowser=True, show_error=True, share=True)