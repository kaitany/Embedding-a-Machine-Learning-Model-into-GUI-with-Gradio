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
def return_prediction(*args, pipeline = pipeline, preprocessor = preprocessor, model = model, label_encoder = label_encoder):

# convert inputs into a dataframe
    #Change the matrix to dataframe
    input_data = pd.DataFrame([args], columns=expected_inputs)

    # Make the prediction using the pipeline containing numerical and categorical transformers
    model_output = pipeline.predict(input_data)


        # Return the prediction
    if model_output == "Yes":
           
           prediction = 1
    else:
          prediction = 0

    

       #return the prediction
    return {"prediction: Customer is likelt to LEAVE": prediction,
            "Prediction: Customer is likely to stay":  1 - prediction}
      
#Set app interface
##inputs
SeniorCitizen = gr.Radio(label = "Is the customer a senior citizen?", choices = ["False","True"], value = "False")
Partner = gr.Radio(label = "Does the customer have a partner?", choices =["No", "Yes"], value = "No")
Dependents = gr.Radio(label = "Are there dependants on the customer?", choices =["Yes","No"], value = "No")
PhoneService = gr.Radio(label = "Does the customer has access to phone service?", choices =["Yes","No"], value = "No")
MultipleLines = gr.Radio(label = "Does the customer possesses MultipleLines?", choices =["Yes","No"], value = "No")
InternetService  = gr.Dropdown(label = "What Internet Service does the customer use?", choices =['No Internet Service', 'DSL', 'Fiber optic'], value = "No Internet Service")
OnlineSecurity = gr.Dropdown(label = "Has the customer access to online security?", choices =['No internet service', 'No', 'Yes'], value = "No")
OnlineBackup = gr.Dropdown(label = "Has the employee had access to Online Backup?", choices =['No Internet', 'Yes', 'No'], value = "No")
DeviceProtection = gr.Dropdown(label = "Has the customer been given device protection?", choices =['No internet service', 'No', 'Yes'], value = "No")
TechSupport = gr.Dropdown(label = "Does the customer has access to technical support?", choices =['No Internet', 'No', 'Yes'], value = "No") 
StreamingTV = gr.Dropdown(label = "Does the customer stream television programmes?", choices =['No internet service', 'No', 'Yes'], value = "No")  
StreamingMovies = gr.Dropdown(label = "Does the customer stream movies?", choices =['No internet service', 'No', 'Yes'], value = "No")
Contract = gr.Dropdown(label = "What are the customers' terms of contract?", choices = ['One year', 'Two year', 'Month-to-month'], value = "Month-to-month")
PaperlessBilling = gr.Radio(label = "Has the employee subscribed to paperless billing?", choices =["Yes","No"], value = "No")
PaymentMethod = gr.Radio(label = "What is the preffered payment method by the customer?", choices = ['Bank transfer (automatic)', 'Mailed check', 'Electronic check',
       'Credit card (automatic)'], value= "Bank transfer (automatic)")
MonthlyCharges = gr.Number(label = "What is the monthly charges paid by the customer on average?", minimum = 0, maximum = 118.65, value = 18.25)
TotalCharges = gr.Number(label = "What is the total charges paid by the customer on average?", minimum = 0, maximum = 8680, value = 18.8)
tenure_category = gr.Dropdown(label = "What is the tenure of the customer?", choices = ['0-6 months', '7-12 months', '13-24 months', '25-36 months',
                                                                                          '37-48 months', '49-60 months', '61+ months'], value= "0-6 months")

# #Outputs[MonthlyCharges, TotalCharges, SeniorCitizen, Partner, Dependents, PhoneService, MultipleLines, InternetService, 
#                      OnlineSecurity, OnlineBackup, DeviceProtection, TechSupport, StreamingTV, StreamingMovies, Contract,
#                           PaperlessBilling, PaymentMethod, tenure_category]

gr.Interface(inputs=[SeniorCitizen, Partner, Dependents, PhoneService, MultipleLines,
                      InternetService, OnlineSecurity, OnlineBackup, DeviceProtection, TechSupport, StreamingTV,
                        StreamingMovies, Contract, PaperlessBilling, PaymentMethod, tenure_category, MonthlyCharges, TotalCharges],
                          fn=return_prediction,
                          outputs=gr.Label("Awaiting Submission..."),
                          title="Customer Churn Prediction App",
                          description="This App was created by team Santorini during our LP4 EDS", live=True).launch(inbrowser=True, show_error=True)