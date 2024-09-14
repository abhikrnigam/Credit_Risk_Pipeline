# -*- coding: utf-8 -*-
"""
Created on Tue Nov 17 21:40:41 2020

@author: win10
"""

# 1. Library imports
import uvicorn
from fastapi import FastAPI
from CreditRisk import CreditRiskData
import numpy as np
import pickle
import pandas as pd

# 2. Create the app object
app = FastAPI()
pickle_in = open("model.pkl","rb")
classifier=pickle.load(pickle_in)

# 3. Index route, opens automatically on http://127.0.0.1:8000
@app.get('/')
def index():
    return {'message': 'Hello, User'}

# 4. Route with a single parameter, returns the parameter within a message
#    Located at: http://127.0.0.1:8000/AnyNameHere
@app.get('/{name}')
def get_name(name: str):
    return {'Welcome To Credit Risk': f'{name}'}

# 3. Expose the prediction functionality, make a prediction from the passed
#    JSON data and return the predicted Bank Note with the confidence
@app.post('/predict')
def predict_creditrisk(data:CreditRiskData):
    data = data.dict()
    age=data['age']
    income=data['income']
    home=data['home']
    emp_length=data['emp_length']
    intent=data['intent']
    amount=data['amount']
    rate=data['rate']
    status=data['status']
    percent_income=data['percent_income']
    cred_length=data['cred_length']
    prediction = classifier.predict([[age,income,home,emp_length,intent,amount,rate,status,percent_income,cred_length]])
    if(prediction[0]>0.5):
        prediction = "Risky"
    else:
        prediction = "Not Risky"
    return {
        'prediction': prediction
    }

# 5. Run the API with uvicorn
#    Will run on http://127.0.0.1:8000
if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=8000)
    
#uvicorn app:app --reload