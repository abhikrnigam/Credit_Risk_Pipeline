import pandas as pd
import preprocessing
import model
from fastapi import FastAPI
import joblib



#run
if __name__ == '__main__':
    preprocessing_object = preprocessing.Preprocessing()
    data = preprocessing_object.DataPreprocessing()
    model_object = model.Model()
    classification_model = model_object.predict(data)
    model_object.save_model(classification_model)
    print(model_object)
    #uvicorn.run(app, host = '127.0.0.1', port=8000)


# uvicorn main:app --reload 