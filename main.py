import pandas as pd
import preprocessing
import model
import uvicorn
from fastapi import FastAPI

#object
app = FastAPI()

#on http://127.0.0.1:8000
@app.get('/')
def index():
    return {'message': 'Hello,Users'}


#on http://127.0.0.1:8000/xyz....
@app.get('/Welcome')
def get_name(name: str)
    return{'Check your credit risk': f'{name}'}


#run
if __name__ == '__main__':
    preprocessing_object = preprocessing.Preprocessing()
    data = preprocessing_object.DataPreprocessing()
    model_object = model.Model()
    classification_model = model_object.predict(data)
    model_object.save_model(classification_model)
    print(model_object)
    uvicorn.run(app, host = '127.0.0.1', port=8000)


# uvicorn main:app --reload 