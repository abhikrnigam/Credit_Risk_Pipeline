import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
import joblib  # Changed from pickle to joblib

class Model:

    def __init__(self):
        pass

    def predict(self, data: pd.DataFrame):
        X = data.drop(columns=['Target'])
        y = data['Target']

        print(f"Data Received to the model")
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        classification_model = XGBClassifier(n_estimators=1000, max_depth=5)
        classification_model.fit(X_train, y_train)

        y_pred = classification_model.predict(X_test)
        accuracy = accuracy_score(y_pred, y_test)

        print("The Accuracy of the model is:", accuracy * 100)

        return classification_model

    def save_model(self, model):
        joblib.dump(model, 'model.joblib')  # Changed from pickle to joblib
        print(f"Model Saved with joblib")
        

    def load_model(self, model_path):
        model = joblib.load(model_path)  # Added model loading method using joblib
        print(f"Model Loaded from {model_path}")
        return model
