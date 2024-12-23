from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, classification_report
import io

app = FastAPI()

iris_data = load_iris()
df = pd.DataFrame(iris_data.data, columns=iris_data.feature_names)
df['species'] = iris_data.target_names[iris_data.target]

scaler = StandardScaler()
df_scaled = df.drop('species', axis=1)
df_scaled = pd.DataFrame(scaler.fit_transform(df_scaled), columns=df.columns[:-1])
df_scaled['species'] = df['species']

X = df_scaled.drop('species', axis=1)
y = df_scaled['species']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

model = DecisionTreeClassifier(random_state=0)
model.fit(X_train, y_train)



class IrisData(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float

@app.get("/")
def read_root():
    return {"message": "predictin da iris"}

@app.post("/predict/")
def predict_species(data: IrisData):
    input_data = np.array([[data.sepal_length, data.sepal_width, data.petal_length, data.petal_width]])
    scaled_input = scaler.transform(input_data)
    prediction = model.predict(scaled_input)
    species = prediction[0]
    
    return {"predicted_species": species}

@app.get("/model_accuracy/")
def model_accuracy():
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    return {"accuracy": accuracy}

@app.get("/classification_report/")
def get_classification_report():
    y_pred = model.predict(X_test)
    report = classification_report(y_test, y_pred, output_dict=True)
    return report


