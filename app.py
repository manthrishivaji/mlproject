import sys
import pandas as pd
# from src.utils import load_object
import os
# from sklearn.preprocessing import StandardScaler


import pickle
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
from typing import List
from fastapi.middleware.cors import CORSMiddleware


# Initialize FastAPI App
app = FastAPI(title="Regression Model API", description="Serve predictions for regression task", version="1.0")

# Define the file paths for model and preprocessor
MODEL_PATH = "artifacts/model.pkl"
PREPROCESSOR_PATH = "artifacts/preprocessor.pkl"


# Add CORS middleware to allow cross-origin requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows requests from any origin
    allow_credentials=True,
    allow_methods=["*"],  # Allows all HTTP methods
    allow_headers=["*"],  # Allows all headers
)


# Load the trained model and preprocessor
try:
    with open(MODEL_PATH, "rb") as model_file:
        model = pickle.load(model_file)
    with open(PREPROCESSOR_PATH, "rb") as preprocessor_file:
        preprocessor = pickle.load(preprocessor_file)
    print("Model and preprocessor loaded successfully.")
except Exception as e:
    raise Exception(f"Error loading model or preprocessor: {e}")

# Define input schema using Pydantic
class InputData(BaseModel):
    writing_score: float
    reading_score: float
    gender: str
    race_ethnicity: str
    parental_level_of_education: str
    lunch: str
    test_preparation_course: str

class MultipleInputs(BaseModel):
    inputs: List[InputData]

# Define root endpoint
@app.get("/")
def read_root():
    return {"message": "Welcome to the Regression Model API"}

# Define prediction endpoint for a single input
@app.post("/predict/")
def predict(data: InputData):
    try:
        # Convert input data to DataFrame
        input_dict = data.dict()
        input_df = pd.DataFrame([input_dict])

        # Preprocess the input
        processed_input = preprocessor.transform(input_df)

        # Make prediction
        prediction = model.predict(processed_input)

        return {"prediction": prediction[0]}

    except Exception as e:
        return {"error": str(e)}

# Define prediction endpoint for multiple inputs
@app.post("/predict/batch/")
def predict_batch(data: MultipleInputs):
    try:
        # Convert list of inputs to DataFrame
        input_data = [item.dict() for item in data.inputs]
        input_df = pd.DataFrame(input_data)

        # Preprocess the input
        processed_input = preprocessor.transform(input_df)

        # Make predictions
        predictions = model.predict(processed_input)

        return {"predictions": predictions.tolist()}

    except Exception as e:
        return {"error": str(e)}

# # Run the app
# if __name__ == "__main__":
#     uvicorn.run(app, host="0.0.0.0", port=8080)

#uvicorn app:app --reload