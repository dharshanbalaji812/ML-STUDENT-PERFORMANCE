
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import pandas as pd

app = FastAPI()

class StudentData(BaseModel):
    study_hours: float
    attendance: float
    previous_marks: float

model = joblib.load("models/student_model.pkl")

@app.get("/")
def home():
    return {
        "message":"Welcome to AI Student Performance Prediction System! Use the /predict endpoint to get predictions based on study hours, attendance, and previous marks."
    }

@app.post("/predict")
def predict(student: StudentData):
    try:
        input_data = pd.DataFrame([{
            "study_hours": student.study_hours,
            "attendance": student.attendance,
            "previous_marks": student.previous_marks
        }])

        prediction = model.predict(input_data)

        return {
            "prediction_result": str(prediction[0])
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(e)}"
        )
