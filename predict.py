import sys
from joblib import load
import numpy as np
import os

from train import load_data


def predict_performance(hours_studied, attendance_rate, previous_grade):
    model_path = "student_model.joblib"

    if not os.path.exists(model_path):
        print("Model not found. Please train the model first using train.py")
        return

    # Load the trained model
    model = load(model_path)

    # Prepared the input in expected format
    input_data = np.array([[hours_studied, attendance_rate, previous_grade]])

    # Make prediction
    predicted_score = model.predict(input_data)[0]
    print(f"Predicted Student Performance : {predicted_score:.2f}")

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: Python predict.py <house_studied> <attendace_rate><previous_grade>")
        print("Example: Python predict.py 5 90 75")
    else:
        hours = float(sys.argv[1])
        attendance = float(sys.argv[2])
        previous = float(sys.argv[3])
        predict_performance(hours, attendance, previous)