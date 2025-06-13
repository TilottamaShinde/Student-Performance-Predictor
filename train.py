import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from joblib import dump
import os
def load_data(filepath):
    df = pd.read_csv(filepath)
    return df

def train_model(df):
    # features and target
    X = df[['Hours Studied', 'Attendance', 'Assignments Completed']]
    y = df['Final Grade']

    # Train test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)

    #Model Training
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Evaluate Model
    y_pred = model.predict(X_test)
    print("Mean Squared Error : ", mean_squared_error(y_test, y_pred))
    print("R2 score : ", r2_score(y_test, y_pred))

    return model

def save_model(model, model_path = 'student_model.joblib'):
    directory = os.path.dirname(model_path)
    if directory:
        os.makedirs(directory, exist_ok = True)
    dump(model, model_path)
    print(f"Model saved to {model_path}")

def main():
    df = load_data('student_data.csv')
    model = train_model(df)
    save_model(model)

if __name__ == '__main__':
    main()