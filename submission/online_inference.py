import pandas as pd, joblib
model = joblib.load("model.pkl")
FEATURES = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]
record = pd.read_csv("unseen_data.csv").iloc[[0]]
pred = model.predict(record[FEATURES])[0]
print("Passenger record:\n", record[FEATURES].to_string(index=False))
print("Prediction:", "Survived" if pred == 1 else "Did not survive")
