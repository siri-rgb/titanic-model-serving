import pandas as pd, joblib
model = joblib.load("model.pkl")
FEATURES = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]
batch = pd.read_csv("unseen_data.csv").iloc[1:21].copy()   # 20 records
batch["Prediction"] = model.predict(batch[FEATURES])
batch.to_csv("batch_predictions.csv", index=False)
print(batch[FEATURES + ["Prediction"]])
