import pandas as pd, joblib, time
model = joblib.load("model.pkl")
FEATURES = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]
stream = pd.read_csv("unseen_data.csv").iloc[21:31]        # 10 records
for i, (_, row) in enumerate(stream.iterrows(), 1):
    pred = model.predict(row[FEATURES].to_frame().T)[0]
    print(f"Record {i}: {row[FEATURES].to_dict()} -> {'Survived' if pred == 1 else 'Did not survive'}")
    time.sleep(1)
