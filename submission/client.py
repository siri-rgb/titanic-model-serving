import requests
BASE = "http://127.0.0.1:8000"

print(requests.get(f"{BASE}/health").json())

valid = {"Pclass": 3, "Sex": "male", "Age": 22, "SibSp": 1, "Parch": 0, "Fare": 7.25, "Embarked": "S"}
r = requests.post(f"{BASE}/predict", json=valid)
print("Valid:", r.status_code, r.json())

invalid = {"Pclass": 9, "Sex": "alien", "SibSp": -1, "Fare": "abc"}
r = requests.post(f"{BASE}/predict", json=invalid)
print("Invalid:", r.status_code)
