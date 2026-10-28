# ML Model Deployment and Serving using FastAPI

## Description
Titanic survival prediction model with online, batch and streaming
inference, served through a FastAPI REST API.

## Dataset
Titanic - Machine Learning from Disaster (Kaggle)
https://www.kaggle.com/competitions/titanic/data

## Execution Steps
1. pip install -r requirements.txt
2. python train.py
3. python online_inference.py
4. python batch_inference.py
5. python streaming_inference.py
6. uvicorn app:app --port 8000
7. python client.py

## Results
- Model: Random Forest, accuracy: (0.8156424581005587)
- Batch predictions saved in batch_predictions.csv
- API endpoints: GET /health, POST /predict
- Screenshots are in the screenshots folder
- Observations report: report.pdf
