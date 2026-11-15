# Heart Disease Prediction API

FastAPI service serving a Random Forest classifier (Heart Disease dataset), Dockerized and deployed on Render.

## Run locally
1. Download `heart.csv` from Kaggle into `data/`.
2. `pip install -r requirements.txt && python train.py`
3. `docker-compose build && docker-compose up`
4. Open http://localhost:8000/docs

## Endpoints
- `GET /health`
- `GET /info`
- `POST /predict`

## Example
curl -X POST http://localhost:8000/predict -H "Content-Type: application/json" \
 -d '{"age":55,"sex":1,"cp":0,"trestbps":140,"chol":250,"fbs":0,"restecg":1,"thalach":150,"exang":0,"oldpeak":1.0,"slope":1,"ca":0,"thal":2}'

## Live URL
https://heart-api-31h9.onrender.com
