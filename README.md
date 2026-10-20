# Marketing Sales Predictor

An end-to-end machine learning and MLOps portfolio project predicting sales volume from TV advertising expenditure using linear regression. It includes model development from scratch, evaluation, an independently deployed FastAPI inference backend, a Streamlit frontend, automated tests, and Dockerized deployments.

## Live Demo

- **Streamlit application:** https://marketing-sales-predictor.onrender.com
- **FastAPI Swagger documentation:** https://marketing-sales-api.onrender.com/docs
- **API health check:** https://marketing-sales-api.onrender.com/health

Enter a TV advertising budget (in thousands of US dollars) and click **Predict Sales**. For 100 ($100,000), the model predicts approximately **11.77 thousand units**. Free-hosted services may take some time to wake after inactivity.

## Architecture

```text
Browser
  |
  v
Streamlit frontend (Render, Dockerfile)
  | HTTPS POST /predict  {"tv_budget": 100}
  v
FastAPI backend (Render, Dockerfile.api)
  | Pydantic validation; Pandas input frame
  v
Saved Scikit-learn LinearRegression model (models/marketing_model.joblib)
  |
  v
JSON response {"predicted_sales": 11.7726, "unit": "thousand_units"}
```

Streamlit handles input and presentation only. FastAPI validates and serves predictions; the model is **not loaded in the frontend**. The frontend's `API_URL` environment variable points to the backend base URL.

## Dataset and Model

The historical 200-row Advertising dataset associated with *An Introduction to Statistical Learning* has TV, Radio, Newspaper and Sales variables. This project uses only TV as a feature. An 80/20 train/test split with `random_state=42` was used.

| Model | Test MAE (thousand units) |
| --- | ---: |
| Mean-sales baseline | 4.9625 |
| Custom gradient descent | 2.44442 |
| Scikit-learn LinearRegression | 2.44442 |

The custom and scikit-learn models reduce MAE by about **50.74%** compared with the mean baseline. The fitted line is `Sales ≈ 7.1196 + 0.04653 × TV`, with TV measured in thousands of dollars and Sales in thousands of units.

## Tech Stack

Python, NumPy, Pandas, Scikit-learn, Joblib, FastAPI, Pydantic, Streamlit, Requests, Pytest, Docker, GitHub Actions, and Render.

## Run Locally

Use Python 3.12 for compatibility with the deployed model dependencies. From the project root:

```bash
python -m pip install -r requirements.txt
python -m pip install pytest
uvicorn api.main:app --host 127.0.0.1 --port 8000
```

In another terminal:

```bash
# Windows PowerShell
$env:API_URL = "http://127.0.0.1:8000"
streamlit run app.py
```

For the lean frontend dependencies alone: `python -m pip install -r requirements-frontend.txt`. The backend requires `requirements.txt` as used by `Dockerfile.api`.

Or build and run independent Docker containers:

```bash
docker build -f Dockerfile.api -t marketing-sales-api .
docker build -f Dockerfile -t marketing-sales-frontend .
docker run --rm -p 127.0.0.1:8000:8000 marketing-sales-api
```

Then in another terminal:

```bash
# Windows PowerShell
docker run --rm -p 127.0.0.1:8501:8501 -e API_URL=http://host.docker.internal:8000 marketing-sales-frontend
```

Browse to `http://127.0.0.1:8501`. On Docker Desktop, `host.docker.internal` allows the frontend container to contact the API published on the host.

## API Contract

`GET /health` returns service status.

`POST /predict` accepts JSON:

```json
{"tv_budget": 100}
```

Returns:

```json
{"predicted_sales": 11.7726, "unit": "thousand_units"}
```

Pydantic validates numeric budgets in the range 0–300. Invalid inputs return HTTP 422.

## Testing and CI/CD

```bash
python -m pytest -v
```

Nine tests cover the API's health, valid and invalid inputs, regression predictions, and MAE edge cases. GitHub Actions runs Pytest and builds **both** Docker images on pushes and pull requests. Each Render service uses a separate Dockerfile from this repository.

## Limitations

- Educational, historical benchmark, not a validated commercial forecasting model.
- TV budget is the sole feature; the model does not account for other drivers of sales.
- The estimated relationship is **not causal evidence** of advertising effectiveness or return on investment.
- Out-of-distribution predictions may be unreliable; the API's 0–300 validation range does not guarantee every value is within the training distribution.
- The public API has no authentication, rate limiting, or production-grade monitoring.

## Potential Improvements

Additional predictor variables, model version tracking, CI integration tests, readiness checks, API monitoring and alerting, and retraining workflows.

## Author

[AbubakarMA](https://github.com/AbubakarMA)
