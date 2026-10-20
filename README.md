# Marketing Sales Predictor

An end-to-end Machine Learning and MLOps project that predicts sales volume based on TV advertising expenditure.

This project demonstrates how to develop a linear regression model from scratch, evaluate it against a baseline and Scikit-learn, serve predictions through Streamlit, containerize the application with Docker, and automate testing and builds using GitHub Actions.

## Project Overview

**Problem:** Estimate sales volume from TV advertising expenditure using historical advertising data.

**Solution:** Build and evaluate a simple linear regression model, then deploy a prediction interface that loads a pretrained model.

**Technology:** Python, NumPy, Pandas, Scikit-learn, Streamlit, Matplotlib, Pytest, Docker, Git and GitHub Actions.

## Dataset

The project uses the classic Advertising dataset associated with _An Introduction to Statistical Learning_.

The dataset contains 200 observations and four variables:

- TV advertising expenditure
- Radio advertising expenditure
- Newspaper advertising expenditure
- Sales

The current model uses **TV advertising expenditure** as its only predictor.

The dataset is intended for statistical learning and demonstration. Predictions should not be interpreted as validated commercial sales forecasts or causal estimates of advertising effectiveness.

## Model Development

Two linear regression implementations were evaluated:

1. A custom NumPy implementation trained using gradient descent.
2. Scikit-learn's `LinearRegression`.

Data was divided into 80% training and 20% testing, with `random_state=42`.

A simple mean-sales baseline was used as a reference.

## Model Performance

| Model                         | Test MAE |
| ----------------------------- | -------: |
| Mean baseline                 |   4.9625 |
| Custom gradient descent       |  2.44442 |
| Scikit-learn LinearRegression |  2.44442 |

The regression models achieved approximately **50.74% lower test MAE** than the mean baseline.

The fitted model is:

`Predicted Sales = 7.1196 + 0.04653 × TV Advertising Expenditure`

Advertising expenditure is measured in thousands of dollars, and sales are expressed in thousands of units.

## Application

The Streamlit application allows users to enter a TV advertising budget and receive a sales prediction from the saved Scikit-learn model.

Example:

- TV advertising budget: $100,000
- Predicted sales: approximately 11.77 thousand units

The model is loaded from a serialized artifact instead of being retrained for each prediction.

## Engineering and MLOps

- From-scratch gradient descent implementation
- Feature standardization during custom model training
- Model benchmarking
- Joblib model persistence
- Automated unit tests with Pytest
- Regression tests for NumPy broadcasting errors
- Docker containerization
- GitHub Actions continuous integration

## Run Locally

```bash
git clone https://github.com/AbubakarMA/marketing-sales-predictor.git
cd marketing-sales-predictor
python -m pip install -r requirements.txt
streamlit run app.py
```

Alternatively, run with Docker:

```bash
docker build -t marketing-sales-predictor .
docker run --rm -p 127.0.0.1:8501:8501 marketing-sales-predictor
```

Then visit `http://localhost:8501`.

Use a supported Python environment with dependencies compatible with the saved model artifact.

## Testing

Run the unit test suite:

```bash
python -m pytest -v
```

The initial four unit tests verify predictions, invalid training input, broadcasting-safe MAE, and unequal-length validation.

GitHub Actions automatically runs tests and builds the Docker image.

## Live Demo

Deployment is being finalized. The live application URL will be added after successful validation.

## Limitations

- The model uses only TV advertising expenditure.
- The dataset is a historical educational benchmark.
- Predictions outside the training distribution may be unreliable.
- Predictions should not be used as evidence of advertising causation or guaranteed business performance.

## Future Improvements

- Multiple regression features
- Model versioning and reproducible training pipeline
- Input distribution warnings
- Additional integration tests
- Model monitoring and retraining workflows


## Live Demo

Try the deployed application:

https://marketing-sales-predictor.onrender.com

Enter a TV advertising budget to obtain a predicted sales volume.

The application is deployed on Render using Docker, with automated CI checks through GitHub Actions.

## Author

**AbubakarMA**

GitHub: https://github.com/AbubakarMA
