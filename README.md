# Customer Churn Prediction

## Description
This project is based on the "Customer Churn Prediction 2020" Kaggle challenge and aims to solve the challange but also build an deployable MVP. This MVP includes a separated frontend and backend which are implemented following the microservice architecture and can be run with docker compose.

## Installation
1. Install docker
2. Install docker compose

## Usage
1. Check out the exploratory data analysis (eda.ipynb)
2. You can start the MVP with docker compose. The default ports for the frontend is 7860 and for the backend 8000.

    ```docker compose up```

## Example
### Predict Customer Churn
![Predict Churn](images/predict_tab.png "Predict Customer Churn")

### Understand Customer Churn
![Understand Churn](images/feature_importance_tab.png "Understand what made customers churn")

### Visualize Machine Learning Model
![Visualize ML model](images/visualization_tab.png "Visualize ML model")