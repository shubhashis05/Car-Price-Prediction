# Car Price Prediction using Machine Learning

This project predicts the selling price of a used car from its specifications and ownership details. It uses the **Car Details v3** dataset and includes both a machine learning workflow and a Flask web app for making predictions.

## Project Goals

- Clean and explore the car data through exploratory data analysis (EDA)
- Prepare features, including encoding categorical values and converting vehicle specifications to numeric values
- Train and compare regression models, including Linear Regression, Random Forest, and XGBoost
- Evaluate model performance using MAE, RMSE, and R²
- Use a saved Random Forest model to estimate a car’s selling price from a web form

## Features Used

The model uses the following car details:

- Car name and year
- Kilometres driven
- Fuel type and seller type
- Transmission and owner history
- Mileage, engine capacity, and maximum power
- Number of seats

The target variable is `selling_price`. Torque is excluded from the final model because its values use inconsistent formats in the dataset.

## Web App

The Flask app accepts car details through a web form, prepares the input in the format expected by the trained model, and displays the estimated selling price.

The saved model is located at `model/best_model.pkl`. The app also uses the dataset at `notebook/Car details v3.csv` to build the categorical value mappings.

## Run Locally

Install the project dependencies:

```bash
pip install -r requirements.txt
