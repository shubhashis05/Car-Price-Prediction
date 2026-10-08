import csv
import math
import pickle
from pathlib import Path

import pandas as pd
from flask import Flask, render_template, request


BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "model" / "best_model.pkl"
DATA_PATH = BASE_DIR / "notebook" / "Car details v3.csv"
FEATURES = [
    "name",
    "year",
    "km_driven",
    "fuel",
    "seller_type",
    "transmission",
    "owner",
    "mileage",
    "engine",
    "max_power",
    "seats",
]
CATEGORICAL_FEATURES = ("name", "fuel", "seller_type", "transmission", "owner")

app = Flask(__name__, template_folder="frontend", static_folder="frontend")


def load_label_mappings():
    with DATA_PATH.open("r", newline="", encoding="utf-8-sig") as csv_file:
        rows = list(csv.DictReader(csv_file))

    mappings = {
        column: {
            value: index
            for index, value in enumerate(sorted({row[column] for row in rows}))
        }
        for column in CATEGORICAL_FEATURES
    }
    years = sorted({int(row["year"]) for row in rows})
    mappings["year"] = {year: index for index, year in enumerate(years)}
    return mappings


LABEL_MAPPINGS = load_label_mappings()
CAR_NAMES = sorted(LABEL_MAPPINGS["name"])
with MODEL_PATH.open("rb") as model_file:
    model = pickle.load(model_file)


def render_form(**context):
    return render_template("index.html", car_names=CAR_NAMES, **context)


@app.get("/")
def home():
    return render_form()


@app.post("/predict")
def predict():
    form_data = request.form
    required_fields = set(FEATURES)
    missing_fields = [
        field
        for field in required_fields
        if not form_data.get(field, "").strip()
    ]
    if missing_fields:
        return render_form(
            error="Please complete all car details before predicting.",
        )

    try:
        values = {
            "name": form_data["name"].strip(),
            "year": int(form_data["year"]),
            "km_driven": int(form_data["km_driven"]),
            "fuel": form_data["fuel"].strip(),
            "seller_type": form_data["seller_type"].strip(),
            "transmission": form_data["transmission"].strip(),
            "owner": form_data["owner"].strip(),
            "mileage": float(form_data["mileage"]),
            "engine": int(form_data["engine"]),
            "max_power": float(form_data["max_power"]),
            "seats": int(form_data["seats"]),
        }
    except ValueError:
        return render_form(
            error="Enter valid numeric values for year, distance, mileage, engine, power, and seats.",
        )

    for feature in CATEGORICAL_FEATURES:
        if values[feature] not in LABEL_MAPPINGS[feature]:
            return render_form(
                error=f"'{values[feature]}' is not a car {feature} found in the model's training data.",
            )

    if values["year"] not in LABEL_MAPPINGS["year"]:
        return render_form(
            error="The selected year is outside the years available in the model's training data.",
        )

    numeric_values = (
        values["km_driven"],
        values["mileage"],
        values["engine"],
        values["max_power"],
        values["seats"],
    )
    if any(not math.isfinite(value) or value < 0 for value in numeric_values):
        return render_form(error="Numeric car details cannot be negative.")
    if values["seats"] == 0:
        return render_form(error="Seats must be at least 1.")

    model_input = values.copy()
    for feature in CATEGORICAL_FEATURES:
        model_input[feature] = LABEL_MAPPINGS[feature][values[feature]]
    model_input["year"] = LABEL_MAPPINGS["year"][values["year"]]

    input_frame = pd.DataFrame(
        [[model_input[feature] for feature in FEATURES]],
        columns=FEATURES,
    )
    selling_price = float(model.predict(input_frame)[0])

    return render_form(
        selling_price=f"₹ {selling_price:,.0f}",
    )


if __name__ == "__main__":
    app.run(debug=True)