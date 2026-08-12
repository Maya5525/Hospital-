# ==========================================
# HOSPITAL ANALYTICS
# REGRESSION MODEL
# PREDICT PATIENTS ADMITTED
# ==========================================

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import GradientBoostingRegressor

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

from joblib import dump


# ==========================================
# 1. Load Dataset
# ==========================================

df = pd.read_csv("services_weekly.csv")

print("Dataset Loaded Successfully!")
print("Dataset Shape:", df.shape)


# ==========================================
# 2. Remove Duplicates
# ==========================================

df.drop_duplicates(inplace=True)


# ==========================================
# 3. Handle Missing Values
# ==========================================

numeric_columns = df.select_dtypes(
    include=np.number
).columns

for column in numeric_columns:
    df[column] = df[column].fillna(
        df[column].median()
    )


categorical_columns = df.select_dtypes(
    include="object"
).columns

for column in categorical_columns:

    if not df[column].mode().empty:

        df[column] = df[column].fillna(
            df[column].mode()[0]
        )


# ==========================================
# 4. Features
# ==========================================
# patients_admitted is NOT included
# because it is the value we want to predict.
# ==========================================

X = df[
    [
        "week",
        "month",
        "service",
        "available_beds",
        "patients_request",
        "patients_refused",
        "staff_morale",
        "event"
    ]
]


# ==========================================
# 5. Target
# ==========================================

y = df["patients_admitted"]


# ==========================================
# 6. Categorical Features
# ==========================================

categorical_features = [
    "service",
    "event"
]


# ==========================================
# 7. Preprocessing
# ==========================================

preprocessor = ColumnTransformer(

    transformers=[

        (
            "categorical",

            OneHotEncoder(
                handle_unknown="ignore"
            ),

            categorical_features
        )

    ],

    remainder="passthrough"
)


# ==========================================
# 8. Train / Test Split
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(

    X,
    y,

    test_size=0.20,

    random_state=42
)


# ==========================================
# 9. Regression Model
# ==========================================

model = Pipeline(

    steps=[

        (
            "preprocessor",
            preprocessor
        ),

        (
            "regressor",

            GradientBoostingRegressor(

                n_estimators=200,

                learning_rate=0.03,

                max_depth=2,

                random_state=42

            )
        )

    ]

)


# ==========================================
# 10. Train Model
# ==========================================

model.fit(
    X_train,
    y_train
)

print("\nModel Trained Successfully!")


# ==========================================
# 11. Make Predictions
# ==========================================

y_pred = model.predict(X_test)

# Patients admitted cannot be negative

y_pred = np.maximum(
    y_pred,
    0
)


# ==========================================
# 12. Model Evaluation
# ==========================================

mae = mean_absolute_error(
    y_test,
    y_pred
)

rmse = np.sqrt(
    mean_squared_error(
        y_test,
        y_pred
    )
)

r2 = r2_score(
    y_test,
    y_pred
)


print("\n")
print("=" * 50)
print("MODEL EVALUATION")
print("=" * 50)

print(f"MAE  : {mae:.4f}")
print(f"RMSE : {rmse:.4f}")
print(f"R²   : {r2:.4f}")


# ==========================================
# 13. Actual vs Predicted
# ==========================================

results = pd.DataFrame({

    "Actual Patients Admitted":
        y_test.values,

    "Predicted Patients Admitted":
        y_pred

})


print("\n")
print("=" * 50)
print("ACTUAL VS PREDICTED")
print("=" * 50)

print(results.head(10))


# ==========================================
# 14. New Prediction
# ==========================================

new_data = pd.DataFrame({

    "week": [10],

    "month": [3],

    "service": ["emergency"],

    "available_beds": [25],

    "patients_request": [80],

    "patients_refused": [20],

    "staff_morale": [72],

    "event": ["flu"]

})


prediction = model.predict(
    new_data
)[0]

prediction = max(
    0,
    prediction
)


print("\n")
print("=" * 50)
print("NEW PATIENT ADMISSION PREDICTION")
print("=" * 50)

print(
    f"Predicted Patients Admitted: "
    f"{prediction:.2f}"
)


# ==========================================
# 15. Save Model
# ==========================================

dump(
    model,
    "patients_admitted_model.pkl"
)

print("\nModel Saved Successfully!")
print("File: patients_admitted_model.pkl")
