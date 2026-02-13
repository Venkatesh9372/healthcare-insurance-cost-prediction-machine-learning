# ### Import Required Libraries


import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neighbors import KNeighborsRegressor

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import pickle


# ### Load Dataset


df = pd.read_csv(r"C:\Users\Administrator\Downloads\insurance.csv")

df = df.rename(columns={
    'age': 'Age',
    'sex': 'Gender',
    'bmi': 'BMI',
    'children': 'Children',
    'smoker': 'Smoker',
    'region': 'Region',
    'charges': 'Charges'
})

df.drop_duplicates(inplace=True)


# ### Define Features & Target


X = df[['Age', 'Gender', 'BMI', 'Children', 'Smoker', 'Region']]
y = df['Charges']

# ### Train-Test Split


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42
)

# ### Column Transformer


categorical_cols = ['Gender', 'Smoker', 'Region']
numerical_cols = ['Age', 'BMI', 'Children']

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_cols),
        ('cat', OneHotEncoder(drop='first', handle_unknown='ignore'), categorical_cols)
    ]
)

# ### Model Pipelines


# ##### Linear Regression


lr_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('model', LinearRegression())
])


# ##### Decision Tree


dt_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('model', DecisionTreeRegressor(random_state=42))
])


# ##### Random Forest


rf_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('model', RandomForestRegressor(random_state=42))
])

# ##### Gradient Boosting


gb_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('model', GradientBoostingRegressor(random_state=42))
])

# ##### KNN


knn_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('model', KNeighborsRegressor(n_neighbors=7, weights='distance'))
])

# ### Train, Predict & Evaluate


def evaluate_model(model, X_train, X_test, y_train, y_test):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)

    return mae, mse, rmse, r2

# ### Evaluate All Models


models = {
    "Linear Regression": lr_pipeline,
    "Decision Tree": dt_pipeline,
    "Random Forest": rf_pipeline,
    "Gradient Boosting": gb_pipeline,
    "KNN": knn_pipeline
}

for name, model in models.items():
    mae, mse, rmse, r2 = evaluate_model(model, X_train, X_test, y_train, y_test)
    print(f"\n{name}")
    print(f"MAE: {mae:.2f}")
    print(f"MSE: {mse:.2f}")
    print(f"RMSE: {rmse:.2f}")
    print(f"R² Score: {r2:.4f}")


# ### Hyperparamater Tunning on the Gradient boosting Model


from sklearn.model_selection import GridSearchCV

# Hyperparameter grid for Gradient Boosting
param_grid = {
    "model__n_estimators": [100, 200],
    "model__learning_rate": [0.05, 0.1],
    "model__max_depth": [3, 4],
    "model__subsample": [0.8, 1.0]
}

# GridSearch on Gradient Boosting pipeline
gb_grid = GridSearchCV(
    estimator=gb_pipeline,
    param_grid=param_grid,
    cv=5,
    scoring="r2",
    n_jobs=-1
)

# Fit ONLY on training data
gb_grid.fit(X_train, y_train)

# Best tuned Gradient Boosting model
best_gb_model = gb_grid.best_estimator_

# Best hyperparameters
print("\nBest Gradient Boosting Hyperparameters:")
print(gb_grid.best_params_)


# ### Evaluated


mae, mse, rmse, r2 = evaluate_model(
    best_gb_model, X_train, X_test, y_train, y_test
)

print("\nTuned Gradient Boosting Performance:")
print(f"MAE: {mae:.2f}")
print(f"MSE: {mse:.2f}")
print(f"RMSE: {rmse:.2f}")
print(f"R² Score: {r2:.4f}")


import pickle

# Save the tuned Gradient Boosting pipeline
with open("health_care_insurance_model.pkl", "wb") as file:
    pickle.dump(best_gb_model, file)

print(" Gradient Boosting model saved as health_care_insurance_model.pkl")

