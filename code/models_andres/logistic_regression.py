"""
logistic_regression.py
Classification: LogisticRegression (3-class: down / neutral / up)
Regression:     Ridge Regression
"""

import numpy as np
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import classification_report, mean_squared_error, mean_absolute_error, r2_score

from loader import load_data

# LOAD
data = load_data()
X_train, X_val, X_test = data["X_train"], data["X_val"], data["X_test"]
y_train_cls, y_val_cls, y_test_cls = data["y_train_cls"], data["y_val_cls"], data["y_test_cls"]
y_train_reg, y_val_reg, y_test_reg = data["y_train_reg"], data["y_val_reg"], data["y_test_reg"]

# CLASSIFICATION
cls_model = LogisticRegression(
    max_iter=1000,
    C=1.0,
    class_weight="balanced",
    random_state=42
)

cls_model.fit(X_train, y_train_cls)

print("=" * 50)
print("LOGISTIC REGRESSION — Classification (Val)")
print("=" * 50)
val_preds = cls_model.predict(X_val)
print(classification_report(y_val_cls, val_preds))

print("=" * 50)
print("LOGISTIC REGRESSION — Classification (Test)")
print("=" * 50)
test_preds = cls_model.predict(X_test)
print(classification_report(y_test_cls, test_preds))

# REGRESSION
reg_model = Ridge(alpha=1.0)
reg_model.fit(X_train, y_train_reg)

print("=" * 50)
print("RIDGE REGRESSION — Regression (Val)")
print("=" * 50)
val_reg_preds = reg_model.predict(X_val)
print(f"  RMSE: {np.sqrt(mean_squared_error(y_val_reg, val_reg_preds)):.4f}")
print(f"  MAE:  {mean_absolute_error(y_val_reg, val_reg_preds):.4f}")
print(f"  R (squared):   {r2_score(y_val_reg, val_reg_preds):.4f}")

print("\n" + "=" * 50)
print("RIDGE REGRESSION — Regression (Test)")
print("=" * 50)
test_reg_preds = reg_model.predict(X_test)
print(f"  RMSE: {np.sqrt(mean_squared_error(y_test_reg, test_reg_preds)):.4f}")
print(f"  MAE:  {mean_absolute_error(y_test_reg, test_reg_preds):.4f}")
print(f"  R(squared):   {r2_score(y_test_reg, test_reg_preds):.4f}")
