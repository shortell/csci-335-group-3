"""
svm.py
Classification: SVC (RBF kernel, 3-class)
Regression:     SVR (RBF kernel)
"""

import numpy as np
from sklearn.svm import SVC, SVR
from sklearn.metrics import classification_report, mean_squared_error, mean_absolute_error, r2_score

from loader import load_data

# LOAD
data = load_data()
X_train, X_val, X_test = data["X_train"], data["X_val"], data["X_test"]
y_train_cls, y_val_cls, y_test_cls = data["y_train_cls"], data["y_val_cls"], data["y_test_cls"]
y_train_reg, y_val_reg, y_test_reg = data["y_train_reg"], data["y_val_reg"], data["y_test_reg"]

# CLASSIFICATION
cls_model = SVC(
    kernel="rbf",
    C=1.0,
    gamma="scale",
    class_weight="balanced",
    random_state=42
)

cls_model.fit(X_train, y_train_cls)

print("=" * 50)
print("SVM — Classification (Val)")
print("=" * 50)
val_preds = cls_model.predict(X_val)
print(classification_report(y_val_cls, val_preds))

print("=" * 50)
print("SVM — Classification (Test)")
print("=" * 50)
test_preds = cls_model.predict(X_test)
print(classification_report(y_test_cls, test_preds))

# REGRESSION
reg_model = SVR(kernel="rbf", C=1.0, epsilon=0.1)
reg_model.fit(X_train, y_train_reg)

print("=" * 50)
print("SVR — Regression (Val)")
print("=" * 50)
val_reg_preds = reg_model.predict(X_val)
print(f"  RMSE: {np.sqrt(mean_squared_error(y_val_reg, val_reg_preds)):.4f}")
print(f"  MAE:  {mean_absolute_error(y_val_reg, val_reg_preds):.4f}")
print(f"  R(squared):   {r2_score(y_val_reg, val_reg_preds):.4f}")

print("\n" + "=" * 50)
print("SVR — Regression (Test)")
print("=" * 50)
test_reg_preds = reg_model.predict(X_test)
print(f"  RMSE: {np.sqrt(mean_squared_error(y_test_reg, test_reg_preds)):.4f}")
print(f"  MAE:  {mean_absolute_error(y_test_reg, test_reg_preds):.4f}")
print(f"  R(squared):   {r2_score(y_test_reg, test_reg_preds):.4f}")
