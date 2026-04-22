import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, ParameterGrid
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import classification_report, accuracy_score, f1_score
from sklearn.impute import SimpleImputer
import os

# Define file paths
input_file = r'data\final\musk_events_k10_replies_True.csv'
output_file = r'logistic_regression_report.txt'

# Load the data
df = pd.read_csv(input_file)

# Features
features = [
    'mentions_tesla', 'is_reply', 'is_quote', 'is_retweet',
    'positive', 'negative', 'neutral', 'close_delta_z',
    'volume_delta_z', 'price_cv', 'volume_cv'
]

X = df[features].copy()
target = 'close_t3_z'

report_content = "Logistic Regression Models Report (Binary Classification)\n"
report_content += "========================================================\n\n"

imputer = SimpleImputer(strategy='median')
X_imputed = imputer.fit_transform(X)

threshold = 0.5

if target not in df.columns:
    print(f"Target column missing: {target}")
    exit()

report_content += f"Target: {target}\n"
report_content += f"Threshold: > {threshold} (Buy 1), <= {threshold} (Don't Buy 0)\n"
report_content += "-" * 40 + "\n"

def categorize_z(z):
    if pd.isna(z):
        return np.nan
    if z > threshold:
        return 'buy'
    else:
        return 'dont_buy'

y = df[target].apply(categorize_z)

valid_idx = y.notna()
X_valid = X_imputed[valid_idx]
y_valid = y[valid_idx]

X_train, X_test, y_train, y_test = train_test_split(X_valid, y_valid, test_size=0.2, random_state=42, stratify=y_valid)

scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

param_grid = {
    'C': [0.01, 0.1, 1.0, 10.0],
    'solver': ['lbfgs', 'newton-cg'],
    'class_weight': [None, 'balanced']
}

grid = ParameterGrid(param_grid)
results = []

report_content += f"Total Valid Samples: {len(y_valid)}\n"
report_content += f"Testing {len(list(grid))} hyperparameter combinations...\n\n"

for params in grid:
    try:
        clf = LogisticRegression(max_iter=5000, random_state=42, n_jobs=-1, **params)
        clf.fit(X_train_scaled, y_train)
        
        y_pred = clf.predict(X_test_scaled)
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, pos_label='buy', zero_division=0)
        from sklearn.metrics import precision_score
        precision = precision_score(y_test, y_pred, pos_label='buy', zero_division=0)
        rep = classification_report(y_test, y_pred, zero_division=0)
        
        results.append({
            'params': params,
            'acc': acc,
            'f1': f1,
            'precision': precision,
            'rep': rep
        })
    except Exception as e:
        continue

# Sort by Precision descending
results.sort(key=lambda x: x['precision'], reverse=True)

report_content += "MODELS SORTED BY BUY PRECISION (DESCENDING):\n"
report_content += "-" * 50 + "\n"
for rank, res in enumerate(results, 1):
    report_content += f"Rank {rank} | Prec(Buy): {res['precision']:.4f} | F1: {res['f1']:.4f} | Acc: {res['acc']:.4f} | Params: {res['params']}\n"

if results:
    top3_acc = sorted(results, key=lambda x: x['acc'], reverse=True)[:3]
    top3_f1 = sorted(results, key=lambda x: x['f1'], reverse=True)[:3]
    top3_prec = sorted(results, key=lambda x: x['precision'], reverse=True)[:3]

    report_content += "\n========================================================\n"
    report_content += "TOP 3 MODELS BY BUY PRECISION:\n"
    report_content += "-" * 50 + "\n"
    for i, res in enumerate(top3_prec, 1):
        report_content += f"{i}. Prec: {res['precision']:.4f} | F1: {res['f1']:.4f} | Acc: {res['acc']:.4f}\nParams: {res['params']}\n"
        if i == 1:
            report_content += f"Best Precision Classification Report:\n{res['rep']}\n"

    report_content += "========================================================\n"
    report_content += "TOP 3 MODELS BY F1 SCORE:\n"
    report_content += "-" * 50 + "\n"
    for i, res in enumerate(top3_f1, 1):
        report_content += f"{i}. F1: {res['f1']:.4f} | Prec: {res['precision']:.4f} | Acc: {res['acc']:.4f}\nParams: {res['params']}\n"

    report_content += "========================================================\n"
    report_content += "TOP 3 MODELS BY ACCURACY:\n"
    report_content += "-" * 50 + "\n"
    for i, res in enumerate(top3_acc, 1):
        report_content += f"{i}. Acc: {res['acc']:.4f} | Prec: {res['precision']:.4f} | F1: {res['f1']:.4f}\nParams: {res['params']}\n"
else:
    report_content += "No valid models were trained.\n"

with open(output_file, 'w') as f:
    f.write(report_content)

print(f"Report saved to {output_file}")
