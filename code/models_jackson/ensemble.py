import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, ParameterGrid, cross_val_predict, cross_val_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import classification_report, accuracy_score, f1_score, precision_score
from sklearn.impute import SimpleImputer
import warnings
warnings.filterwarnings('ignore')

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, StackingClassifier
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier

input_file = r'data\final\musk_events_k9_replies_True.csv'
output_file = r'ensemble_report.txt'

df = pd.read_csv(input_file)

features = [
    'mentions_tesla', 'is_reply', 'is_quote', 'is_retweet',
    'positive', 'negative', 'neutral', 'close_delta_z',
    'volume_delta_z', 'price_cv', 'volume_cv', 'hour', 'day_of_week'
]

X = df[features].copy()
target = 'close_t3_z'
threshold = 0.5

imputer = SimpleImputer(strategy='median')
X_imputed = imputer.fit_transform(X)

def categorize_z(z):
    if pd.isna(z):
        return np.nan
    return 1 if z > threshold else 0 # 1 is Buy, 0 is Don't Buy (Integer mapping fixes XGBoost compatibility inside ensembles)

y = df[target].apply(categorize_z)
valid_idx = y.notna()

X_valid = X_imputed[valid_idx]
y_valid = y[valid_idx]

X_train, X_test, y_train, y_test = train_test_split(X_valid, y_valid, test_size=0.2, random_state=42, stratify=y_valid)

scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# --- Define Top Models ---
clf_lr = LogisticRegression(C=0.01, class_weight=None, solver='lbfgs', max_iter=2000, random_state=42)
clf_svm = SVC(C=0.1, class_weight='balanced', gamma='scale', kernel='linear', probability=True, random_state=42)
clf_rf = RandomForestClassifier(class_weight='balanced_subsample', max_depth=None, min_samples_leaf=2, min_samples_split=5, n_estimators=100, random_state=42, n_jobs=-1)
clf_mlp = MLPClassifier(activation='relu', alpha=0.001, hidden_layer_sizes=(8, 4), learning_rate='constant', max_iter=2000, random_state=42)
clf_xgb = XGBClassifier(colsample_bytree=0.8, learning_rate=0.05, max_depth=5, n_estimators=50, subsample=0.8, eval_metric='logloss', random_state=42, n_jobs=-1)

estimators = [
    ('lr', clf_lr),
    ('svm', clf_svm),
    ('rf', clf_rf),
    ('mlp', clf_mlp),
    ('xgb', clf_xgb)
]

# --- 1. Soft Voting Classifier ---
print("Training Voting Ensemble...")
voting_clf = VotingClassifier(estimators=estimators, voting='soft', n_jobs=-1)
voting_clf.fit(X_train_scaled, y_train)
voting_preds = voting_clf.predict(X_test_scaled)
voting_prec = precision_score(y_test, voting_preds, zero_division=0)

# --- 2. Tuned Stacking Classifier (with passthrough) ---
print("Generating Out-of-Fold predictions for Base Models (One-Time)...")
meta_features = []
base_models_fitted = []

for name, clf in estimators:
    # Get cross-validated probabilities for the meta-model to train on
    oof_preds = cross_val_predict(clf, X_train_scaled, y_train, cv=5, method='predict_proba', n_jobs=-1)[:, 1]
    meta_features.append(oof_preds)
    
    # Fit on the full training set for inference later
    clf.fit(X_train_scaled, y_train)
    base_models_fitted.append((name, clf))

# Horizontally stack (passthrough) original features + new model predictions
X_train_meta = np.hstack([X_train_scaled, np.column_stack(meta_features)])

# Define grid for the XGBoost Meta-Model
stacking_grid = {
    'learning_rate': [0.01, 0.05, 0.1],
    'max_depth': [2, 3, 4],
    'n_estimators': [50, 100],
    'subsample': [0.8, 1.0],
    'scale_pos_weight': [1, 1.5, 2] # Explicitly allows it to boost Recall vs Precision!
}

print("Running Grid Search on the XGBoost Meta-Model (Optimizing for F1)...")
best_meta_score = -1
best_meta_clf = None

for p in ParameterGrid(stacking_grid):
    meta_clf = XGBClassifier(eval_metric='logloss', random_state=42, n_jobs=-1, **p)
    # Validate strictly using F1 score to balance Precision and Recall perfectly
    scores = cross_val_score(meta_clf, X_train_meta, y_train, cv=5, scoring='f1', n_jobs=-1)
    mean_score = np.mean(scores)
    
    if mean_score > best_meta_score:
        best_meta_score = mean_score
        best_meta_clf = meta_clf

# Retrain the ultimate winner on the full meta-train set
best_meta_clf.fit(X_train_meta, y_train)

# Generate test meta-features
meta_features_test = []
for name, clf in base_models_fitted:
    test_preds = clf.predict_proba(X_test_scaled)[:, 1]
    meta_features_test.append(test_preds)

X_test_meta = np.hstack([X_test_scaled, np.column_stack(meta_features_test)])

stacking_preds = best_meta_clf.predict(X_test_meta)
stacking_prec = precision_score(y_test, stacking_preds, zero_division=0)
best_stacking_params = best_meta_clf.get_params()


with open(output_file, 'w') as f:
    f.write("ENSEMBLE MODELS REPORT (Binary Classification)\n")
    f.write("========================================================\n")
    f.write(f"Target: {target}  |  Threshold: > {threshold}\n")
    f.write("Features: Scaled via MinMaxScaler. XGBoost target encoded directly as 1 and 0.\n")
    f.write("Base Models Included: Logistic Regression, SVM, Random Forest, MLP, XGBoost\n")
    f.write("Note: Models instantiated using the exact #1 configurations found in their grid searches.\n")
    f.write("========================================================\n\n")

    f.write("--- 1. SOFT VOTING CLASSIFIER ---\n")
    f.write(f"Buy Precision: {voting_prec:.4f}\n")
    f.write(f"Accuracy:      {accuracy_score(y_test, voting_preds):.4f}\n")
    f.write(f"F1 Score:      {f1_score(y_test, voting_preds, zero_division=0):.4f}\n")
    f.write(f"Predicted Buys: {voting_preds.sum()} (Out of {len(y_test)} total cases)\n")
    f.write("\nVoting Classification Report:\n")
    f.write(classification_report(y_test, voting_preds, target_names=["Don't Buy (0)", "Buy (1)"], zero_division=0))
    f.write("\n\n" + "-"*80 + "\n\n")

    f.write("--- 2. STACKING CLASSIFIER (TUNED XGBOOST META-MODEL) ---\n")
    f.write(f"Best Meta-Model Params: { {k: best_stacking_params[k] for k in ['learning_rate', 'max_depth', 'n_estimators', 'subsample', 'scale_pos_weight']} }\n")
    f.write(f"Buy Precision: {stacking_prec:.4f}\n")
    f.write(f"Accuracy:      {accuracy_score(y_test, stacking_preds):.4f}\n")
    f.write(f"F1 Score:      {f1_score(y_test, stacking_preds, zero_division=0):.4f}\n")
    f.write(f"Predicted Buys: {stacking_preds.sum()} (Out of {len(y_test)} total cases)\n")
    f.write("\nStacking Classification Report:\n")
    f.write(classification_report(y_test, stacking_preds, target_names=["Don't Buy (0)", "Buy (1)"], zero_division=0))
    
print(f"Done! Evaluated Voting and Stacking Ensemble Models.")
print(f"Results saved to {output_file}")
