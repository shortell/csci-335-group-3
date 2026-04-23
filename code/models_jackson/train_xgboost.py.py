import pandas as pd
import numpy as np
import sys, os
script_dir = os.path.abspath(os.path.dirname(__file__))
if script_dir in sys.path:
    sys.path.remove(script_dir)
from sklearn.model_selection import train_test_split, ParameterGrid
from xgboost import XGBClassifier
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.metrics import balanced_accuracy_score, f1_score, precision_score, fbeta_score, classification_report
from sklearn.impute import SimpleImputer
from sklearn.utils.class_weight import compute_sample_weight

INPUT  = r'data\final\musk_events_k10_replies_True.csv'
REPORT = r'xgboost_report.txt'
TARGET = 'max_z_next5'
THRESH = 1.5
FEATURES = [
    'mentions_tesla', 'is_reply', 'is_quote', 'is_retweet',
    'positive', 'negative', 'neutral',
    'close_delta_z', 'volume_delta_z',
    'price_cv', 'volume_cv', 'close_position',
    'up_bar_volume_ratio', 'bullish_bar_ratio'
]

df = pd.read_csv(INPUT)
X = SimpleImputer(strategy='median').fit_transform(df[FEATURES])
y_raw = df[TARGET].apply(lambda z: 'buy' if not pd.isna(z) and z > THRESH else ('dont_buy' if not pd.isna(z) else np.nan))
mask = y_raw.notna().values
X, y = X[mask], np.array(y_raw[y_raw.notna()])

X_tv, X_test, y_tv, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
X_train, X_val, y_train, y_val = train_test_split(X_tv, y_tv, test_size=0.2, random_state=42, stratify=y_tv)

scaler = MinMaxScaler()
Xtr = scaler.fit_transform(X_train)
Xva = scaler.transform(X_val)
Xte = scaler.transform(X_test)

le = LabelEncoder()
ytr_enc = le.fit_transform(y_train)
yva_enc = le.transform(y_val)
# buy is index 0 after alphabetical encoding ('buy' < 'dont_buy')
BUY_IDX = list(le.classes_).index('buy')

def predict_thresh(clf, Xs, t=0.5):
    return np.where(clf.predict_proba(Xs)[:, BUY_IDX] >= t, 'buy', 'dont_buy')

def tune_threshold(clf, Xva, y_val):
    """Find threshold maximising Fbeta(0.5) — weights precision 2x recall."""
    best_t, best_s = 0.5, 0.0
    for t in np.arange(0.30, 0.85, 0.05):
        s = fbeta_score(y_val, predict_thresh(clf, Xva, t), beta=0.5, pos_label='buy', zero_division=0)
        if s > best_s:
            best_s, best_t = s, float(t)
    return best_t

def get_metrics(clf, Xs, ys_str, t=0.5):
    yp = predict_thresh(clf, Xs, t)
    return {
        'bal_acc': balanced_accuracy_score(ys_str, yp),
        'f1':      f1_score(ys_str, yp, pos_label='buy', zero_division=0),
        'prec':    precision_score(ys_str, yp, pos_label='buy', zero_division=0),
    }

param_grid = {
    'learning_rate': [0.01, 0.05, 0.1, 0.2],
    'max_depth': [3, 5, 7],
    'n_estimators': [200],
    'subsample': [0.7, 0.8, 1.0],
    'colsample_bytree': [0.7, 1.0],
    'min_child_weight': [1, 3]
}

results = []
for params in ParameterGrid(param_grid):
    try:
        sw = compute_sample_weight('balanced', y=ytr_enc)
        clf = XGBClassifier(eval_metric='logloss', random_state=42, n_jobs=-1,
                            early_stopping_rounds=20, **params)
        clf.fit(Xtr, ytr_enc, eval_set=[(Xva, yva_enc)], sample_weight=sw, verbose=False)
        m = get_metrics(clf, Xva, y_val)
        composite = (m['bal_acc'] + m['f1'] + m['prec']) / 3
        results.append({**m, 'composite': composite, 'params': params, 'clf': clf})
    except Exception:
        continue

best_acc       = max(results, key=lambda r: r['bal_acc'])
best_f1        = max(results, key=lambda r: r['f1'])
best_composite = max(results, key=lambda r: r['composite'])
best_composite['threshold'] = tune_threshold(best_composite['clf'], Xva, y_val)

n, b = len(y), (y == 'buy').sum()
lines = [
    "XGBoost Report", "=" * 55,
    f"Target: {TARGET}  |  Threshold > {THRESH}",
    f"Total: {n}  |  Buy: {b} ({100*b/n:.1f}%)  |  No-Buy: {n-b} ({100*(n-b)/n:.1f}%)",
    f"Train: {len(y_train)}  |  Val: {len(y_val)}  |  Test: {len(y_test)}",
    f"Features ({len(FEATURES)}): {', '.join(FEATURES)}", "",
]

def model_block(title, r, t=0.5):
    m_tr = get_metrics(r['clf'], Xtr, y_train, t)
    m_va = get_metrics(r['clf'], Xva, y_val, t)
    m_te = get_metrics(r['clf'], Xte, y_test, t)
    note = f"  (prob threshold: {t:.2f})" if t != 0.5 else ""
    block = [f"[ {title} ]", f"Params: {r['params']}{note}",
             f"Best iteration: {r['clf'].best_iteration}",
             f"{'Metric':<18} {'Train':>8} {'Val':>8} {'Test':>8}", "-" * 45]
    for k, label in [('bal_acc', 'Balanced Acc'), ('f1', 'F1 (buy)'), ('prec', 'Precision (buy)')]:
        block.append(f"{label:<18} {m_tr[k]:>8.4f} {m_va[k]:>8.4f} {m_te[k]:>8.4f}")
    block.append("\nClassification Report (Test):")
    block.append(classification_report(y_test, predict_thresh(r['clf'], Xte, t), zero_division=0))
    return "\n".join(block)

lines.append(model_block("Best Balanced Accuracy",       best_acc))
lines.append(model_block("Best F1 (buy)",                 best_f1))
lines.append(model_block("Best Composite (Bal+F1+Prec)", best_composite, best_composite['threshold']))

with open(REPORT, 'w') as f:
    f.write("\n".join(lines))
print(f"Done. Report: {REPORT}")
