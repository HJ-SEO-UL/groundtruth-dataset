#!/usr/bin/env python3
import numpy as np
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import confusion_matrix, classification_report
from collections import Counter
import pickle

X = np.load('features_X_v3.npy')
y_raw = np.load('features_y_v3.npy')

class_map = {
    'idle_front': 'idle_standing', 'idle_side': 'idle_standing', 'idle_back': 'idle_standing',
    'walking_slow': 'walking', 'walking_normal': 'walking', 'walking_fast': 'walking',
    'squat_far': 'squatting_static', 'squat_mid': 'squatting_static', 'squat_far2': 'squatting_static',
    'squatting_moving': 'squatting_moving',
    'overhead_motion': 'overhead_motion',
    'rhythmic_ground': 'rhythmic_ground',
}
y = np.array([class_map[l] for l in y_raw])

# Outlier 제거
print("Outlier 제거 중...")
scaler_pre = StandardScaler()
X_pre = scaler_pre.fit_transform(X)
clean_mask = np.ones(len(X), dtype=bool)
for cls in np.unique(y):
    idx = np.where(y == cls)[0]
    iso = IsolationForest(contamination=0.15, random_state=42)
    preds = iso.fit_predict(X_pre[idx])
    clean_mask[idx[preds == -1]] = False

X_clean = X[clean_mask]
y_clean = y[clean_mask]
print(f"정제 후: {len(X_clean)} 샘플")

print(f"\n클래스별 분포:")
for cls, cnt in sorted(Counter(y_clean).items()):
    print(f"  {cls}: {cnt}")

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_clean)

# 5-fold CV
print(f"\n5-fold Cross Validation...")
clf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_val_score(clf, X_scaled, y_clean, cv=cv, scoring='accuracy')

print(f"\n=== Random Forest v3 5-fold CV ===")
for i, s in enumerate(scores):
    print(f"  Fold {i+1}: {s:.4f}")
print(f"평균 정확도: {scores.mean():.4f} ± {scores.std():.4f}")

# 최종 학습
print(f"\n최종 모델 학습 중...")
clf.fit(X_scaled, y_clean)

classes = sorted(np.unique(y_clean))
print(f"\n=== 분류 보고서 ===")
print(classification_report(y_clean, clf.predict(X_scaled)))

# Feature importance
print(f"\n=== Feature Importance ===")
feature_names = [
    'h', 'w', 'd', 'n', 'hw', 'delta_pos', 'upper_z_mean',
    'h_mean', 'h_std', 'h_range',
    'pos_mean', 'pos_std', 'pos_total',
    'upper_z_std', 'upper_z_range', 'upper_z_mean_t'
]
for name, imp in sorted(zip(feature_names, clf.feature_importances_), key=lambda x: -x[1]):
    print(f"  {name:16s}: {imp:.4f}")

# 모델 저장
with open('rf_model_v3.pkl', 'wb') as f:
    pickle.dump({
        'model': clf,
        'scaler': scaler,
        'classes': classes,
        'feature_names': feature_names
    }, f)
print(f"\nrf_model_v3.pkl 저장 완료")

print(f"\n=== 전체 성능 히스토리 ===")
print(f"SVM v1 (6 features):                     64.6%")
print(f"SVM v2 (12 features + outlier):          82.9%")
print(f"RF v2  (12 features + outlier):          88.9%")
print(f"RF v3  (16 features + outlier):          {scores.mean()*100:.1f}%")