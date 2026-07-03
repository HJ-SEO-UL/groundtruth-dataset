#!/usr/bin/env python3
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix
from collections import Counter
import pickle
import os

os.makedirs('results_figures', exist_ok=True)

# 스타일
plt.rcParams.update({
    'font.family':       'DejaVu Sans',
    'font.size':         11,
    'axes.facecolor':    '#0D0D0D',
    'figure.facecolor':  '#0D0D0D',
    'axes.edgecolor':    '#333333',
    'axes.labelcolor':   'white',
    'xtick.color':       'white',
    'ytick.color':       'white',
    'text.color':        'white',
    'grid.color':        '#222222',
    'grid.linewidth':    0.5,
})

CLASS_MAP = {
    'idle_front': 'IdleStanding', 'idle_side': 'IdleStanding', 'idle_back': 'IdleStanding',
    'walking_slow': 'Walking', 'walking_normal': 'Walking', 'walking_fast': 'Walking',
    'squat_far': 'Squatting', 'squat_mid': 'Squatting', 'squat_far2': 'Squatting',
    'squatting_moving': 'SquattingMoving',
    'overhead_motion':  'Watering',
    'rhythmic_ground':  'GroundToolUse\n(Ho-Mi)',
}

CLASS_COLORS = {
    'IdleStanding':        '#CCCCCC',
    'Walking':             '#3498DB',
    'Squatting':           '#E67E22',
    'SquattingMoving':     '#E74C3C',
    'Watering':            '#1ABC9C',
    'GroundToolUse\n(Ho-Mi)': '#9B59B6',
}

# 데이터 로드 및 정제
print("데이터 로드 중...")
X     = np.load('features_X_v3.npy')
y_raw = np.load('features_y_v3.npy')
y     = np.array([CLASS_MAP[l] for l in y_raw])

scaler_pre = StandardScaler()
X_pre      = scaler_pre.fit_transform(X)
clean_mask = np.ones(len(X), dtype=bool)
for cls in np.unique(y):
    idx  = np.where(y == cls)[0]
    iso  = IsolationForest(contamination=0.15, random_state=42)
    pred = iso.fit_predict(X_pre[idx])
    clean_mask[idx[pred == -1]] = False

X_clean = X[clean_mask]
y_clean = y[clean_mask]
scaler  = StandardScaler()
X_scaled = scaler.fit_transform(X_clean)

classes = sorted(np.unique(y_clean))
clf     = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
cv      = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# CV confusion matrix
print("CV Confusion Matrix 계산 중...")
all_true, all_pred = [], []
for train_idx, test_idx in cv.split(X_scaled, y_clean):
    clf.fit(X_scaled[train_idx], y_clean[train_idx])
    all_pred.extend(clf.predict(X_scaled[test_idx]))
    all_true.extend(y_clean[test_idx])

cm = confusion_matrix(all_true, all_pred, labels=classes)

# 최종 학습
clf.fit(X_scaled, y_clean)

# ─────────────────────────────────────────────
# 그림 1 — Confusion Matrix
# ─────────────────────────────────────────────
print("그림 1: Confusion Matrix...")
fig, ax = plt.subplots(figsize=(10, 8))
fig.patch.set_facecolor('#0D0D0D')
ax.set_facecolor('#0D0D0D')

cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
im = ax.imshow(cm_norm, cmap='Blues', vmin=0, vmax=1)

short = [c.replace('\n', '/') for c in classes]
ax.set_xticks(range(len(classes)))
ax.set_yticks(range(len(classes)))
ax.set_xticklabels(short, rotation=30, ha='right', fontsize=10, color='white')
ax.set_yticklabels(short, fontsize=10, color='white')

for i in range(len(classes)):
    for j in range(len(classes)):
        val = cm_norm[i,j]
        txt = f"{val:.2f}\n({cm[i,j]})"
        color = 'white' if val < 0.5 else '#0D0D0D'
        ax.text(j, i, txt, ha='center', va='center',
                fontsize=9, color=color, fontweight='bold')

recall_vals = [cm[i,i]/cm[i].sum()*100 for i in range(len(classes))]
for i, r in enumerate(recall_vals):
    ax.text(len(classes)+0.1, i, f'{r:.1f}%',
            va='center', fontsize=9,
            color=list(CLASS_COLORS.values())[i % len(CLASS_COLORS)])

ax.set_xlabel('Predicted', fontsize=12, color='white', labelpad=10)
ax.set_ylabel('Actual',    fontsize=12, color='white', labelpad=10)
ax.set_title('HOMI — Activity Classification\nConfusion Matrix (5-fold CV)',
             fontsize=13, color='white', pad=15, fontweight='bold')

cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
cb.ax.yaxis.set_tick_params(color='white')
plt.setp(cb.ax.yaxis.get_ticklabels(), color='white')

acc = sum(cm[i,i] for i in range(len(classes))) / cm.sum()
ax.text(0.02, -0.12, f'Overall Accuracy: {acc*100:.1f}%',
        transform=ax.transAxes, fontsize=11,
        color='#1ABC9C', fontweight='bold')

plt.tight_layout()
plt.savefig('results_figures/01_confusion_matrix.png', dpi=150,
            bbox_inches='tight', facecolor='#0D0D0D')
plt.close()
print("  저장: results_figures/01_confusion_matrix.png")

# ─────────────────────────────────────────────
# 그림 2 — Recall per Class (바 차트)
# ─────────────────────────────────────────────
print("그림 2: Recall per Class...")
fig, ax = plt.subplots(figsize=(10, 5))
fig.patch.set_facecolor('#0D0D0D')
ax.set_facecolor('#0D0D0D')

short_labels = [c.replace('\n', '/') for c in classes]
colors       = [list(CLASS_COLORS.values())[i % len(CLASS_COLORS)]
                for i in range(len(classes))]

bars = ax.barh(short_labels, recall_vals, color=colors, edgecolor='none', height=0.6)

for bar, val in zip(bars, recall_vals):
    ax.text(val + 0.5, bar.get_y() + bar.get_height()/2,
            f'{val:.1f}%', va='center', fontsize=11,
            color='white', fontweight='bold')

ax.axvline(x=acc*100, color='white', linestyle='--', linewidth=1, alpha=0.5)
ax.text(acc*100 + 0.3, -0.6, f'Avg {acc*100:.1f}%',
        fontsize=9, color='white', alpha=0.7)

ax.set_xlim(0, 110)
ax.set_xlabel('Recall (%)', fontsize=12, color='white')
ax.set_title('HOMI — Recall per Activity Class',
             fontsize=13, color='white', pad=15, fontweight='bold')
ax.grid(axis='x', alpha=0.3)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.tight_layout()
plt.savefig('results_figures/02_recall_per_class.png', dpi=150,
            bbox_inches='tight', facecolor='#0D0D0D')
plt.close()
print("  저장: results_figures/02_recall_per_class.png")

# ─────────────────────────────────────────────
# 그림 3 — Feature Importance
# ─────────────────────────────────────────────
print("그림 3: Feature Importance...")
feature_names = [
    'h', 'w', 'd', 'n', 'h/w', 'Δpos', 'upper_z',
    'h_mean', 'h_std', 'h_range',
    'pos_mean', 'pos_std', 'pos_total',
    'upper_z_std', 'upper_z_range', 'upper_z_mean'
]
importances  = clf.feature_importances_
sorted_idx   = np.argsort(importances)
sorted_names = [feature_names[i] for i in sorted_idx]
sorted_imps  = importances[sorted_idx]

# 색상: temporal feature는 강조
temporal_set = {'h_mean','h_std','h_range','pos_mean','pos_std',
                'pos_total','upper_z_std','upper_z_range','upper_z_mean'}
bar_colors   = ['#9B59B6' if n in temporal_set else '#3498DB'
                for n in sorted_names]

fig, ax = plt.subplots(figsize=(9, 7))
fig.patch.set_facecolor('#0D0D0D')
ax.set_facecolor('#0D0D0D')

bars = ax.barh(sorted_names, sorted_imps * 100,
               color=bar_colors, edgecolor='none', height=0.7)

for bar, val in zip(bars, sorted_imps * 100):
    ax.text(val + 0.2, bar.get_y() + bar.get_height()/2,
            f'{val:.1f}%', va='center', fontsize=9, color='white')

legend_handles = [
    mpatches.Patch(color='#9B59B6', label='Temporal features'),
    mpatches.Patch(color='#3498DB', label='Geometric features'),
]
ax.legend(handles=legend_handles, loc='lower right',
          facecolor='#1A1A1A', edgecolor='#444444',
          labelcolor='white', fontsize=10)

ax.set_xlabel('Importance (%)', fontsize=12, color='white')
ax.set_title('HOMI — Feature Importance (Random Forest)',
             fontsize=13, color='white', pad=15, fontweight='bold')
ax.grid(axis='x', alpha=0.3)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.tight_layout()
plt.savefig('results_figures/03_feature_importance.png', dpi=150,
            bbox_inches='tight', facecolor='#0D0D0D')
plt.close()
print("  저장: results_figures/03_feature_importance.png")

# ─────────────────────────────────────────────
# 그림 4 — 정확도 향상 히스토리
# ─────────────────────────────────────────────
print("그림 4: 정확도 향상 히스토리...")
versions = ['SVM v1\n(6 features)', 'SVM v2\n(12 feat +\noutlier)', 'RF v2\n(12 feat +\noutlier)', 'RF v3\n(16 feat +\noutlier)']
accs     = [64.6, 82.9, 88.9, 95.1]
colors_v  = ['#555555', '#E67E22', '#3498DB', '#1ABC9C']

fig, ax = plt.subplots(figsize=(10, 5))
fig.patch.set_facecolor('#0D0D0D')
ax.set_facecolor('#0D0D0D')

bars = ax.bar(versions, accs, color=colors_v, edgecolor='none', width=0.5)

for bar, val in zip(bars, accs):
    ax.text(bar.get_x() + bar.get_width()/2, val + 0.5,
            f'{val:.1f}%', ha='center', va='bottom',
            fontsize=12, color='white', fontweight='bold')

ax.set_ylim(50, 100)
ax.set_ylabel('5-fold CV Accuracy (%)', fontsize=12, color='white')
ax.set_title('HOMI — Classification Accuracy Improvement',
             fontsize=13, color='white', pad=15, fontweight='bold')
ax.grid(axis='y', alpha=0.3)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# 화살표로 개선 표시
for i in range(len(accs)-1):
    diff = accs[i+1] - accs[i]
    ax.annotate(f'+{diff:.1f}%p',
                xy=(i+1, accs[i+1]-2),
                ha='center', fontsize=9,
                color='#F39C12', fontweight='bold')

plt.tight_layout()
plt.savefig('results_figures/04_accuracy_history.png', dpi=150,
            bbox_inches='tight', facecolor='#0D0D0D')
plt.close()
print("  저장: results_figures/04_accuracy_history.png")

# ─────────────────────────────────────────────
# 그림 5 — S8 타임라인
# ─────────────────────────────────────────────
print("그림 5: S8 타임라인...")
predictions   = np.load('s8_predictions.npy')
frame_indices = np.load('s8_frame_indices.npy')
confidences   = np.load('s8_confidences.npy')

PRED_COLOR_MAP = {
    'idle_standing':    '#CCCCCC',
    'walking':          '#3498DB',
    'squatting_static': '#E67E22',
    'squatting_moving': '#E74C3C',
    'overhead_motion':  '#1ABC9C',
    'rhythmic_ground':  '#9B59B6',
}

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 6),
                                gridspec_kw={'height_ratios': [3, 1]})
fig.patch.set_facecolor('#0D0D0D')
for ax in [ax1, ax2]:
    ax.set_facecolor('#0D0D0D')

# 실제 구간 배경
segments = [
    (0,    25.9,  'Squat+Walk',  '#1a1a2e'),
    (25.9, 70.1,  'Watering',    '#0d2137'),
    (70.1, 201.2, 'Ho-Mi',       '#1a0d2e'),
    (201.2,236.2, 'Squat',       '#1a1a0d'),
]
for t0, t1, name, bg in segments:
    ax1.axvspan(t0, t1, alpha=0.3, color=bg)
    ax1.text((t0+t1)/2, 0.92, name,
             ha='center', va='top', transform=ax1.get_xaxis_transform(),
             fontsize=9, color='#AAAAAA', style='italic')

# 예측 scatter
times = frame_indices / 10.0
for pred_cls, color in PRED_COLOR_MAP.items():
    mask = predictions == pred_cls
    if mask.sum() > 0:
        ax1.scatter(times[mask], confidences[mask],
                    c=color, s=4, alpha=0.6, label=pred_cls.replace('_',' ').title())

ax1.set_xlim(0, 240)
ax1.set_ylim(0, 1.05)
ax1.set_ylabel('Confidence', fontsize=10, color='white')
ax1.set_title('HOMI S8 — Mixed Activity Prediction Timeline',
              fontsize=13, color='white', pad=10, fontweight='bold')
ax1.legend(loc='lower right', fontsize=8,
           facecolor='#1A1A1A', edgecolor='#444444',
           labelcolor='white', ncol=3, markerscale=3)
ax1.grid(alpha=0.2)

# 하단: 예측 클래스 컬러바
for i, (fi, pred) in enumerate(zip(frame_indices, predictions)):
    t = fi / 10.0
    color = PRED_COLOR_MAP.get(pred, '#888888')
    ax2.barh(0, 0.1, left=t, height=1, color=color, edgecolor='none')

ax2.set_xlim(0, 240)
ax2.set_ylim(-0.5, 0.5)
ax2.set_xlabel('Time (s)', fontsize=10, color='white')
ax2.set_yticks([])
ax2.grid(False)
ax2.spines['top'].set_visible(False)
ax2.spines['left'].set_visible(False)
ax2.spines['right'].set_visible(False)

# 구간 구분선
for t0, _, _, _ in segments[1:]:
    ax1.axvline(x=t0, color='#444444', linewidth=1, linestyle='--')
    ax2.axvline(x=t0, color='#444444', linewidth=1, linestyle='--')

plt.tight_layout()
plt.savefig('results_figures/05_s8_timeline.png', dpi=150,
            bbox_inches='tight', facecolor='#0D0D0D')
plt.close()
print("  저장: results_figures/05_s8_timeline.png")

# ─────────────────────────────────────────────
# 그림 6 — 물통 검증
# ─────────────────────────────────────────────
print("그림 6: 물통 검증...")
fig, ax = plt.subplots(figsize=(7, 5))
fig.patch.set_facecolor('#0D0D0D')
ax.set_facecolor('#0D0D0D')

categories = ['Actual\nDiameter', 'Detected\nDiameter']
values     = [100.0, 99.9]
colors_w   = ['#3498DB', '#1ABC9C']

bars = ax.bar(categories, values, color=colors_w,
              edgecolor='none', width=0.4)

for bar, val in zip(bars, values):
    ax.text(bar.get_x() + bar.get_width()/2, val + 0.2,
            f'{val:.1f} cm', ha='center', va='bottom',
            fontsize=14, color='white', fontweight='bold')

ax.set_ylim(98, 102)
ax.set_ylabel('Diameter (cm)', fontsize=12, color='white')
ax.set_title('HOMI — Metric Accuracy Validation\nWater Tank Reference Object',
             fontsize=12, color='white', pad=15, fontweight='bold')
ax.grid(axis='y', alpha=0.3)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

ax.text(0.5, 0.15,
        'Measurement Error: 0.1 cm (0.1%)',
        ha='center', transform=ax.transAxes,
        fontsize=12, color='#F39C12', fontweight='bold')

plt.tight_layout()
plt.savefig('results_figures/06_tank_validation.png', dpi=150,
            bbox_inches='tight', facecolor='#0D0D0D')
plt.close()
print("  저장: results_figures/06_tank_validation.png")

print("\n모든 그림 저장 완료: results_figures/")
print("파일 목록:")
for f in sorted(os.listdir('results_figures')):
    print(f"  {f}")