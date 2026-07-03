#!/usr/bin/env python3
import open3d as o3d
import numpy as np
import glob
import pickle
from collections import Counter

ROI = {
    'x': (1.5, 4.0),
    'y': (-2.0, 3.3),
    'z': (-0.5, 1.5)
}

# 배경 높이 맵
print("배경 높이 맵 로드 중...")
files_s1 = sorted(glob.glob('pcd/S1/*.pcd'))
grid_res = 0.1
all_pts = []
for f in files_s1:
    pcd = o3d.io.read_point_cloud(f)
    pts = np.asarray(pcd.points)
    mask = (
        (pts[:,0] > ROI['x'][0]) & (pts[:,0] < ROI['x'][1]) &
        (pts[:,1] > ROI['y'][0]) & (pts[:,1] < ROI['y'][1]) &
        (pts[:,2] > ROI['z'][0]) & (pts[:,2] < ROI['z'][1])
    )
    all_pts.append(pts[mask])

all_pts = np.vstack(all_pts)
x_bins = np.arange(ROI['x'][0], ROI['x'][1], grid_res)
y_bins = np.arange(ROI['y'][0], ROI['y'][1], grid_res)
x_idx_bg = np.digitize(all_pts[:,0], x_bins) - 1
y_idx_bg = np.digitize(all_pts[:,1], y_bins) - 1

bg_max_z = {}
for i in range(len(all_pts)):
    key = (x_idx_bg[i], y_idx_bg[i])
    z = all_pts[i, 2]
    if key not in bg_max_z or z > bg_max_z[key]:
        bg_max_z[key] = z

def get_bg_max_z(x, y):
    xi = int((x - ROI['x'][0]) / grid_res)
    yi = int((y - ROI['y'][0]) / grid_res)
    vals = []
    for dx in [-1, 0, 1]:
        for dy in [-1, 0, 1]:
            key = (xi+dx, yi+dy)
            if key in bg_max_z:
                vals.append(bg_max_z[key])
    return max(vals) if vals else 0.0

def extract_person_cluster(pcd_file):
    pcd = o3d.io.read_point_cloud(pcd_file)
    pts = np.asarray(pcd.points)
    mask = (
        (pts[:,0] > ROI['x'][0]) & (pts[:,0] < ROI['x'][1]) &
        (pts[:,1] > ROI['y'][0]) & (pts[:,1] < ROI['y'][1]) &
        (pts[:,2] > ROI['z'][0]) & (pts[:,2] < ROI['z'][1])
    )
    pts = pts[mask]
    if len(pts) < 5:
        return None
    above_bg = np.zeros(len(pts))
    for i in range(len(pts)):
        above_bg[i] = pts[i,2] - get_bg_max_z(pts[i,0], pts[i,1])
    person_pts = pts[above_bg > 0.10]
    if len(person_pts) < 5:
        return None
    tmp = o3d.geometry.PointCloud()
    tmp.points = o3d.utility.Vector3dVector(person_pts)
    with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Error):
        labels = np.array(tmp.cluster_dbscan(
            eps=0.25, min_points=5, print_progress=False))
    if len(labels) == 0 or labels.max() < 0:
        return None
    largest = np.argmax(np.bincount(labels[labels >= 0]))
    result = person_pts[labels == largest]
    return result if len(result) >= 5 else None

def extract_single_features(person_pts, prev_center=None):
    h  = person_pts[:,2].max() - person_pts[:,2].min()
    w  = person_pts[:,0].max() - person_pts[:,0].min()
    d  = person_pts[:,1].max() - person_pts[:,1].min()
    n  = len(person_pts)
    hw = h / (w + 1e-6)
    center = person_pts.mean(axis=0)
    delta_pos = np.linalg.norm(center[:2] - prev_center[:2]) if prev_center is not None else 0.0
    z_threshold = np.percentile(person_pts[:,2], 80)
    upper_pts = person_pts[person_pts[:,2] >= z_threshold]
    upper_z_mean = upper_pts[:,2].mean() if len(upper_pts) > 0 else person_pts[:,2].max()
    return np.array([h, w, d, n, hw, delta_pos, upper_z_mean]), center

# 모델 로드
print("모델 로드 중...")
with open('rf_model_v3.pkl', 'rb') as f:
    model_data = pickle.load(f)
clf     = model_data['model']
scaler  = model_data['scaler']
classes = model_data['classes']
print(f"클래스: {classes}")

# S8 feature 추출
print("\nS8 feature 추출 중...")
files_s8 = sorted(glob.glob('pcd/S8/*.pcd'))
WINDOW = 10

raw_features = []
raw_centers  = []
prev_center  = None

for i, f in enumerate(files_s8):
    if i % 300 == 0:
        print(f"  {i}/{len(files_s8)}...")
    person_pts = extract_person_cluster(f)
    if person_pts is None:
        raw_features.append(None)
        raw_centers.append(None)
        prev_center = None
        continue
    feat, center = extract_single_features(person_pts, prev_center)
    prev_center = center
    raw_features.append(feat)
    raw_centers.append(center)

# Temporal feature + 예측
print("\n예측 중...")
predictions  = []
frame_indices = []
confidences  = []

for i in range(len(raw_features)):
    if raw_features[i] is None:
        continue
    window_feats = [
        raw_features[j]
        for j in range(max(0,i-WINDOW), min(len(raw_features),i+WINDOW+1))
        if raw_features[j] is not None
    ]
    if len(window_feats) < 3:
        continue

    window_arr     = np.array(window_feats)
    h_values       = window_arr[:, 0]
    pos_values     = window_arr[:, 5]
    upper_z_values = window_arr[:, 6]

    temporal = np.array([
        h_values.mean(), h_values.std(), h_values.max()-h_values.min(),
        pos_values.mean(), pos_values.std(), pos_values.sum(),
        upper_z_values.std(), upper_z_values.max()-upper_z_values.min(), upper_z_values.mean()
    ])

    combined = np.concatenate([raw_features[i], temporal])
    combined_scaled = scaler.transform([combined])

    pred = clf.predict(combined_scaled)[0]
    prob = clf.predict_proba(combined_scaled)[0]
    confidence = prob.max()

    predictions.append(pred)
    frame_indices.append(i)
    confidences.append(confidence)

print(f"\nS8 감지된 프레임: {len(predictions)}/{len(files_s8)} ({len(predictions)/len(files_s8)*100:.1f}%)")
print(f"평균 신뢰도: {np.mean(confidences):.3f}")

print(f"\n예측 클래스 분포:")
for cls, cnt in sorted(Counter(predictions).items()):
    pct = cnt/len(predictions)*100
    avg_conf = np.mean([confidences[i] for i,p in enumerate(predictions) if p==cls])
    print(f"  {cls:22s}: {cnt:4d}프레임 ({pct:5.1f}%)  평균신뢰도={avg_conf:.3f}")

# 시간순 예측 출력 (처음 100프레임)
print(f"\n시간순 예측 (처음 50프레임):")
for i in range(min(50, len(predictions))):
    t = frame_indices[i] / 10.0
    print(f"  t={t:6.1f}s  {predictions[i]:22s}  conf={confidences[i]:.3f}")

# 저장
np.save('s8_predictions.npy', np.array(predictions))
np.save('s8_frame_indices.npy', np.array(frame_indices))
np.save('s8_confidences.npy', np.array(confidences))
print(f"\n저장 완료: s8_predictions.npy")