#!/usr/bin/env python3
import open3d as o3d
import numpy as np
import glob
import os

ROI = {
    'x': (1.5, 4.0),
    'y': (-2.0, 3.3),
    'z': (-0.5, 1.5)
}

# 배경 높이 맵 구축
print("배경 높이 맵 구축 중...")
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
x_idx = np.digitize(all_pts[:,0], x_bins) - 1
y_idx = np.digitize(all_pts[:,1], y_bins) - 1

bg_max_z = {}
for i in range(len(all_pts)):
    key = (x_idx[i], y_idx[i])
    z = all_pts[i, 2]
    if key not in bg_max_z or z > bg_max_z[key]:
        bg_max_z[key] = z

print(f"배경 높이 맵: {len(bg_max_z)} 셀")

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

    # 상위 20% 포인트 (팔/머리 위치)
    z_threshold = np.percentile(person_pts[:,2], 80)
    upper_pts = person_pts[person_pts[:,2] >= z_threshold]
    upper_z_mean = upper_pts[:,2].mean() if len(upper_pts) > 0 else person_pts[:,2].max()

    return np.array([h, w, d, n, hw, delta_pos, upper_z_mean]), center

SESSIONS = ['S2', 'S3', 'S4', 'S5', 'S6', 'S7']
WINDOW = 10

all_features = []
all_labels   = []
all_sessions = []

for session in SESSIONS:
    pcd_folder  = f'pcd/{session}'
    labels_file = f'labels_{session}_labels.npy'
    files_npy   = f'labels_{session}_files.npy'

    if not os.path.exists(labels_file):
        continue

    pcd_files = np.load(files_npy)
    labels    = np.load(labels_file)

    print(f"\n{session} feature 추출 중... ({len(pcd_files)}프레임)")

    raw_features = []
    raw_labels   = []
    raw_centers  = []
    prev_center  = None

    for i, (pcd_file, label) in enumerate(zip(pcd_files, labels)):
        if i % 300 == 0:
            print(f"  {i}/{len(pcd_files)}...")

        person_pts = extract_person_cluster(pcd_file)
        if person_pts is None:
            raw_features.append(None)
            raw_labels.append(label)
            raw_centers.append(None)
            prev_center = None
            continue

        feat, center = extract_single_features(person_pts, prev_center)
        prev_center = center
        raw_features.append(feat)
        raw_labels.append(label)
        raw_centers.append(center)

    # Temporal feature 추가
    success = 0
    fail    = 0

    for i in range(len(raw_features)):
        if raw_features[i] is None:
            fail += 1
            continue

        window_feats = [
            raw_features[j]
            for j in range(max(0, i-WINDOW), min(len(raw_features), i+WINDOW+1))
            if raw_features[j] is not None
        ]

        if len(window_feats) < 3:
            fail += 1
            continue

        window_arr = np.array(window_feats)

        h_values       = window_arr[:, 0]
        pos_values     = window_arr[:, 5]
        upper_z_values = window_arr[:, 6]

        # 기본 temporal (6개)
        h_mean    = h_values.mean()
        h_std     = h_values.std()
        h_range   = h_values.max() - h_values.min()
        pos_mean  = pos_values.mean()
        pos_std   = pos_values.std()
        pos_total = pos_values.sum()

        # 팔 움직임 temporal (3개)
        upper_z_std    = upper_z_values.std()
        upper_z_range  = upper_z_values.max() - upper_z_values.min()
        upper_z_mean_t = upper_z_values.mean()

        combined = np.concatenate([
            raw_features[i],
            [h_mean, h_std, h_range, pos_mean, pos_std, pos_total,
             upper_z_std, upper_z_range, upper_z_mean_t]
        ])

        all_features.append(combined)
        all_labels.append(raw_labels[i])
        all_sessions.append(session)
        success += 1

    print(f"  성공: {success} / 실패: {fail} ({success/(success+fail)*100:.1f}%)")

X = np.array(all_features)
y = np.array(all_labels)
sessions = np.array(all_sessions)

np.save('features_X_v3.npy', X)
np.save('features_y_v3.npy', y)
np.save('features_sessions_v3.npy', sessions)

print(f"\n완료! 총 {len(X)} feature")
print(f"feature shape: {X.shape}  (기본 7개 + temporal 9개 = 16개)")
print(f"\n클래스별 분포:")
from collections import Counter
for cls, cnt in sorted(Counter(y).items()):
    print(f"  {cls}: {cnt}")