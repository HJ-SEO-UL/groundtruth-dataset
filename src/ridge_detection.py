#!/usr/bin/env python3
import open3d as o3d
import numpy as np
import glob

ROI = {
    'x': (-1.2, 3.85),
    'y': (-2.0, 3.3),
    'z': (-2.0, 1.5)
}

def apply_roi(points):
    mask = (
        (points[:,0] > ROI['x'][0]) & (points[:,0] < ROI['x'][1]) &
        (points[:,1] > ROI['y'][0]) & (points[:,1] < ROI['y'][1]) &
        (points[:,2] > ROI['z'][0]) & (points[:,2] < ROI['z'][1])
    )
    return points[mask]

print("S1 프레임 로딩 중...")
files = sorted(glob.glob('pcd/S1/*.pcd'))
combined = o3d.geometry.PointCloud()
for f in files:
    pcd = o3d.io.read_point_cloud(f)
    pts = apply_roi(np.asarray(pcd.points))
    tmp = o3d.geometry.PointCloud()
    tmp.points = o3d.utility.Vector3dVector(pts)
    combined += tmp

combined = combined.voxel_down_sample(0.03)
pts_all = np.asarray(combined.points)

# 물통/부품 제거
no_obj_mask = ~(
    (pts_all[:,0] > -1.2) & (pts_all[:,0] < 0.5) &
    (pts_all[:,1] > 0.5) & (pts_all[:,1] < 3.3)
)
pts_all = pts_all[no_obj_mask]
print(f"물통/부품 제거 후: {len(pts_all):,}")

# 로컬 높이 계산
grid_res = 0.05
x_bins = np.arange(ROI['x'][0], ROI['x'][1], grid_res)
y_bins = np.arange(ROI['y'][0], ROI['y'][1], grid_res)
x_idx = np.digitize(pts_all[:,0], x_bins) - 1
y_idx = np.digitize(pts_all[:,1], y_bins) - 1

grid_min_z = {}
for i in range(len(pts_all)):
    key = (x_idx[i], y_idx[i])
    z = pts_all[i, 2]
    if key not in grid_min_z or z < grid_min_z[key]:
        grid_min_z[key] = z

local_relief = np.zeros(len(pts_all))
for i in range(len(pts_all)):
    key = (x_idx[i], y_idx[i])
    z = pts_all[i, 2]
    neighbor_mins = []
    for dx in range(-2, 3):
        for dy in range(-2, 3):
            nkey = (key[0]+dx, key[1]+dy)
            if nkey in grid_min_z:
                neighbor_mins.append(grid_min_z[nkey])
    if neighbor_mins:
        local_ground = np.percentile(neighbor_mins, 10)
        local_relief[i] = z - local_ground

# 두둑 상단만 (로컬 높이 4cm~25cm)
ridge_mask = (local_relief > 0.04) & (local_relief < 0.25)
ridge_pts = pts_all[ridge_mask]
ground_pts = pts_all[~ridge_mask]
print(f"두둑 상단 후보: {len(ridge_pts):,}")

# 두둑 정의 — x+y 범위 동시 적용
ridge_defs = [
    {'name': 'R1', 'y': (-1.85, -0.85), 'x': (2.41, 3.85), 'color': [1,0,0]},
    {'name': 'R2', 'y': (-0.50,  0.65), 'x': (2.42, 3.85), 'color': [0,0.8,0]},
    {'name': 'R4', 'y': ( 0.68,  1.15), 'x': (2.65, 3.85), 'color': [0,0,1]},
    {'name': 'R5', 'y': ( 1.18,  1.65), 'x': (2.42, 3.67), 'color': [1,0.8,0]},
    {'name': 'R6', 'y': ( 1.90,  2.50), 'x': (2.37, 3.85), 'color': [1,0,1]},
    {'name': 'R7', 'y': ( 2.65,  3.20), 'x': (2.23, 3.85), 'color': [0,1,1]},
]

colors = np.ones((len(ridge_pts), 3)) * 0.5
labels = np.full(len(ridge_pts), -1)

print("\n두둑별 포인트 수:")
for i, rd in enumerate(ridge_defs):
    y0, y1 = rd['y']
    x0, x1 = rd['x']
    mask = (
        (ridge_pts[:,1] >= y0) & (ridge_pts[:,1] < y1) &
        (ridge_pts[:,0] >= x0) & (ridge_pts[:,0] < x1)
    )
    n = mask.sum()
    print(f"  {rd['name']}: y={y0:.2f}~{y1:.2f} x={x0:.2f}~{x1:.2f}  포인트={n}")
    colors[mask] = rd['color']
    labels[mask] = i

# 시각화
ridge_pcd = o3d.geometry.PointCloud()
ridge_pcd.points = o3d.utility.Vector3dVector(ridge_pts)
ridge_pcd.colors = o3d.utility.Vector3dVector(colors)

ground_pcd = o3d.geometry.PointCloud()
ground_pcd.points = o3d.utility.Vector3dVector(ground_pts)
ground_pcd.paint_uniform_color([0.4, 0.3, 0.2])

coord = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0)
print("\n빨강=R1 / 초록=R2 / 파랑=R4 / 노랑=R5 / 핑크=R6 / 하늘=R7 / 회색=미분류 / 갈색=지면")
o3d.visualization.draw_geometries(
    [ground_pcd, ridge_pcd, coord],
    window_name='Ridge Detection v12',
    width=1400, height=900
)