#!/usr/bin/env python3
import open3d as o3d
import numpy as np
import glob
import os
import pickle

# 확정된 ROI 값
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

def build_background_model(pcd_folder, num_frames=30):
    print(f"배경 모델 구축 중 (S1 첫 {num_frames}프레임 사용)...")
    files = sorted(glob.glob(os.path.join(pcd_folder, '*.pcd')))[:num_frames]
    all_points = []

    for f in files:
        pcd = o3d.io.read_point_cloud(f)
        pts = np.asarray(pcd.points)
        pts_roi = apply_roi(pts)
        if len(pts_roi) > 0:
            all_points.append(pts_roi)

    background = np.vstack(all_points)
    print(f"배경 모델 완료: {len(background):,} 포인트")
    print(f"x 범위: {background[:,0].min():.2f} ~ {background[:,0].max():.2f}")
    print(f"y 범위: {background[:,1].min():.2f} ~ {background[:,1].max():.2f}")
    print(f"z 범위: {background[:,2].min():.2f} ~ {background[:,2].max():.2f}")
    return background

def save_background_model(background, path='background_model.npy'):
    np.save(path, background)
    print(f"배경 모델 저장 완료: {path}")

def visualize_background(background):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(background)

    z = background[:,2]
    z_norm = (z - z.min()) / (z.max() - z.min() + 1e-6)
    colors = np.stack([z_norm, z_norm * 0.6, 1 - z_norm], axis=1)
    pcd.colors = o3d.utility.Vector3dVector(colors)

    coord = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0)
    print("배경 모델 시각화 중...")
    o3d.visualization.draw_geometries(
        [pcd, coord],
        window_name='Background Model (S1)',
        width=1400, height=900
    )

if __name__ == "__main__":
    background = build_background_model('pcd/S1', num_frames=30)
    save_background_model(background)
    visualize_background(background)