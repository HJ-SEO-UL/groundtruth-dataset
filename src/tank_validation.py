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

# S1 100프레임 합치기
files = sorted(glob.glob('pcd/S1/*.pcd'))[:100]
combined = o3d.geometry.PointCloud()
for f in files:
    pcd = o3d.io.read_point_cloud(f)
    pts = apply_roi(np.asarray(pcd.points))
    tmp = o3d.geometry.PointCloud()
    tmp.points = o3d.utility.Vector3dVector(pts)
    combined += tmp

pts_all = np.asarray(combined.points)

# 물통은 센서 서쪽(y 음수 방향)에 위치
# 배경 모델 z 범위 -0.33~0.94 기준으로
# 물통은 x 음수 방향, y 음수 방향에 위치
# 물통 영역만 잘라서 확인
tank_mask = (
    (pts_all[:,0] > -1.2) & (pts_all[:,0] < 0.5) &
    (pts_all[:,1] > -2.0) & (pts_all[:,1] < 0.0)
)
tank_pts = pts_all[tank_mask]
print(f"물통 후보 포인트 수: {len(tank_pts)}")

if len(tank_pts) > 0:
    x_size = tank_pts[:,0].max() - tank_pts[:,0].min()
    y_size = tank_pts[:,1].max() - tank_pts[:,1].min()
    z_size = tank_pts[:,2].max() - tank_pts[:,2].min()

    print(f"\n물통 감지 크기:")
    print(f"x 방향 폭: {x_size*100:.1f}cm")
    print(f"y 방향 폭: {y_size*100:.1f}cm")
    print(f"z 방향 높이: {z_size*100:.1f}cm")

    # 지름 추정 (x, y 중 더 큰 값)
    detected_diameter = max(x_size, y_size) * 100
    actual_diameter = 100.0
    error = abs(detected_diameter - actual_diameter)
    error_pct = error / actual_diameter * 100

    print(f"\n실제 지름: {actual_diameter:.1f}cm")
    print(f"감지 지름: {detected_diameter:.1f}cm")
    print(f"오차: {error:.1f}cm ({error_pct:.1f}%)")

    # 시각화
    tank_pcd = o3d.geometry.PointCloud()
    tank_pcd.points = o3d.utility.Vector3dVector(tank_pts)
    tank_pcd.paint_uniform_color([1.0, 0.0, 0.0])

    full_pcd = o3d.geometry.PointCloud()
    full_pcd.points = o3d.utility.Vector3dVector(pts_all)
    z = pts_all[:,2]
    z_norm = (z - z.min()) / (z.max() - z.min() + 1e-6)
    colors = np.stack([z_norm, z_norm * 0.6, 1 - z_norm], axis=1)
    full_pcd.colors = o3d.utility.Vector3dVector(colors)

    coord = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0)
    print("\n빨간색 = 물통 영역")
    o3d.visualization.draw_geometries(
        [full_pcd, tank_pcd, coord],
        window_name='물통 스케일 검증',
        width=1400, height=900
    )
else:
    print("물통 포인트를 찾지 못했어. 범위 조정 필요.")
    # 물통 스케일 검증 최종 결과
TANK_ROI = {
    'x': (-1.2, 0.0),
    'y': (1.2, 2.3),
    'z': (-0.3, 1.1)
}
# 감지 지름: 99.9cm / 실제: 100.0cm / 오차: 0.1cm (0.1%)