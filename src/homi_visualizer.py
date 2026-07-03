#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2, PointField
from std_msgs.msg import Header
from visualization_msgs.msg import Marker
from scipy.spatial import KDTree
import open3d as o3d
import numpy as np
import glob
import struct
from collections import deque

ROI = {
    'x': (-1.2, 3.85),
    'y': (-2.0, 3.3),
    'z': (-2.0, 1.5)
}

RIDGE_DEFS = [
    {'name': 'R1', 'y': (-1.85,-0.85), 'x': (2.41,3.85), 'color': (1.0, 0.2, 0.2)},
    {'name': 'R2', 'y': (-0.50, 0.65), 'x': (2.42,3.85), 'color': (0.2, 1.0, 0.3)},
    {'name': 'R3', 'y': ( 0.68, 1.15), 'x': (2.65,3.85), 'color': (0.2, 0.5, 1.0)},
    {'name': 'R4', 'y': ( 1.18, 1.65), 'x': (2.42,3.67), 'color': (1.0, 0.8, 0.0)},
    {'name': 'R5', 'y': ( 1.90, 2.50), 'x': (2.37,3.85), 'color': (0.8, 0.2, 1.0)},
    {'name': 'R6', 'y': ( 2.65, 3.20), 'x': (2.23,3.85), 'color': (0.0, 0.9, 0.8)},
]

CLASS_COLORS = {
    'idle_standing':    (0.8, 0.8, 0.8),
    'walking':          (0.2, 0.6, 1.0),
    'squatting_static': (1.0, 0.5, 0.0),
    'squatting_moving': (1.0, 0.2, 0.2),
    'overhead_motion':  (0.0, 0.9, 0.7),
    'rhythmic_ground':  (0.8, 0.2, 1.0),
    'unknown':          (1.0, 1.0, 1.0),
}

CLASS_LABELS = {
    'idle_standing':    'IdleStanding',
    'walking':          'Walking',
    'squatting_static': 'Squatting',
    'squatting_moving': 'SquattingMoving',
    'overhead_motion':  'Watering',
    'rhythmic_ground':  'GroundToolUse(Ho-Mi)',
    'unknown':          '...',
}

def make_pointcloud2(points, colors, frame_id='map'):
    header = Header()
    header.frame_id = frame_id
    fields = [
        PointField(name='x',   offset=0,  datatype=PointField.FLOAT32, count=1),
        PointField(name='y',   offset=4,  datatype=PointField.FLOAT32, count=1),
        PointField(name='z',   offset=8,  datatype=PointField.FLOAT32, count=1),
        PointField(name='rgb', offset=12, datatype=PointField.FLOAT32, count=1),
    ]
    data = []
    for i in range(len(points)):
        x = float(points[i,0])
        y = float(points[i,1])
        z = float(points[i,2])
        r = int(np.clip(colors[i,0], 0, 1) * 255)
        g = int(np.clip(colors[i,1], 0, 1) * 255)
        b = int(np.clip(colors[i,2], 0, 1) * 255)
        rgb = struct.unpack('f', struct.pack('I', (r<<16)|(g<<8)|b))[0]
        data.append(struct.pack('ffff', x, y, z, rgb))
    msg = PointCloud2()
    msg.header       = header
    msg.height       = 1
    msg.width        = len(points)
    msg.fields       = fields
    msg.is_bigendian = False
    msg.point_step   = 16
    msg.row_step     = 16 * len(points)
    msg.data         = b''.join(data)
    msg.is_dense     = True
    return msg

def make_label_marker(text, frame_id='map'):
    """행동 이름 — 흰색, 고정 위치"""
    marker = Marker()
    marker.header.frame_id    = frame_id
    marker.id                 = 0
    marker.type               = Marker.TEXT_VIEW_FACING
    marker.action             = Marker.ADD
    marker.pose.position.x    = 2.0
    marker.pose.position.y    = -2.2
    marker.pose.position.z    = 1.7
    marker.pose.orientation.w = 1.0
    marker.scale.z            = 0.09
    marker.color.r            = 1.0
    marker.color.g            = 1.0
    marker.color.b            = 1.0
    marker.color.a            = 1.0
    marker.text               = text
    return marker

def make_conf_marker(text, frame_id='map'):
    """신뢰도 — 어두운 회색, 이름 바로 아래 고정 위치"""
    marker = Marker()
    marker.header.frame_id    = frame_id
    marker.id                 = 1
    marker.type               = Marker.TEXT_VIEW_FACING
    marker.action             = Marker.ADD
    marker.pose.position.x    = 2.0
    marker.pose.position.y    = -2.2
    marker.pose.position.z    = 1.55
    marker.pose.orientation.w = 1.0
    marker.scale.z            = 0.07
    marker.color.r            = 0.5
    marker.color.g            = 0.5
    marker.color.b            = 0.5
    marker.color.a            = 1.0
    marker.text               = text
    return marker

class HomiVisualizer(Node):

    def __init__(self):
        super().__init__('homi_visualizer')

        self.ridge_pub  = self.create_publisher(PointCloud2, '/homi/ridges',     10)
        self.person_pub = self.create_publisher(PointCloud2, '/homi/person',     10)
        self.bg_pub     = self.create_publisher(PointCloud2, '/homi/background', 10)
        self.label_pub  = self.create_publisher(Marker,      '/homi/label',      10)
        self.conf_pub   = self.create_publisher(Marker,      '/homi/confidence', 10)

        self.get_logger().info("배경 모델 로드 중...")
        bg_pts = np.load('background_model.npy')
        self.bg_tree = KDTree(bg_pts)
        self.get_logger().info(f"배경 포인트: {len(bg_pts)}")

        self.get_logger().info("두둑 포인트클라우드 구축 중...")
        self.ridge_msg, self.bg_msg = self.build_ridge_and_bg_cloud()

        self.get_logger().info("예측 데이터 로드 중...")
        predictions   = np.load('s8_predictions.npy')
        frame_indices = np.load('s8_frame_indices.npy')
        confidences   = np.load('s8_confidences.npy')
        self.pred_dict = {int(frame_indices[i]): predictions[i] for i in range(len(predictions))}
        self.conf_dict = {int(frame_indices[i]): confidences[i] for i in range(len(confidences))}

        self.files_s8      = sorted(glob.glob('pcd/S8/*.pcd'))
        self.frame_idx     = 0
        self.loop          = True
        self.ACCUMULATE    = 15
        self.person_buffer = deque(maxlen=self.ACCUMULATE)

        self.get_logger().info(f"준비 완료. {len(self.files_s8)} 프레임 재생 시작...")
        self.timer = self.create_timer(0.1, self.publish_frame)

    def apply_roi(self, pts):
        mask = (
            (pts[:,0] > ROI['x'][0]) & (pts[:,0] < ROI['x'][1]) &
            (pts[:,1] > ROI['y'][0]) & (pts[:,1] < ROI['y'][1]) &
            (pts[:,2] > ROI['z'][0]) & (pts[:,2] < ROI['z'][1])
        )
        return pts[mask]

    def build_ridge_and_bg_cloud(self):
        files_s1 = sorted(glob.glob('pcd/S1/*.pcd'))
        combined = o3d.geometry.PointCloud()
        for f in files_s1:
            pcd = o3d.io.read_point_cloud(f)
            pts = self.apply_roi(np.asarray(pcd.points))
            tmp = o3d.geometry.PointCloud()
            tmp.points = o3d.utility.Vector3dVector(pts)
            combined += tmp

        combined = combined.voxel_down_sample(0.03)
        pts_all  = np.asarray(combined.points)

        ridge_colors = np.ones((len(pts_all), 3)) * 0.15
        is_ridge     = np.zeros(len(pts_all), dtype=bool)
        for rd in RIDGE_DEFS:
            mask = (
                (pts_all[:,0] >= rd['x'][0]) & (pts_all[:,0] < rd['x'][1]) &
                (pts_all[:,1] >= rd['y'][0]) & (pts_all[:,1] < rd['y'][1])
            )
            ridge_colors[mask] = rd['color']
            is_ridge[mask]     = True

        ridge_pts = pts_all[is_ridge]
        ridge_col = ridge_colors[is_ridge]
        ridge_msg = make_pointcloud2(ridge_pts, ridge_col)

        bg_pts_vis = pts_all[~is_ridge]
        bg_col     = np.ones((len(bg_pts_vis), 3)) * 0.15
        bg_msg     = make_pointcloud2(bg_pts_vis, bg_col)

        return ridge_msg, bg_msg

    def extract_person_pts(self, pcd_file):
        pcd = o3d.io.read_point_cloud(pcd_file)
        pts = self.apply_roi(np.asarray(pcd.points))
        if len(pts) < 3:
            return None
        dists, _ = self.bg_tree.query(pts, k=1)
        person_pts = pts[dists > 0.05]
        return person_pts if len(person_pts) >= 3 else None

    def publish_frame(self):
        if self.frame_idx >= len(self.files_s8):
            if self.loop:
                self.frame_idx = 0
                self.person_buffer.clear()
                self.get_logger().info("루프 재시작...")
            else:
                self.timer.cancel()
                return

        now = self.get_clock().now().to_msg()
        t   = self.frame_idx / 10.0

        # 두둑 + 배경 퍼블리시
        self.ridge_msg.header.stamp = now
        self.ridge_pub.publish(self.ridge_msg)
        self.bg_msg.header.stamp = now
        self.bg_pub.publish(self.bg_msg)

        # 사람 포인트 추출 + 버퍼 누적
        person_pts = self.extract_person_pts(self.files_s8[self.frame_idx])
        pred = self.pred_dict.get(self.frame_idx, 'unknown')
        conf = self.conf_dict.get(self.frame_idx, 0.0)

        if person_pts is not None:
            self.person_buffer.append(person_pts)

        if len(self.person_buffer) > 0:
            accumulated = np.vstack(list(self.person_buffer))
            tmp_pcd = o3d.geometry.PointCloud()
            tmp_pcd.points = o3d.utility.Vector3dVector(accumulated)
            tmp_pcd = tmp_pcd.voxel_down_sample(0.02)
            accumulated = np.asarray(tmp_pcd.points)

            color  = CLASS_COLORS.get(pred, (1.0, 1.0, 1.0))
            colors = np.tile(color, (len(accumulated), 1))
            person_msg = make_pointcloud2(accumulated, colors)
            person_msg.header.stamp = now
            self.person_pub.publish(person_msg)

            if self.frame_idx % 100 == 0:
                self.get_logger().info(
                    f"t={t:.1f}s  frame={self.frame_idx}"
                    f"  accumulated={len(accumulated)}pts"
                    f"  pred={pred}  conf={conf:.2f}"
                )

        # 레이블 퍼블리시 — 이름(흰색)과 신뢰도(회색) 분리
        label_str = CLASS_LABELS.get(pred, '...')
        conf_str  = f"{conf:.0%}" if pred != 'unknown' else ''

        label_marker = make_label_marker(label_str)
        label_marker.header.stamp = now
        self.label_pub.publish(label_marker)

        conf_marker = make_conf_marker(conf_str)
        conf_marker.header.stamp = now
        self.conf_pub.publish(conf_marker)

        self.frame_idx += 1


def main():
    rclpy.init()
    node = HomiVisualizer()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()