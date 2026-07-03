#!/usr/bin/env python3
import numpy as np
import glob
import os
from rosbag2_py import SequentialReader, StorageOptions, ConverterOptions

def get_bag_timestamps(bag_path):
    """bag 파일의 시작 타임스탬프 반환 (초 단위)"""
    reader = SequentialReader()
    storage_options = StorageOptions(uri=bag_path, storage_id='sqlite3')
    converter_options = ConverterOptions(
        input_serialization_format='cdr',
        output_serialization_format='cdr'
    )
    reader.open(storage_options, converter_options)
    # 첫 번째 메시지 타임스탬프
    if reader.has_next():
        topic, data, timestamp = reader.read_next()
        return timestamp / 1e9  # 나노초 → 초
    return None

# 각 세션의 클래스 구간 정의
# (시작초, 끝초, 클래스명)
SESSION_LABELS = {
    'S2': {
        'bag': 'S2_idle_standing',
        'segments': [
            (0,      61.37,  'idle_front'),
            (61.37, 121.63,  'idle_side'),
            (121.63, 999,    'idle_back'),
        ]
    },
    'S3': {
        'bag': 'S3_walking',
        'segments': [
            (0,      64.02,  'walking_slow'),
            (64.02, 121.97,  'walking_normal'),
            (121.97, 999,    'walking_fast'),
        ]
    },
    'S4': {
        'bag': 'S4_squatting_static',
        'segments': [
            (0,      79.23,  'squat_far'),
            (79.23, 119.08,  'squat_mid'),
            (119.08, 999,    'squat_far2'),
        ]
    },
    'S5': {
        'bag': 'S5_squatting_moving',
        'segments': [
            (0, 999, 'squatting_moving'),
        ]
    },
    'S6': {
        'bag': 'S6_overhead_motion',
        'segments': [
            (0, 999, 'overhead_motion'),
        ]
    },
    'S7': {
        'bag': 'S7_rhythmic_ground',
        'segments': [
            (0, 999, 'rhythmic_ground'),
        ]
    },
}

# PCD 파일별 라벨 생성
print("PCD 파일별 라벨 생성 중...")
all_labels = {}

for session, info in SESSION_LABELS.items():
    bag_path = info['bag']
    pcd_folder = f'pcd/{session}'

    if not os.path.exists(pcd_folder):
        print(f"  {session}: PCD 폴더 없음, 스킵")
        continue

    # bag 시작 타임스탬프
    bag_start = get_bag_timestamps(bag_path)
    if bag_start is None:
        print(f"  {session}: bag 읽기 실패")
        continue

    print(f"\n{session} (bag 시작: {bag_start:.2f}s)")

    # PCD 파일 목록 (순서대로)
    pcd_files = sorted(glob.glob(f'{pcd_folder}/*.pcd'))

    session_labels = []
    for i, pcd_file in enumerate(pcd_files):
        # PCD 파일 인덱스 기반 시간 추정 (10Hz)
        elapsed = i / 10.0  # 초

        # 해당 구간의 클래스 찾기
        label = 'unknown'
        for seg_start, seg_end, seg_label in info['segments']:
            if seg_start <= elapsed < seg_end:
                label = seg_label
                break

        session_labels.append({
            'file': pcd_file,
            'elapsed': elapsed,
            'label': label
        })

    # 클래스별 통계
    from collections import Counter
    label_counts = Counter(item['label'] for item in session_labels)
    for lbl, cnt in label_counts.items():
        print(f"  {lbl}: {cnt}프레임")

    all_labels[session] = session_labels

# numpy 파일로 저장
print("\n라벨 파일 저장 중...")
for session, labels in all_labels.items():
    files = [item['file'] for item in labels]
    lbls  = [item['label'] for item in labels]
    elapsed = [item['elapsed'] for item in labels]

    np.save(f'labels_{session}_files.npy', np.array(files))
    np.save(f'labels_{session}_labels.npy', np.array(lbls))
    np.save(f'labels_{session}_elapsed.npy', np.array(elapsed))
    print(f"  {session}: {len(files)}개 프레임 저장")

print("\n완료. 다음 단계: feature 추출")