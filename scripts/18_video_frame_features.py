from __future__ import annotations

# ==================================================
# 18. 영상의 각 Frame을 38-D Feature로 바꿉니다
# ==================================================
# 교재 26번 설명대로 작성한 코드입니다.
# (교재 본문에 실린 18번 코드는 17번 코드와 같은 내용이 들어가 있어서,
#  26번의 설명·실행 예시·출력 결과를 기준으로 작성했습니다.)
#
# 20번에서 이미지 한 장에 적용한 38-D Feature 계산을
# 영상의 Frame마다 반복합니다.
#
# MP4 열기
#   → Frame 한 장 읽기
#   → YOLO Pose 실행
#   → 17개 Keypoint 추출
#   → Confidence 계산
#   → Hip Center + Shoulder Width 정규화
#   → 34개 Normalized 좌표 + 팔꿈치·무릎 각도 4개
#   → 38-D Frame Feature
#   → Frame 번호 + 시간과 함께 CSV 저장
#   → 다음 Frame

# ==================================================
# ① 필요한 라이브러리와 공통 함수 가져오기
# ==================================================
# argparse : 실행 옵션 처리
# csv      : Frame별 Feature를 CSV로 저장
# sys      : Python 모듈 검색 경로 설정
# Path     : 파일 및 폴더 경로 처리
# cv2      : 영상 열기와 Frame 읽기
# numpy    : Confidence 평균과 개수 계산
# YOLO     : Frame에서 Keypoint 추출

import argparse
import csv
import sys
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO


# ==================================================
# ② 프로젝트 경로 설정 및 공통 모듈 가져오기
# ==================================================

ROOT = Path(__file__).resolve().parents[1]

sys.path.insert(
    0,
    str(ROOT),
)

from src.common import (
    load_config,
    resolve_device,
)

from src.feature_utils import (
    normalize_by_shoulder_width,
    selected_joint_angles,
)

from src.pose_utils import (
    COCO_KEYPOINT_NAMES,
)


# ==================================================
# ③ 실행 옵션 설정
# ==================================================
# --source     : 분석할 영상 경로 (기본: feature_demo.mp4)
# --kp-conf    : 낮은 Confidence 기준 (기본 0.50)
# --max-frames : 처리할 최대 Frame 수 (0이면 전체)

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--source",
        default=str(
            ROOT
            / "data"
            / "day03"
            / "videos"
            / "feature_demo.mp4"
        ),
    )

    parser.add_argument(
        "--kp-conf",
        type=float,
        default=0.50,
    )

    parser.add_argument(
        "--max-frames",
        type=int,
        default=150,
    )

    return parser.parse_args()


# ==================================================
# ④ make_feature_names() - 38개 Feature 이름
# ==================================================
# 16번 코드와 같은 순서입니다.
# 17개 관절 x, y (34개) + 관절각도 4개 = 38개

ANGLE_NAMES = [
    "left_elbow_angle",
    "right_elbow_angle",
    "left_knee_angle",
    "right_knee_angle",
]


def make_feature_names():
    names = []

    for keypoint_name in COCO_KEYPOINT_NAMES:
        names.append(f"{keypoint_name}_x")
        names.append(f"{keypoint_name}_y")

    names.extend(ANGLE_NAMES)

    return names


# ==================================================
# ⑤ empty_row() - Pose를 쓸 수 없는 Frame
# ==================================================
# Pose를 찾지 못했거나 정규화·38-D 계산이 불가능하면
# valid_pose = 0으로 기록하고 Feature는 비워 둡니다.
# Frame을 삭제하지 않으므로 시간 순서가 유지됩니다.

def empty_row(frame_id, time_sec, feature_names):
    row = {
        "frame_id": frame_id,
        "time_sec": round(time_sec, 4),
        "valid_pose": 0,
        "mean_keypoint_conf": "",
        "low_keypoint_count": "",
    }

    for name in feature_names:
        row[name] = ""

    return row


def main() -> None:

    # ==============================================
    # ⑥ 영상과 실행환경 확인
    # ==============================================

    args = parse_args()

    source = Path(args.source)

    if not source.exists():
        raise FileNotFoundError(source)

    config = load_config()

    device = resolve_device(config["device"])

    model = YOLO(config["pose_model"])

    capture = cv2.VideoCapture(str(source))

    if not capture.isOpened():
        raise RuntimeError(f"영상을 열 수 없습니다: {source}")

    fps = capture.get(cv2.CAP_PROP_FPS) or 30.0

    feature_names = make_feature_names()

    rows = []

    frame_id = 0

    # ==============================================
    # ⑦ Frame을 한 장씩 읽으며 반복
    # ==============================================

    while True:

        if args.max_frames > 0 and frame_id >= args.max_frames:
            break

        ok, frame = capture.read()

        if not ok:
            break

        frame_id += 1

        time_sec = (frame_id - 1) / fps

        # ------------------------------------------
        # YOLO Pose 실행
        # ------------------------------------------
        result = model.predict(
            source=frame,
            imgsz=config["image_size"],
            conf=config["person_confidence"],
            device=device,
            verbose=False,
        )[0]

        if (
            result.keypoints is None
            or result.keypoints.xy is None
            or len(result.keypoints.xy) == 0
        ):
            rows.append(empty_row(frame_id, time_sec, feature_names))
            continue

        # ------------------------------------------
        # 첫 번째 사람의 Keypoint와 Confidence
        # ------------------------------------------
        points = result.keypoints.xy[0].cpu().numpy()

        if result.keypoints.conf is not None:
            conf = result.keypoints.conf[0].cpu().numpy()
        else:
            conf = np.ones(17, dtype=np.float32)

        # ------------------------------------------
        # Hip Center + Shoulder Width 정규화
        # 계산할 수 없으면 valid_pose = 0
        # ------------------------------------------
        try:
            normalized, _, _ = normalize_by_shoulder_width(points)
        except ValueError:
            rows.append(empty_row(frame_id, time_sec, feature_names))
            continue

        angles = selected_joint_angles(points)

        # ------------------------------------------
        # 38-D Feature 만들기
        # ------------------------------------------
        feature_values = []

        for point in normalized:
            feature_values.extend([float(point[0]), float(point[1])])

        feature_values.extend([angles[name] for name in ANGLE_NAMES])

        # ------------------------------------------
        # CSV 1행 만들기
        # Confidence는 38-D에 넣지 않고 품질 정보로 따로 기록
        # ------------------------------------------
        row = {
            "frame_id": frame_id,
            "time_sec": round(time_sec, 4),
            "valid_pose": 1,
            "mean_keypoint_conf": round(float(np.mean(conf)), 4),
            "low_keypoint_count": int(np.sum(conf < args.kp_conf)),
        }

        for name, value in zip(feature_names, feature_values):
            row[name] = round(float(value), 6)

        rows.append(row)

    capture.release()

    # ==============================================
    # ⑧ CSV 저장
    # ==============================================

    output_path = (
        ROOT
        / "reports"
        / "day03"
        / "tables"
        / f"{source.stem}_frame_features.csv"
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = list(empty_row(0, 0.0, feature_names).keys())

    with output_path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    # ==============================================
    # ⑨ 결과 출력
    # ==============================================

    valid_count = sum(1 for row in rows if row["valid_pose"] == 1)

    print(f"Total frames: {len(rows)}")
    print(f"Valid pose frames: {valid_count}")
    print(f"Feature count per frame: {len(feature_names)}")
    print(f"Saved: {output_path}")


# ==================================================
# 프로그램 실행 시작
# ==================================================

if __name__ == "__main__":
    main()
