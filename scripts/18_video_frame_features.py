from __future__ import annotations

# ==================================================
# 18. 영상의 각 Frame을 38-D Feature로 바꿉니다
#     + 3팀(무인 상점 이상행동) 추가 Feature
# ==================================================
# 20번에서 이미지 한 장에 적용한 38-D Feature 계산을
# 영상의 Frame마다 반복합니다.
#
# 3팀은 여기에 다음 정보를 함께 저장합니다.
# - 사람 BBox 원본 좌표 (화면 어디에 있었는지 보존)
# - torso_tilt_deg / hip_ankle_dy / bbox_aspect_ratio
# - lie_down (17번 드러눕기 규칙 결과)

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

from src.team_features import (
    TEAM_FEATURE_NAMES,
    lie_down_rule,
    team_features,
)


# ==================================================
# ③ 실행 옵션 설정
# ==================================================
# --source     : 분석할 영상 경로
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
# 사람을 찾지 못했거나 정규화를 할 수 없으면
# valid_pose = 0으로 기록하고 Feature는 비워 둡니다.

def empty_row(
    frame_id,
    time_sec,
    feature_names,
):
    row = {
        "frame_id": frame_id,
        "time_sec": round(time_sec, 4),
        "valid_pose": 0,
        "mean_keypoint_conf": "",
        "low_keypoint_count": "",
        "bbox_x1": "",
        "bbox_y1": "",
        "bbox_x2": "",
        "bbox_y2": "",
        "shoulder_width": "",
    }

    for name in feature_names:
        row[name] = ""

    for name in TEAM_FEATURE_NAMES:
        row[name] = ""

    row["lie_down"] = ""

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

    device = resolve_device(
        config["device"]
    )

    model = YOLO(
        config["pose_model"]
    )

    capture = cv2.VideoCapture(
        str(source)
    )

    if not capture.isOpened():
        raise RuntimeError(
            f"영상을 열 수 없습니다: {source}"
        )

    fps = capture.get(cv2.CAP_PROP_FPS) or 30.0

    feature_names = make_feature_names()

    rows = []

    frame_id = 0

    # ==============================================
    # ⑦ Frame을 한 장씩 읽으며 반복
    # ==============================================

    while True:

        if (
            args.max_frames > 0
            and frame_id >= args.max_frames
        ):
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
            rows.append(
                empty_row(frame_id, time_sec, feature_names)
            )
            continue

        # ------------------------------------------
        # 첫 번째 사람의 Keypoint, Confidence, BBox
        # ------------------------------------------
        points = (
            result.keypoints.xy[0]
            .cpu()
            .numpy()
        )

        if result.keypoints.conf is not None:
            conf = (
                result.keypoints.conf[0]
                .cpu()
                .numpy()
            )
        else:
            conf = np.ones(17, dtype=np.float32)

        box = (
            result.boxes.xyxy[0]
            .cpu()
            .numpy()
        )

        # ------------------------------------------
        # 정규화 (Hip Center + Shoulder Width)
        # 계산할 수 없으면 valid_pose = 0
        # ------------------------------------------
        try:
            normalized, _, scale = normalize_by_shoulder_width(
                points
            )
        except ValueError:
            rows.append(
                empty_row(frame_id, time_sec, feature_names)
            )
            continue

        angles = selected_joint_angles(points)

        # ------------------------------------------
        # 38-D Feature 만들기
        # ------------------------------------------
        feature_values = []

        for point in normalized:
            feature_values.extend(
                [float(point[0]), float(point[1])]
            )

        feature_values.extend(
            [angles[name] for name in ANGLE_NAMES]
        )

        # ------------------------------------------
        # 3팀 추가 Feature + 드러눕기 규칙
        # ------------------------------------------
        team = team_features(points, conf, box)

        _, lie_down = lie_down_rule(team)

        # ------------------------------------------
        # CSV 1행 만들기
        # ------------------------------------------
        row = {
            "frame_id": frame_id,
            "time_sec": round(time_sec, 4),
            "valid_pose": 1,
            "mean_keypoint_conf": round(float(np.mean(conf)), 4),
            "low_keypoint_count": int(np.sum(conf < args.kp_conf)),
            "bbox_x1": round(float(box[0]), 1),
            "bbox_y1": round(float(box[1]), 1),
            "bbox_x2": round(float(box[2]), 1),
            "bbox_y2": round(float(box[3]), 1),
            "shoulder_width": round(float(scale), 4),
        }

        for name, value in zip(feature_names, feature_values):
            row[name] = round(float(value), 6)

        for name in TEAM_FEATURE_NAMES:
            row[name] = round(float(team[name]), 4)

        row["lie_down"] = int(lie_down)

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

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fieldnames = list(
        empty_row(0, 0.0, feature_names).keys()
    )

    with output_path.open(
        "w",
        newline="",
        encoding="utf-8-sig",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
        )
        writer.writeheader()
        writer.writerows(rows)

    # ==============================================
    # ⑨ 결과 출력
    # ==============================================

    valid_rows = [
        row for row in rows if row["valid_pose"] == 1
    ]

    lie_rows = [
        row for row in valid_rows if row["lie_down"] == 1
    ]

    print(f"Source: {source.name}")
    print(f"Total frames: {len(rows)}")
    print(f"Valid pose frames: {len(valid_rows)}")
    print(f"Feature count per frame: {len(feature_names)}")
    print(f"Team feature count: {len(TEAM_FEATURE_NAMES)}")

    if lie_rows:
        print(
            f"Lie-down frames: {len(lie_rows)} "
            f"({lie_rows[0]['time_sec']:.2f}s ~ "
            f"{lie_rows[-1]['time_sec']:.2f}s)"
        )
    else:
        print("Lie-down frames: 0")

    print(f"Saved: {output_path}")


# ==================================================
# 프로그램 실행 시작
# ==================================================

if __name__ == "__main__":
    main()
