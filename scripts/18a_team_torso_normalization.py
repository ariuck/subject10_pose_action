from __future__ import annotations

# ==================================================
# 3팀 추가 실습 — 어깨 폭 정규화 vs 상체 길이 정규화
# ==================================================
# Mini Challenge 32번에서 측면 카메라로 찍은
# 누운 자세(Near / Far)를 어깨 폭으로 정규화하자
# 오히려 차이가 커졌습니다.
#
# 옆에서 보면 두 어깨가 겹쳐 어깨 폭이 몇 px밖에 안 되기 때문입니다.
#
# 이 코드는 같은 두 이미지를
#   1) Pixel 좌표
#   2) Hip Center + Shoulder Width (교재 방식)
#   3) Hip Center + Torso Length (어깨 중심-골반 중심 거리)
# 세 가지로 비교합니다.

# ==================================================
# ① 필요한 라이브러리와 공통 함수 가져오기
# ==================================================

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
from ultralytics import YOLO

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
    center_keypoints,
    normalize_by_shoulder_width,
)

from src.pose_utils import (
    COCO_KEYPOINT_NAMES,
)

from src.team_features import (
    torso_length,
)


# ==================================================
# ② 실행 옵션
# ==================================================

def parse_args():
    parser = argparse.ArgumentParser()

    images = ROOT / "data" / "day03" / "challenge" / "images"

    parser.add_argument(
        "--a",
        default=str(images / "target_near.jpg"),
    )

    parser.add_argument(
        "--b",
        default=str(images / "target_far.jpg"),
    )

    parser.add_argument(
        "--output",
        default="team_torso_norm_target_near_far.csv",
    )

    return parser.parse_args()


# ==================================================
# ③ extract_first_person() - 첫 번째 사람 Keypoint
# ==================================================

def extract_first_person(
    model,
    source,
    config,
    device,
):
    result = model.predict(
        source=str(source),
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
        raise RuntimeError(
            f"Pose를 찾지 못했습니다: {source}"
        )

    return (
        result.keypoints.xy[0]
        .cpu()
        .numpy()
    )


# ==================================================
# ④ normalize_by_torso_length()
# ==================================================
# 교재 정규화와 같은 순서로
# 1. 골반 중심을 원점으로 이동하고
# 2. 어깨 폭 대신 상체 길이로 나눕니다.

def normalize_by_torso_length(points):
    centered, center = center_keypoints(points)

    scale = torso_length(points)

    if scale < 1e-8:
        raise ValueError(
            "상체 길이를 계산할 수 없습니다."
        )

    return centered / scale, center, scale


# ==================================================
# ⑤ per_joint_distance() - 관절별 거리
# ==================================================

def per_joint_distance(a, b):
    return np.linalg.norm(
        np.asarray(a) - np.asarray(b),
        axis=1,
    )


def main() -> None:
    args = parse_args()

    source_a = Path(args.a)
    source_b = Path(args.b)

    for source in (source_a, source_b):
        if not source.exists():
            raise FileNotFoundError(source)

    config = load_config()

    device = resolve_device(
        config["device"]
    )

    model = YOLO(
        config["pose_model"]
    )

    # ==============================================
    # ⑥ 두 이미지에서 Keypoint 추출
    # ==============================================

    pixel_a = extract_first_person(model, source_a, config, device)
    pixel_b = extract_first_person(model, source_b, config, device)

    # ==============================================
    # ⑦ 세 가지 좌표로 관절별 거리 계산
    # ==============================================

    shoulder_a, _, sw_a = normalize_by_shoulder_width(pixel_a)
    shoulder_b, _, sw_b = normalize_by_shoulder_width(pixel_b)

    torso_a, _, tl_a = normalize_by_torso_length(pixel_a)
    torso_b, _, tl_b = normalize_by_torso_length(pixel_b)

    pixel_distance = per_joint_distance(pixel_a, pixel_b)
    shoulder_distance = per_joint_distance(shoulder_a, shoulder_b)
    torso_distance = per_joint_distance(torso_a, torso_b)

    # ==============================================
    # ⑧ CSV 저장
    # ==============================================

    output_path = (
        ROOT
        / "reports"
        / "day03"
        / "tables"
        / args.output
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with output_path.open(
        "w",
        newline="",
        encoding="utf-8-sig",
    ) as file:
        writer = csv.writer(file)

        writer.writerow(
            [
                "keypoint",
                "pixel_distance",
                "shoulder_width_norm_distance",
                "torso_length_norm_distance",
            ]
        )

        for index, name in enumerate(COCO_KEYPOINT_NAMES):
            writer.writerow(
                [
                    name,
                    round(float(pixel_distance[index]), 4),
                    round(float(shoulder_distance[index]), 4),
                    round(float(torso_distance[index]), 4),
                ]
            )

    # ==============================================
    # ⑨ 결과 출력
    # ==============================================

    print(f"A: {source_a.name}")
    print(f"B: {source_b.name}")
    print()
    print(f"A shoulder width : {sw_a:7.2f}px   torso length : {tl_a:7.2f}px")
    print(f"B shoulder width : {sw_b:7.2f}px   torso length : {tl_b:7.2f}px")
    print()
    print("Mean joint distance")
    print(f"Pixel space            : {pixel_distance.mean():.4f}")
    print(f"Shoulder-width norm    : {shoulder_distance.mean():.4f}")
    print(f"Torso-length norm      : {torso_distance.mean():.4f}")
    print()
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()
