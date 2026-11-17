from __future__ import annotations

# ==================================================
# 3팀 — 무인 상점 이상행동 인식
# 17. 드러눕기(Lie Down) 상대 위치 Feature 규칙 확인
# ==================================================
# 교재 24번(Arms Up 규칙)과 같은 흐름을
# 우리 팀 동작인 "드러눕기"에 맞게 바꾼 코드입니다.
#
# 완성된 AI 분류 모델이 아니라
# "상체가 수평이고, 다리도 바닥과 나란한가?"를
# 간단한 규칙으로 확인하는 연습 코드입니다.

# ==================================================
# ① 필요한 라이브러리와 공통 함수 가져오기
# ==================================================
# argparse : 터미널에서 입력한 실행 옵션 처리
# sys      : Python 모듈 검색 경로 설정
# Path     : 이미지 파일 및 프로젝트 경로 관리
# numpy    : 관절 Confidence 기본값 생성
# YOLO     : 이미지에서 사람의 관절 위치 추정

import argparse
import sys
from pathlib import Path

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
    LEFT_HIP,
    LEFT_SHOULDER,
    RIGHT_HIP,
    RIGHT_SHOULDER,
    normalize_by_shoulder_width,
)

from src.team_features import (
    LIE_ANKLE_DY,
    LIE_TILT_DEG,
    lie_down_rule,
    team_features,
    torso_length,
    visible_ankle,
)


# ==================================================
# ③ 터미널 실행 옵션 설정
# ==================================================
# --source  : 분석할 이미지 파일 경로
# --kp-conf : 관절 신뢰도 최소 기준 (기본 0.50)

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--source",
        default=str(
            ROOT
            / "data"
            / "day03"
            / "challenge"
            / "images"
            / "target_near.jpg"
        ),
    )

    parser.add_argument(
        "--kp-conf",
        type=float,
        default=0.50,
    )

    return parser.parse_args()


def main() -> None:

    # ==============================================
    # ④ 이미지와 실행환경 확인
    # ==============================================

    args = parse_args()

    source = Path(args.source)

    if not source.exists():
        raise FileNotFoundError(source)

    config = load_config()

    device = resolve_device(
        config["device"]
    )

    # ==============================================
    # ⑤ YOLO Pose 모델 실행
    # ==============================================

    model = YOLO(
        config["pose_model"]
    )

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
            "Pose를 찾지 못했습니다."
        )

    # ==============================================
    # ⑥ 관절 좌표, Confidence, 사람 BBox 추출
    # ==============================================
    # 드러눕기는 사람 박스 모양(가로로 긴지)도 보기 때문에
    # 교재 코드와 달리 BBox도 함께 가져옵니다.

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
        conf = np.ones(
            17,
            dtype=np.float32,
        )

    box = (
        result.boxes.xyxy[0]
        .cpu()
        .numpy()
    )

    # ==============================================
    # ⑦ 필요한 관절의 신뢰도 검사
    # ==============================================
    # 상체 기울기 : 양쪽 어깨 + 양쪽 골반 (4개 모두 필요)
    # 다리 높이   : 잘 보이는 쪽 발목 1개
    #
    # 옆에서 찍으면 반대쪽 발목은 가려지기 쉬워서
    # 발목은 Confidence가 높은 쪽 하나만 사용합니다.

    torso_ids = [
        LEFT_SHOULDER,
        RIGHT_SHOULDER,
        LEFT_HIP,
        RIGHT_HIP,
    ]

    min_torso_conf = float(
        np.min(conf[torso_ids])
    )

    ankle_side, _, ankle_conf = visible_ankle(
        points,
        conf,
    )

    print(f"Source: {source.name}")
    print(f"Min required confidence (shoulder/hip): {min_torso_conf:.3f}")
    print(f"Visible ankle: {ankle_side} ({ankle_conf:.3f})")

    if min_torso_conf < args.kp_conf:
        print(
            "SKIP: 어깨·골반 Confidence가 낮아 "
            "상체 기울기를 믿을 수 없습니다."
        )
        return

    if ankle_conf < args.kp_conf:
        print(
            "SKIP: 발목 Confidence가 낮아 "
            "다리 높이를 믿을 수 없습니다."
        )
        return

    # ==============================================
    # ⑧ 어깨 너비 기준 좌표 정규화 (교재 방식)
    # ==============================================
    # 교재와 같은 정규화를 적용하고,
    # 어깨-골반 높이 차이를 38-D 좌표로도 확인합니다.
    #
    # 옆에서 본 누운 자세는 두 어깨가 겹쳐
    # 어깨 폭이 아주 작아질 수 있으므로
    # 어깨 폭 / 상체 길이 비율도 함께 출력합니다.

    normalized, _, scale = normalize_by_shoulder_width(
        points
    )

    length = torso_length(points)

    shoulder_hip_y = float(
        (
            normalized[LEFT_SHOULDER][1]
            + normalized[RIGHT_SHOULDER][1]
        ) / 2.0
    )

    print(
        f"Shoulder width: {scale:.1f}px / "
        f"Torso length: {length:.1f}px "
        f"(ratio {scale / length:.2f})"
    )

    print(
        f"Shoulder - hip y (shoulder-width normalized): "
        f"{shoulder_hip_y:.3f}"
    )

    if scale / length < 0.3:
        print(
            "  주의: 어깨가 겹쳐 보여 어깨 폭이 작습니다. "
            "어깨 폭 정규화 값이 크게 부풀 수 있습니다."
        )

    # ==============================================
    # ⑨ 팀 Feature 3개 계산 + 드러눕기 판단
    # ==============================================

    features = team_features(
        points,
        conf,
        box,
    )

    rule_tilt_only, lie_down = lie_down_rule(
        features
    )

    # ==============================================
    # ⑩ 최종 결과 출력
    # ==============================================

    print(
        f"Torso tilt: {features['torso_tilt_deg']:.1f}° "
        "(0°=upright, 90°=horizontal)"
    )

    print(
        f"Hip - ankle dy: {features['hip_ankle_dy']:.3f} "
        "(torso length unit, +: ankle below hip)"
    )

    print(
        f"BBox aspect (w/h): {features['bbox_aspect_ratio']:.3f}"
    )

    print(
        f"Rule 1 torso tilt only (>= {LIE_TILT_DEG:.0f}°): "
        f"{rule_tilt_only}"
    )

    print(
        f"Rule 2 + legs level (|dy| < {LIE_ANKLE_DY}): "
        f"{lie_down}"
    )

    print(f"Lie Down: {lie_down}")


# ==================================================
# 프로그램 실행 시작
# ==================================================

if __name__ == "__main__":
    main()
