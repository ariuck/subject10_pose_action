from __future__ import annotations

# ==================================================
# 3팀 — 무인 상점 이상행동 인식 : 팀 추가 Feature
# ==================================================
# 교재의 38-D Feature(정규화 좌표 34개 + 관절각도 4개)는
# 그대로 사용하고, 드러눕기를 설명하기 위한
# Feature 3개를 따로 계산합니다.
#
# 1. torso_tilt_deg    : 상체 기울기 (0° 서 있음, 90° 수평)
# 2. hip_ankle_dy      : 골반과 발목의 높이 차이 (상체 길이 단위)
# 3. bbox_aspect_ratio : 사람 BBox 가로/세로 비율
#
# 세 값은 모두 "각도" 또는 "비율"이라
# 사람이 화면에서 크거나 작아도 값이 크게 변하지 않습니다.

import math

import numpy as np

from src.feature_utils import (
    LEFT_ANKLE,
    LEFT_HIP,
    LEFT_SHOULDER,
    RIGHT_ANKLE,
    RIGHT_HIP,
    RIGHT_SHOULDER,
    midpoint,
)


# ==================================================
# [01] 팀 Feature 이름과 드러눕기 판정 기준
# ==================================================
# LIE_TILT_DEG
# → 상체가 세로 방향에서 60° 이상 기울면
#   "상체가 누운 쪽에 가깝다"고 봅니다.
#
# LIE_ANKLE_DY
# → 발목과 골반의 높이 차이가 상체 길이의 0.6배보다 작으면
#   "다리도 바닥과 나란하다"고 봅니다.

TEAM_FEATURE_NAMES = [
    "torso_tilt_deg",
    "hip_ankle_dy",
    "bbox_aspect_ratio",
]

LIE_TILT_DEG = 60.0
LIE_ANKLE_DY = 0.6


# ==================================================
# [02] 어깨 중심 / 골반 중심 / 상체 길이
# ==================================================

def shoulder_center(keypoints):
    return midpoint(
        keypoints[LEFT_SHOULDER],
        keypoints[RIGHT_SHOULDER],
    )


def hip_center(keypoints):
    return midpoint(
        keypoints[LEFT_HIP],
        keypoints[RIGHT_HIP],
    )


def torso_length(keypoints) -> float:
    """
    어깨 중심과 골반 중심 사이 거리입니다.

    옆에서 본 누운 자세에서는 두 어깨가 겹쳐
    어깨 폭이 거의 0이 될 수 있습니다.
    상체 길이는 이때도 안정적으로 남아 있어
    팀 Feature의 기준 길이로 사용합니다.
    """

    return float(
        np.linalg.norm(
            shoulder_center(keypoints)
            - hip_center(keypoints)
        )
    )


# ==================================================
# [03] torso_tilt_deg() - 상체 기울기
# ==================================================
# 골반 중심 → 어깨 중심 벡터와
# 화면의 위쪽 방향 (0, -1) 사이 각도를 구합니다.
#
#   0°  : 상체가 똑바로 서 있음
#   90° : 상체가 바닥과 수평
#   180°: 머리가 골반보다 아래 (물구나무, 일어나는 중)

def torso_tilt_deg(keypoints) -> float:
    vector = (
        shoulder_center(keypoints)
        - hip_center(keypoints)
    )

    length = float(
        np.linalg.norm(vector)
    )

    if length < 1e-8:
        return float("nan")

    # 위쪽 방향 (0, -1)과의 내적 = -vector_y
    cosine = float(
        np.clip(
            -vector[1] / length,
            -1.0,
            1.0,
        )
    )

    return math.degrees(
        math.acos(cosine)
    )


# ==================================================
# [04] visible_ankle() - 잘 보이는 쪽 발목 선택
# ==================================================
# 옆에서 촬영하면 카메라 반대쪽 발목은 가려져
# Confidence가 낮거나 위치가 틀릴 수 있습니다.
# 그래서 Confidence가 더 높은 발목 하나를 사용합니다.

def visible_ankle(
    keypoints,
    confidence,
):
    if confidence[LEFT_ANKLE] >= confidence[RIGHT_ANKLE]:
        return "left", keypoints[LEFT_ANKLE], float(confidence[LEFT_ANKLE])

    return "right", keypoints[RIGHT_ANKLE], float(confidence[RIGHT_ANKLE])


# ==================================================
# [05] hip_ankle_dy() - 골반과 발목의 높이 차이
# ==================================================
# (발목 y - 골반 y) / 상체 길이
#
#   약 1 이상 : 발목이 골반보다 한참 아래 (서기, 허리 숙이기)
#   0 근처    : 발목과 골반 높이가 비슷함 (눕기, 다리 뻗고 앉기)
#
# 화면 좌표는 아래로 갈수록 y가 커집니다.

def hip_ankle_dy(
    keypoints,
    confidence,
) -> float:
    length = torso_length(keypoints)

    if length < 1e-8:
        return float("nan")

    _, ankle, _ = visible_ankle(
        keypoints,
        confidence,
    )

    return float(
        (ankle[1] - hip_center(keypoints)[1])
        / length
    )


# ==================================================
# [06] bbox_aspect_ratio() - 사람 BBox 가로/세로 비율
# ==================================================
#   1보다 작음 : 세로로 긴 사람 (서 있음)
#   1보다 큼   : 가로로 긴 사람 (누워 있음)
#
# Keypoint가 아니라 YOLO가 찾은 사람 박스를 사용합니다.

def bbox_aspect_ratio(box) -> float:
    x1, y1, x2, y2 = [float(v) for v in box]

    height = y2 - y1

    if height < 1e-8:
        return float("nan")

    return (x2 - x1) / height


# ==================================================
# [07] team_features() - 팀 Feature 3개를 한 번에 계산
# ==================================================

def team_features(
    keypoints,
    confidence,
    box,
) -> dict:
    return {
        "torso_tilt_deg": torso_tilt_deg(keypoints),
        "hip_ankle_dy": hip_ankle_dy(keypoints, confidence),
        "bbox_aspect_ratio": bbox_aspect_ratio(box),
    }


# ==================================================
# [08] lie_down_rule() - 드러눕기 규칙 판정
# ==================================================
# rule_tilt_only
# → 상체 기울기 하나만 본 판정
#
# lie_down
# → 상체 기울기 + 다리 높이까지 본 판정
#
# 두 결과를 함께 반환해서
# "Feature 하나만으로는 부족하다"는 것을 비교할 수 있게 합니다.

def lie_down_rule(features: dict):
    tilt = features["torso_tilt_deg"]
    dy = features["hip_ankle_dy"]

    rule_tilt_only = bool(
        tilt >= LIE_TILT_DEG
    )

    lie_down = bool(
        rule_tilt_only
        and abs(dy) < LIE_ANKLE_DY
    )

    return rule_tilt_only, lie_down
