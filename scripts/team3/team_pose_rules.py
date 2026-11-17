from __future__ import annotations

# ==================================================
# 3팀 — 무인 상점 이상행동 인식
# 17. 팀 동작을 설명하는 상대 위치 Feature 규칙 확인
# ==================================================
# 교재 24번(Arms Up 규칙)과 같은 흐름을
# 우리 팀 17개 동작에 맞게 바꾼 코드입니다.
#
# 한 장의 이미지에서 다음을 확인합니다.
#   - 자세 상태 : 서기 / 앉기 / 숙이기 / 눕기
#   - 팔 올림   : 내려치기·던지기·휘두르기의 준비 자세
#   - 발 들기   : 차기 자세
#
# 완성된 AI 분류 모델이 아니라
# "어떤 Feature가 어떤 동작을 설명하는가?"를
# 간단한 규칙으로 확인하는 연습 코드입니다.

# ==================================================
# ① 필요한 라이브러리와 공통 함수 가져오기
# ==================================================

import argparse
import sys
from pathlib import Path

import numpy as np
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[2]

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
    shoulder_width,
)

from src.team_features import (
    ARM_RAISED,
    LEG_RAISED,
    LIE_ANKLE_DY,
    LIE_TILT_DEG,
    action_flags,
    lie_down_rule,
    pose_features,
    pose_state,
    torso_length,
    visible_ankle,
    visible_knee_angle,
)


# ==================================================
# ② 터미널 실행 옵션 설정
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
            / "team3"
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


STATE_NAMES = {
    "stand": "서기",
    "sit": "앉기",
    "bend": "숙이기",
    "lie": "눕기",
}


def main() -> None:

    # ==============================================
    # ③ 이미지와 실행환경 확인, YOLO Pose 실행
    # ==============================================

    args = parse_args()

    source = Path(args.source)

    if not source.exists():
        raise FileNotFoundError(source)

    config = load_config()

    device = resolve_device(config["device"])

    model = YOLO(config["pose_model"])

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
        raise RuntimeError("Pose를 찾지 못했습니다.")

    # ==============================================
    # ④ 관절 좌표, Confidence, 사람 BBox 추출
    # ==============================================

    points = result.keypoints.xy[0].cpu().numpy()

    if result.keypoints.conf is not None:
        conf = result.keypoints.conf[0].cpu().numpy()
    else:
        conf = np.ones(17, dtype=np.float32)

    box = result.boxes.xyxy[0].cpu().numpy()

    # ==============================================
    # ⑤ 필요한 관절의 신뢰도 검사
    # ==============================================
    # 상체 기울기 : 양쪽 어깨 + 양쪽 골반 (4개 모두 필요)
    # 다리 높이   : 잘 보이는 쪽 발목 1개

    torso_ids = [LEFT_SHOULDER, RIGHT_SHOULDER, LEFT_HIP, RIGHT_HIP]

    min_torso_conf = float(np.min(conf[torso_ids]))

    ankle_side, _, ankle_conf = visible_ankle(points, conf)

    print(f"Source: {source.name}")
    print(f"Min required confidence (shoulder/hip): {min_torso_conf:.3f}")
    print(f"Visible ankle: {ankle_side} ({ankle_conf:.3f})")

    if min_torso_conf < args.kp_conf:
        print("SKIP: 어깨·골반 Confidence가 낮아 상체 기울기를 믿을 수 없습니다.")
        return

    if ankle_conf < args.kp_conf:
        print("SKIP: 발목 Confidence가 낮아 다리 높이를 믿을 수 없습니다.")
        return

    # ==============================================
    # ⑥ 기준 길이 확인 (어깨 폭 vs 상체 길이)
    # ==============================================
    # 측면에서는 두 어깨가 겹쳐 어깨 폭이 작아집니다.
    # 팀 Feature는 상체 길이로 나눕니다.

    scale = shoulder_width(points)

    length = torso_length(points)

    print(
        f"Shoulder width: {scale:.1f}px / "
        f"Torso length: {length:.1f}px "
        f"(ratio {scale / length:.2f})"
    )

    # ==============================================
    # ⑦ 팀 자세 Feature 계산
    # ==============================================

    features = pose_features(points, conf, box)

    knee = visible_knee_angle(points, conf)

    print()
    print("Team pose features")
    print(f"torso_tilt_deg       : {features['torso_tilt_deg']:7.2f}°  (0°=서기, 90°=수평)")
    print(f"hip_ankle_dy         : {features['hip_ankle_dy']:7.3f}   (+: 발목이 골반보다 아래)")
    print(f"bbox_aspect_ratio    : {features['bbox_aspect_ratio']:7.3f}   (>1: 가로로 긴 사람)")
    print(f"wrist_above_shoulder : {features['wrist_above_shoulder']:7.3f}   (+: 손목이 어깨보다 위)")
    print(f"ankle_height_diff    : {features['ankle_height_diff']:7.3f}   (큼: 한 발을 듦)")
    print(f"visible knee angle   : {knee:7.2f}°")

    # ==============================================
    # ⑧ 규칙 판정
    # ==============================================

    state = pose_state(features, knee)

    rule_tilt_only, lie_down = lie_down_rule(features)

    arm_raised, leg_raised = action_flags(features)

    print()
    print("Rules")
    print(f"Lie rule 1 torso tilt only (>= {LIE_TILT_DEG:.0f}°): {rule_tilt_only}")
    print(f"Lie rule 2 + legs level (|dy| < {LIE_ANKLE_DY}): {lie_down}")
    print(f"Arm raised (wrist above shoulder >= {ARM_RAISED}): {arm_raised}")
    print(f"Leg raised (ankle height diff >= {LEG_RAISED}): {leg_raised}")
    print(f"Pose state: {state} ({STATE_NAMES[state]})")


# ==================================================
# 프로그램 실행 시작
# ==================================================

if __name__ == "__main__":
    main()
