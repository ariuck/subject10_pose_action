"""3팀 Keypoint CSV → 45-D Feature Sequence CSV (4일차 RGB 문서 STEP 4, 강사님 3팀 추천 Feature).

입력 : data/team_projects/team03/manifests/keypoint_dataset_manifest.csv  (team_extract_keypoint_dataset.py 결과)
       data/team_projects/team03/manifests/action_segments.csv           (team_action_segments.py 결과)
출력 : data/team_projects/team03/sequences/<clip_id>_sequence.csv          (Clip마다 1개, 덮어쓰기 없음)
       data/team_projects/team03/manifests/sequence_manifest.csv

Frame 하나의 Feature
- 교재 38-D : 골반 중심 + 어깨 폭 정규화 좌표 34개 + 팔꿈치·무릎 각도 4개
              (src/feature_utils.py, 4일차 22번과 같은 계산)
- 3팀 +7    : 강사님 3팀 추천 Feature(+7 → 45-D)를 우리 영상에 맞춘 것
    1 torso_tilt_deg     상체 기울기 (0° 서기, 90° 수평)
    2 bbox_aspect_ratio  사람 BBox 가로/세로 비율
    3 hip_floor_height   골반과 바닥의 상대 높이 = (BBox 아래 끝 y - 골반 중심 y) / 상체 길이
    4 left_ankle_speed   왼발목 속도 (상체 길이/초, 0.1초 간격)
    5 right_ankle_speed  오른발목 속도
    6 left_wrist_speed   왼손목 속도   ┐ 강사님 표의 "양발과 문 사이 거리 2개" 대신 사용.
    7 right_wrist_speed  오른손목 속도 ┘ 촬영 영상에 문·물체가 없어 거리를 계산할 수 없고,
                                         내려치기·던지기·휘두르기처럼 손을 빠르게 쓰는 동작이 많기 때문
- 참고 열(학습 Feature 아님): 원본 골반 위치·상체 길이·BBox·Confidence·동작 구간(in_action)·자세 규칙

실패 Frame도 지우지 않는다 (pose_ok / feature_ok / feature45_ok로 표시).
"""
import argparse
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.common import load_config  # noqa: E402
from src.feature_utils import normalize_by_shoulder_width, selected_joint_angles  # noqa: E402
from src.pose_utils import COCO_KEYPOINT_NAMES  # noqa: E402
from src.team_features import (  # noqa: E402
    ARM_RAISED,
    LEG_RAISED,
    action_flags,
    ankle_height_diff,
    hip_ankle_dy,
    pose_state,
    torso_length,
    torso_tilt_deg,
    visible_knee_angle,
    wrist_above_shoulder,
)

TEAM = ROOT / "data" / "team_projects" / "team03"
DEFAULT_MANIFEST = TEAM / "manifests" / "keypoint_dataset_manifest.csv"
DEFAULT_SEGMENTS = TEAM / "manifests" / "action_segments.csv"
DEFAULT_OUTPUT_DIR = TEAM / "sequences"
DEFAULT_SEQ_MANIFEST = TEAM / "manifests" / "sequence_manifest.csv"
LAG_SEC = 0.1

BASE_FEATURES = [f"{name}_{axis}" for name in COCO_KEYPOINT_NAMES for axis in ("x", "y")] + [
    "left_elbow_angle", "right_elbow_angle", "left_knee_angle", "right_knee_angle"]
TEAM_FEATURES = ["torso_tilt_deg", "bbox_aspect_ratio", "hip_floor_height",
                 "left_ankle_speed", "right_ankle_speed", "left_wrist_speed", "right_wrist_speed"]
FEATURES_45 = BASE_FEATURES + TEAM_FEATURES
SPEED_JOINTS = {"left_ankle_speed": "left_ankle", "right_ankle_speed": "right_ankle",
                "left_wrist_speed": "left_wrist", "right_wrist_speed": "right_wrist"}
assert len(BASE_FEATURES) == 38 and len(FEATURES_45) == 45


def frame_arrays(df: pd.DataFrame):
    xy = np.stack([
        np.stack([pd.to_numeric(df[f"{n}_x_px"], errors="coerce").to_numpy(float),
                  pd.to_numeric(df[f"{n}_y_px"], errors="coerce").to_numpy(float)], 1)
        for n in COCO_KEYPOINT_NAMES], 1)                      # (F, 17, 2)
    conf = np.stack([pd.to_numeric(df[f"{n}_conf"], errors="coerce").to_numpy(float)
                     for n in COCO_KEYPOINT_NAMES], 1)          # (F, 17)
    box = df[["box_x1", "box_y1", "box_x2", "box_y2"]].apply(pd.to_numeric, errors="coerce").to_numpy(float)
    return xy, conf, box


def build_clip(item, segment, kp_threshold: float, output_dir: Path) -> dict:
    df = pd.read_csv(ROOT / item.keypoint_csv)
    xy, conf, box = frame_arrays(df)
    t = pd.to_numeric(df["time_sec"], errors="coerce").to_numpy(float)
    fps = 1.0 / float(np.median(np.diff(t)))
    lag = max(1, round(LAG_SEC * fps))
    pose = pd.to_numeric(df["pose_detected"], errors="coerce").fillna(0).to_numpy(int)

    torso_series = np.array([torso_length(k) if p else np.nan for k, p in zip(xy, pose)])
    torso_clip = float(np.nanmedian(torso_series)) if np.isfinite(torso_series).any() else np.nan

    rows = []
    for i in range(len(df)):
        row = {
            "clip_id": item.clip_id, "subject": item.subject, "action_id": item.action_id,
            "action_label": item.action_label, "take": item.take,
            "frame_id": int(df.frame_id.iloc[i]), "time_sec": float(t[i]),
            "in_action": int(segment.start_frame <= int(df.frame_id.iloc[i]) <= segment.end_frame),
            "pose_ok": int(pose[i]), "feature_ok": 0, "feature45_ok": 0,
            "mean_keypoint_conf": "", "low_keypoint_count": "",
            "hip_center_x_px": "", "hip_center_y_px": "", "shoulder_width_px": "", "torso_length_px": "",
            "box_x1": "", "box_y1": "", "box_x2": "", "box_y2": "",
            **{name: "" for name in FEATURES_45},
            "pose_state": "", "arm_raised": "", "leg_raised": "",
        }
        if pose[i]:
            k, c = xy[i], conf[i]
            row["mean_keypoint_conf"] = round(float(np.mean(c)), 4)
            row["low_keypoint_count"] = int(np.sum(c < kp_threshold))
            row.update({key: round(float(v), 2) for key, v in zip(("box_x1", "box_y1", "box_x2", "box_y2"), box[i])})
            values = {}
            try:
                normalized, center, scale = normalize_by_shoulder_width(k)
                angles = selected_joint_angles(k)
                base = [float(v) for p in normalized for v in p] + [
                    angles["left_elbow_angle"], angles["right_elbow_angle"],
                    angles["left_knee_angle"], angles["right_knee_angle"]]
                if len(base) == 38 and np.all(np.isfinite(base)):
                    values.update(zip(BASE_FEATURES, base))
                    row["feature_ok"] = 1
                    row["hip_center_x_px"] = round(float(center[0]), 3)
                    row["hip_center_y_px"] = round(float(center[1]), 3)
                    row["shoulder_width_px"] = round(float(scale), 3)
            except ValueError:
                pass

            length = torso_series[i]
            row["torso_length_px"] = round(float(length), 3) if np.isfinite(length) else ""
            values["torso_tilt_deg"] = torso_tilt_deg(k)
            bw, bh = box[i][2] - box[i][0], box[i][3] - box[i][1]
            values["bbox_aspect_ratio"] = bw / bh if bh > 1e-8 else math.nan
            hip_y = (k[11][1] + k[12][1]) / 2
            values["hip_floor_height"] = (box[i][3] - hip_y) / torso_clip if torso_clip > 1e-8 else math.nan
            for name, joint in SPEED_JOINTS.items():
                j = COCO_KEYPOINT_NAMES.index(joint)
                if i >= lag and pose[i - lag] and torso_clip > 1e-8:
                    values[name] = float(np.linalg.norm(xy[i][j] - xy[i - lag][j])) / torso_clip / (lag / fps)
                else:
                    values[name] = math.nan
            for name, v in values.items():
                if np.isfinite(v):
                    row[name] = round(float(v), 6)
            row["feature45_ok"] = int(row["feature_ok"] == 1 and all(
                np.isfinite(values.get(n, math.nan)) for n in TEAM_FEATURES))

            features = {"torso_tilt_deg": values["torso_tilt_deg"], "hip_ankle_dy": hip_ankle_dy(k, c),
                        "wrist_above_shoulder": wrist_above_shoulder(k, c), "ankle_height_diff": ankle_height_diff(k, c)}
            if np.isfinite(features["torso_tilt_deg"]) and np.isfinite(features["hip_ankle_dy"]):
                row["pose_state"] = pose_state(features, visible_knee_angle(k, c))
            arm, leg = action_flags(features)
            row["arm_raised"], row["leg_raised"] = int(arm), int(leg)
        rows.append(row)

    out = output_dir / f"{item.clip_id}_sequence.csv"
    pd.DataFrame(rows).to_csv(out, index=False, encoding="utf-8-sig")
    seq = pd.DataFrame(rows)
    return {
        "clip_id": item.clip_id, "subject": item.subject, "action_id": item.action_id,
        "action_label": item.action_label, "take": item.take,
        "sequence_csv": out.relative_to(ROOT).as_posix(), "keypoint_csv": item.keypoint_csv,
        "rows": len(seq), "pose_ok_rows": int(seq.pose_ok.sum()), "feature_ok_rows": int(seq.feature_ok.sum()),
        "feature45_ok_rows": int(seq.feature45_ok.sum()), "in_action_rows": int(seq.in_action.sum()),
        "start_sec": segment.start_sec, "end_sec": segment.end_sec,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest-path", default=str(DEFAULT_MANIFEST))
    parser.add_argument("--segments-path", default=str(DEFAULT_SEGMENTS))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--sequence-manifest", default=str(DEFAULT_SEQ_MANIFEST))
    args = parser.parse_args()

    kp_threshold = float(load_config().get("keypoint_confidence", 0.5))
    manifest = pd.read_csv(ROOT / args.manifest_path)
    segments = pd.read_csv(ROOT / args.segments_path).set_index("clip_id")
    output_dir = ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    results = []
    for n, item in enumerate(manifest.itertuples(), start=1):
        res = build_clip(item, segments.loc[item.clip_id], kp_threshold, output_dir)
        results.append(res)
        if n % 17 == 0 or n == len(manifest):
            print(f"[{n}/{len(manifest)}] {item.clip_id} rows={res['rows']} "
                  f"feature_ok={res['feature_ok_rows']} feature45_ok={res['feature45_ok_rows']}")

    summary = pd.DataFrame(results)
    out = ROOT / args.sequence_manifest
    out.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(out, index=False, encoding="utf-8-sig")
    print()
    print(f"Feature per frame: 38 + {len(TEAM_FEATURES)} = {len(FEATURES_45)}")
    print(f"Clips: {len(summary)}  Frames: {summary.rows.sum()}  pose_ok: {summary.pose_ok_rows.sum()}  "
          f"feature_ok(38): {summary.feature_ok_rows.sum()}  feature45_ok: {summary.feature45_ok_rows.sum()}")
    print(f"[SAVED] {out.relative_to(ROOT)}")
    print(f"(pose_state 기준: arm_raised >= {ARM_RAISED}, leg_raised >= {LEG_RAISED})")


if __name__ == "__main__":
    main()
