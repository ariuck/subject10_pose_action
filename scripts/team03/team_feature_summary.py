"""3팀 45-D Sequence에서 동작 구간(in_action=1)만 골라 동작별 팀 Feature를 요약한다.

출력: reports/team_projects/team03/tables/feature_by_action.csv   (동작 × 방향)
      reports/team_projects/team03/tables/feature_by_clip.csv     (Clip별 값)

Clip별 요약 값 (동작 구간 안에서)
- tilt_max           : 상체 기울기 상위 95% 값 (가장 많이 기울었을 때, 이상값 제외)
- hip_floor_min      : 골반-바닥 높이 하위 5% 값 (가장 낮게 내려갔을 때)
- aspect_max         : BBox 가로/세로 비율 상위 95% 값
- wrist_speed_p95    : 두 손목 중 빠른 쪽 속도의 상위 95% 값
- ankle_speed_p95    : 두 발목 중 빠른 쪽 속도의 상위 95% 값
- lie/sit/bend/stand : 팀 자세 규칙(pose_state) 비율
"""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
TEAM = ROOT / "data" / "team_projects" / "team03"
OUT = ROOT / "reports" / "team_projects" / "team03" / "tables"


def clip_summary(path: Path) -> dict:
    df = pd.read_csv(path)
    a = df[(df.in_action == 1) & (df.pose_ok == 1)]
    wrist = a[["left_wrist_speed", "right_wrist_speed"]].max(axis=1)
    ankle = a[["left_ankle_speed", "right_ankle_speed"]].max(axis=1)
    states = a.pose_state.value_counts(normalize=True)
    return {
        "clip_id": df.clip_id.iloc[0], "subject": df.subject.iloc[0], "action_id": df.action_id.iloc[0],
        "action_label": df.action_label.iloc[0], "take": df["take"].iloc[0],
        "view": {"T01": "front", "T02": "side", "T03": "back"}[df["take"].iloc[0]],
        "action_frames": len(a),
        "tilt_max": a.torso_tilt_deg.quantile(0.95),
        "hip_floor_min": a.hip_floor_height.quantile(0.05),
        "aspect_max": a.bbox_aspect_ratio.quantile(0.95),
        "wrist_speed_p95": wrist.quantile(0.95),
        "ankle_speed_p95": ankle.quantile(0.95),
        **{f"{s}_ratio": float(states.get(s, 0.0)) for s in ("lie", "sit", "bend", "stand")},
        "arm_raised_ratio": float(pd.to_numeric(a.arm_raised, errors="coerce").mean()),
        "leg_raised_ratio": float(pd.to_numeric(a.leg_raised, errors="coerce").mean()),
    }


def main() -> None:
    seq = pd.read_csv(TEAM / "manifests" / "sequence_manifest.csv")
    clips = pd.DataFrame([clip_summary(ROOT / p) for p in seq.sequence_csv])
    OUT.mkdir(parents=True, exist_ok=True)
    clips.to_csv(OUT / "feature_by_clip.csv", index=False, encoding="utf-8-sig")
    value_cols = [c for c in clips.columns if c not in
                  ("clip_id", "subject", "action_id", "action_label", "take", "view", "action_frames")]
    by_action = clips.groupby(["action_id", "action_label", "view"])[value_cols].median().round(2).reset_index()
    by_action.to_csv(OUT / "feature_by_action.csv", index=False, encoding="utf-8-sig")
    pd.set_option("display.width", 250)
    print(by_action.to_string(index=False))
    print(f"\n[SAVED] {(OUT / 'feature_by_action.csv').relative_to(ROOT)}")


if __name__ == "__main__":
    main()
