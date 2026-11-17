from __future__ import annotations

# ==================================================
# 3팀 추가 실습 — 17개 동작 영상의 팀 Feature 요약
# ==================================================
# 18번 코드로 만든 Frame별 Feature CSV를
# 동작 목록(configs/team3_actions.csv)과 연결해서
# 영상마다 한 줄로 요약합니다.
#
# 목적:
#   "어떤 Feature가 어떤 동작에서 크게 변하는가?"를
#   17개 동작 전체에서 비교하기 위해서입니다.
#
# 실행 전 준비:
#   python scripts/18_video_frame_features.py \
#     --source data/day02/videos/20261113_xxxxxx.mp4 --max-frames 0
#   (36개 영상 모두)

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

sys.path.insert(
    0,
    str(ROOT),
)


# ==================================================
# ① 실행 옵션
# ==================================================

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--actions",
        default=str(ROOT / "configs" / "team3_actions.csv"),
    )

    parser.add_argument(
        "--tables",
        default=str(ROOT / "reports" / "day03" / "tables"),
    )

    parser.add_argument(
        "--output",
        default="team3_action_summary.csv",
    )

    return parser.parse_args()


# ==================================================
# ② summarize_video() - 영상 하나를 한 줄로 요약
# ==================================================
# 동작의 "최고점"을 보기 위해 95 백분위수(p95)를 사용합니다.
# 최댓값은 Keypoint가 한 Frame 튀는 것에 크게 흔들리기 때문입니다.
#
# 움직임 Feature(속도)는 Frame 사이 Keypoint 흔들림이 섞이므로
# 3 Frame 이동 중앙값으로 한 번 다듬은 뒤 p95를 구합니다.

def summarize_video(df: pd.DataFrame) -> dict:
    valid = df[df["valid_pose"] == 1].copy()

    if valid.empty:
        return {"frames": len(df), "valid_rate": 0.0}

    def p95(column, smooth=False):
        values = pd.to_numeric(valid[column], errors="coerce")
        if smooth:
            values = values.rolling(3, center=True, min_periods=1).median()
        return round(float(values.quantile(0.95)), 3)

    def rate(mask):
        return round(float(mask.mean()) * 100.0, 1)

    return {
        "frames": len(df),
        "valid_rate": rate(df["valid_pose"] == 1),
        "mean_conf": round(float(valid["mean_keypoint_conf"].mean()), 3),
        "tilt_p95": p95("torso_tilt_deg"),
        "wrist_above_p95": p95("wrist_above_shoulder"),
        "ankle_diff_p95": p95("ankle_height_diff"),
        "aspect_p95": p95("bbox_aspect_ratio"),
        "wrist_speed_p95": p95("wrist_speed", smooth=True),
        "ankle_speed_p95": p95("ankle_speed", smooth=True),
        "hip_speed_p95": p95("hip_speed", smooth=True),
        "stand_pct": rate(valid["pose_state"] == "stand"),
        "sit_pct": rate(valid["pose_state"] == "sit"),
        "bend_pct": rate(valid["pose_state"] == "bend"),
        "lie_pct": rate(valid["pose_state"] == "lie"),
        "arm_raised_pct": rate(valid["arm_raised"] == 1),
        "leg_raised_pct": rate(valid["leg_raised"] == 1),
    }


def main() -> None:
    args = parse_args()

    actions = pd.read_csv(args.actions)

    tables = Path(args.tables)

    rows = []

    for _, action in actions.iterrows():
        path = tables / f"{action['video']}_frame_features.csv"

        if not path.exists():
            print(f"MISSING: {path.name}")
            continue

        summary = summarize_video(pd.read_csv(path))

        rows.append({**action.to_dict(), **summary})

    result = pd.DataFrame(rows)

    output_path = tables / args.output

    result.to_csv(output_path, index=False, encoding="utf-8-sig")

    # ==============================================
    # ③ 결과 출력
    # ==============================================

    columns = [
        "action_id", "action_name", "view",
        "tilt_p95", "wrist_above_p95", "ankle_diff_p95",
        "wrist_speed_p95", "ankle_speed_p95", "hip_speed_p95",
        "sit_pct", "bend_pct", "lie_pct", "arm_raised_pct", "leg_raised_pct",
        "mean_conf",
    ]

    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 30)

    print(result[columns].to_string(index=False))
    print()
    print(f"Saved: {output_path}")

    # ==============================================
    # ④ Git에 남길 Markdown 요약표 저장
    # ==============================================
    # CSV는 Git에 올리지 않으므로
    # 같은 결과를 Markdown 표로도 저장합니다.

    md_columns = [
        ("action_id", "#"),
        ("action_name", "동작"),
        ("view", "시점"),
        ("video", "영상"),
        ("tilt_p95", "상체 기울기 p95"),
        ("wrist_above_p95", "팔 올림 p95"),
        ("wrist_speed_p95", "손목 속도 p95"),
        ("ankle_speed_p95", "발목 속도 p95"),
        ("hip_speed_p95", "골반 속도 p95"),
        ("sit_pct", "앉기 %"),
        ("bend_pct", "숙이기 %"),
        ("lie_pct", "눕기 %"),
        ("arm_raised_pct", "팔 올림 %"),
        ("mean_conf", "평균 Conf"),
    ]

    lines = [
        "# 3팀 17개 동작 영상 — 팀 Feature 요약",
        "",
        "`python scripts/20_team_action_summary.py` 결과 (영상 36개, 18번 코드의 Frame별 CSV 사용)",
        "",
        "- p95: 영상 전체 Frame 중 95 백분위수 값 (동작의 최고점에 가까운 값)",
        "- 속도 단위: 1초에 상체 길이의 몇 배만큼 움직였는지 (3 Frame 중앙값으로 다듬은 뒤 계산)",
        "- 앉기 / 숙이기 / 눕기 %: 17번 규칙의 자세 상태가 나온 Frame 비율",
        "",
        "| " + " | ".join(title for _, title in md_columns) + " |",
        "|" + "---|" * len(md_columns),
    ]

    for _, row in result.iterrows():
        lines.append(
            "| " + " | ".join(str(row[key]) for key, _ in md_columns) + " |"
        )

    md_path = ROOT / "reports" / "day03" / "challenge" / "team3_action_summary.md"

    md_path.parent.mkdir(parents=True, exist_ok=True)

    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"Saved: {md_path}")


if __name__ == "__main__":
    main()
