"""3팀 Keypoint CSV의 움직임 크기로 동작 시작·끝 구간을 계산해 기록한다.

영상·Keypoint CSV는 자르지 않고, 구간만 data/team_projects/team03/manifests/action_segments.csv에 적는다.
(실패 Frame을 지우지 않는 것과 같은 원리)

움직임 크기(motion, 상체 길이/초)
1. 몸 관절 12개(어깨·팔꿈치·손목·골반·무릎·발목)의 Pixel 좌표가 0.1초 동안 움직인 평균 거리
   (60fps에서 바로 앞 Frame과 비교하면 YOLO 좌표 떨림이 크게 잡혀서 0.1초 간격을 쓴다)
2. 영상의 상체 길이(어깨 중심~골반 중심) 중앙값으로 나눠 사람 크기 차이를 없앤다
   (측면에서 불안정한 어깨 폭 대신 상체 길이를 쓴다)
3. 0.5초 이동 평균으로 부드럽게 만든다

구간 결정
- 기준값 = 하위 5% + (상위 95% - 하위 5%) x 0.2  (영상마다 따로 계산)
- 기준값을 0.15초 이상 넘은 첫 시점 = 시작, 마지막 시점 = 끝, 앞뒤 0.2초 여유
- 누워 있는 시간처럼 중간에 움직임이 적은 구간은 시작~끝 사이에 그대로 포함된다
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
TEAM = ROOT / "data" / "team_projects" / "team03"
DEFAULT_MANIFEST = TEAM / "manifests" / "keypoint_dataset_manifest.csv"
DEFAULT_OUTPUT = TEAM / "manifests" / "action_segments.csv"

JOINTS = ["left_shoulder", "right_shoulder", "left_elbow", "right_elbow", "left_wrist", "right_wrist",
          "left_hip", "right_hip", "left_knee", "right_knee", "left_ankle", "right_ankle"]
LAG_SEC = 0.1
SMOOTH_SEC = 0.5
THRESH_FRAC = 0.2
MIN_ACTIVE_SEC = 0.15
PAD_SEC = 0.2


def joint_xy(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """(Frame, 12) x / y Pixel 좌표. Pose가 없으면 NaN."""
    xs = np.stack([pd.to_numeric(df[f"{j}_x_px"], errors="coerce").to_numpy(float) for j in JOINTS], 1)
    ys = np.stack([pd.to_numeric(df[f"{j}_y_px"], errors="coerce").to_numpy(float) for j in JOINTS], 1)
    return xs, ys


def motion_curve(df: pd.DataFrame) -> tuple[np.ndarray, float, float]:
    """Frame별 움직임 크기(상체 길이/초), fps, 상체 길이(px) 중앙값."""
    t = pd.to_numeric(df["time_sec"], errors="coerce").to_numpy(float)
    fps = 1.0 / float(np.median(np.diff(t)))
    xs, ys = joint_xy(df)
    shoulder = np.stack([xs[:, 0] + xs[:, 1], ys[:, 0] + ys[:, 1]], 1) / 2
    hip = np.stack([xs[:, 6] + xs[:, 7], ys[:, 6] + ys[:, 7]], 1) / 2
    torso = float(np.nanmedian(np.linalg.norm(shoulder - hip, axis=1)))

    lag = max(1, round(LAG_SEC * fps))
    dist = np.full(len(df), np.nan)
    step = np.sqrt((xs[lag:] - xs[:-lag]) ** 2 + (ys[lag:] - ys[:-lag]) ** 2)
    valid = np.isfinite(step).any(axis=1)
    mean_step = np.full(len(step), np.nan)
    mean_step[valid] = np.nanmean(step[valid], axis=1)
    dist[lag:] = mean_step / torso / (lag / fps)

    s = pd.Series(dist).interpolate(limit_direction="both")
    win = max(1, round(SMOOTH_SEC * fps))
    s = s.rolling(win, center=True, min_periods=1).mean()
    return s.to_numpy(), fps, torso


def find_segment(m: np.ndarray, fps: float, n: int) -> dict:
    p5, p95 = np.nanpercentile(m, [5, 95])
    thr = p5 + THRESH_FRAC * (p95 - p5)
    above = np.nan_to_num(m, nan=0.0) > thr
    min_len = max(1, round(MIN_ACTIVE_SEC * fps))
    runs, start = [], None
    for i, a in enumerate(np.append(above, False)):
        if a and start is None:
            start = i
        elif not a and start is not None:
            if i - start >= min_len:
                runs.append((start, i - 1))
            start = None
    s_idx, e_idx = (runs[0][0], runs[-1][1]) if runs else (0, n - 1)
    pad = round(PAD_SEC * fps)
    return {"start_idx": max(0, s_idx - pad), "end_idx": min(n - 1, e_idx + pad),
            "motion_p5": p5, "motion_p95": p95, "threshold": thr}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest-path", default=str(DEFAULT_MANIFEST))
    parser.add_argument("--output-path", default=str(DEFAULT_OUTPUT))
    args = parser.parse_args()

    manifest = pd.read_csv(ROOT / args.manifest_path)
    rows = []
    for item in manifest.itertuples():
        df = pd.read_csv(ROOT / item.keypoint_csv)
        t = pd.to_numeric(df["time_sec"], errors="coerce").to_numpy(float)
        m, fps, torso = motion_curve(df)
        seg = find_segment(m, fps, len(df))
        s, e = seg["start_idx"], seg["end_idx"]
        rows.append({
            "clip_id": item.clip_id,
            "subject": item.subject,
            "action_id": item.action_id,
            "action_label": item.action_label,
            "take": item.take,
            "fps": round(fps, 3),
            "duration_sec": round(float(t[-1]), 3),
            "start_frame": int(df.frame_id.iloc[s]),
            "end_frame": int(df.frame_id.iloc[e]),
            "start_sec": round(float(t[s]), 3),
            "end_sec": round(float(t[e]), 3),
            "active_sec": round(float(t[e] - t[s]), 3),
            "trim_head_sec": round(float(t[s]), 3),
            "trim_tail_sec": round(float(t[-1] - t[e]), 3),
            "torso_px": round(torso, 1),
            "motion_p5": round(float(seg["motion_p5"]), 3),
            "motion_p95": round(float(seg["motion_p95"]), 3),
            "threshold": round(float(seg["threshold"]), 3),
            "method": "auto_motion",
        })

    out = ROOT / args.output_path
    out.parent.mkdir(parents=True, exist_ok=True)
    result = pd.DataFrame(rows)
    result.to_csv(out, index=False, encoding="utf-8-sig")
    print(result[["trim_head_sec", "trim_tail_sec", "active_sec"]].describe().round(2).to_string())
    print(f"[SAVED] {out.relative_to(ROOT)} ({len(result)} rows)")


if __name__ == "__main__":
    sys.path.insert(0, str(ROOT))
    main()
