"""Sequence CSV의 움직임 크기로 동작 시작·끝 구간을 계산해 기록한다.

영상과 Sequence CSV는 자르지 않고, 구간만 configs/team3_action_segments.csv에 적는다.

움직임 크기(motion, 상체 길이/초)
1. 정규화 좌표를 원래 Pixel 좌표로 되돌린다: 좌표 * shoulder_width + hip_center
2. 몸통 관절 12개(어깨·팔꿈치·손목·골반·무릎·발목)가 0.1초 동안 움직인 거리를 평균한다.
   (60fps에서 바로 앞 Frame과 비교하면 YOLO 좌표 떨림이 크게 잡혀서 0.1초 간격을 쓴다)
3. 사람 크기 차이를 없애기 위해 영상의 상체 길이(어깨 중심~골반 중심) 중앙값으로 나눈다.
   (측면에서 불안정한 어깨 폭 대신 상체 길이를 쓴다)
4. 0.5초 이동 평균으로 부드럽게 만든다.

구간 결정
- 기준값 = 하위 5% + (상위 95% - 하위 5%) x 0.2  (영상마다 따로 계산)
- 기준값을 0.15초 이상 넘은 첫 시점 = 시작, 마지막 시점 = 끝
- 앞뒤로 0.2초 여유를 둔다.
- 누워 있는 시간처럼 중간에 움직임이 적은 구간은 시작~끝 사이에 그대로 포함된다.

사용 예:
    python scripts/team3/team_action_segments.py              # 있는 Sequence 전부
    python scripts/team3/team_action_segments.py --person P3
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
LABEL_FILE = ROOT / "configs" / "team3_all_actions.csv"
SEQ_ROOT = ROOT / "reports" / "team3_videos" / "sequences"
OUTPUT = ROOT / "configs" / "team3_action_segments.csv"

JOINTS = ["left_shoulder", "right_shoulder", "left_elbow", "right_elbow", "left_wrist", "right_wrist",
          "left_hip", "right_hip", "left_knee", "right_knee", "left_ankle", "right_ankle"]
LAG_SEC = 0.1
SMOOTH_SEC = 0.5
THRESH_FRAC = 0.2
MIN_ACTIVE_SEC = 0.15
PAD_SEC = 0.2


def motion_curve(df: pd.DataFrame) -> tuple[np.ndarray, float, float]:
    """Frame별 움직임 크기(상체 길이/초), fps, 상체 길이(px)."""
    t = df.time_sec.to_numpy()
    fps = 1.0 / np.median(np.diff(t))
    ok = (df.feature_ok == 1).to_numpy()
    sw = df.shoulder_width.to_numpy()
    xs = np.stack([df[f"{j}_x"].to_numpy() * sw + df.hip_center_x.to_numpy() for j in JOINTS], 1)
    ys = np.stack([df[f"{j}_y"].to_numpy() * sw + df.hip_center_y.to_numpy() for j in JOINTS], 1)
    xs[~ok] = np.nan
    ys[~ok] = np.nan

    shoulder = np.stack([xs[:, 0] + xs[:, 1], ys[:, 0] + ys[:, 1]], 1) / 2
    hip = np.stack([xs[:, 6] + xs[:, 7], ys[:, 6] + ys[:, 7]], 1) / 2
    torso = float(np.nanmedian(np.linalg.norm(shoulder - hip, axis=1)))

    lag = max(1, round(LAG_SEC * fps))
    dist = np.full(len(df), np.nan)
    step = np.sqrt((xs[lag:] - xs[:-lag]) ** 2 + (ys[lag:] - ys[:-lag]) ** 2)
    dist[lag:] = np.nanmean(step, axis=1) / torso / (lag / fps)

    s = pd.Series(dist).interpolate(limit_direction="both")
    win = max(1, round(SMOOTH_SEC * fps))
    s = s.rolling(win, center=True, min_periods=1).mean()
    return s.to_numpy(), fps, torso


def find_segment(m: np.ndarray, t: np.ndarray, fps: float) -> dict:
    p5, p95 = np.nanpercentile(m, [5, 95])
    thr = p5 + THRESH_FRAC * (p95 - p5)
    above = m > thr

    # 기준값을 MIN_ACTIVE_SEC 이상 연속으로 넘은 구간만 인정
    min_len = max(1, round(MIN_ACTIVE_SEC * fps))
    runs, start = [], None
    for i, a in enumerate(np.append(above, False)):
        if a and start is None:
            start = i
        elif not a and start is not None:
            if i - start >= min_len:
                runs.append((start, i - 1))
            start = None
    if not runs:
        s_idx, e_idx = 0, len(t) - 1
    else:
        s_idx, e_idx = runs[0][0], runs[-1][1]

    pad = round(PAD_SEC * fps)
    s_idx = max(0, s_idx - pad)
    e_idx = min(len(t) - 1, e_idx + pad)
    return {"start_idx": s_idx, "end_idx": e_idx, "motion_p5": p5, "motion_p95": p95, "threshold": thr}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--person", default=None, help="P1~P5 (생략하면 Sequence가 있는 사람 전부)")
    args = parser.parse_args()

    labels = pd.read_csv(LABEL_FILE, keep_default_na=False).set_index("sample_id")
    people = [args.person] if args.person else sorted(p.name for p in SEQ_ROOT.iterdir() if p.is_dir())

    rows = []
    for person_id in people:
        for path in sorted((SEQ_ROOT / person_id).glob("*_sequence.csv")):
            sample_id = path.stem.removesuffix("_sequence")
            df = pd.read_csv(path, encoding="utf-8-sig")
            t = df.time_sec.to_numpy()
            m, fps, torso = motion_curve(df)
            seg = find_segment(m, t, fps)
            s, e = seg["start_idx"], seg["end_idx"]
            lab = labels.loc[sample_id]
            rows.append({
                "sample_id": sample_id,
                "person_id": person_id,
                "action_id": lab.action_id,
                "action_label": lab.action_label,
                "view": lab.view,
                "fps": round(fps, 2),
                "duration_sec": round(float(t[-1]), 3),
                "start_frame": int(df.frame_id.iloc[s]),
                "end_frame": int(df.frame_id.iloc[e]),
                "start_sec": round(float(t[s]), 3),
                "end_sec": round(float(t[e]), 3),
                "active_sec": round(float(t[e] - t[s]), 3),
                "trim_head_sec": round(float(t[s]), 3),
                "trim_tail_sec": round(float(t[-1] - t[e]), 3),
                "torso_px": round(torso, 1),
                "motion_p5": round(seg["motion_p5"], 3),
                "motion_p95": round(seg["motion_p95"], 3),
                "threshold": round(seg["threshold"], 3),
                "method": "auto_motion",
            })

    new = pd.DataFrame(rows)
    if OUTPUT.exists() and args.person:
        old = pd.read_csv(OUTPUT, keep_default_na=False)
        new = pd.concat([old[~old.sample_id.isin(new.sample_id)], new])
    new = new.sort_values("sample_id").reset_index(drop=True)
    new.to_csv(OUTPUT, index=False, encoding="utf-8")
    print(f"[SAVED] {OUTPUT.relative_to(ROOT)} ({len(new)} rows)")


if __name__ == "__main__":
    main()
