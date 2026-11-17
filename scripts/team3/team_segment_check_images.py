"""동작 구간 확인용 이미지 만들기.

영상 1개 = 한 줄: [처음] [시작] [가운데] [끝] [마지막] [움직임 그래프]
그래프의 초록 선 = 시작, 빨간 선 = 끝, 회색 가로선 = 기준값

사용 예:
    python scripts/team3/team_segment_check_images.py --person P3
"""
import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from team_action_segments import SEQ_ROOT, motion_curve  # noqa: E402

LABEL_FILE = ROOT / "configs" / "team3_all_actions.csv"
SEGMENT_FILE = ROOT / "configs" / "team3_action_segments.csv"
OUT_DIR = ROOT / "reports" / "team3_videos" / "segments"
TH = 150
CELL_W = 270
GRAPH_W = 420
ROWS_PER_IMAGE = 12


def read_frame(cap, idx):
    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
    ok, frame = cap.read()
    return frame if ok else None


def cell(frame, text, color=(255, 255, 255)):
    canvas = np.full((TH, CELL_W, 3), 255, np.uint8)
    if frame is not None:
        h, w = frame.shape[:2]
        scale = min(TH / h, CELL_W / w)
        f = cv2.resize(frame, (int(w * scale), int(h * scale)))
        x0 = (CELL_W - f.shape[1]) // 2
        canvas[: f.shape[0], x0: x0 + f.shape[1]] = f
    cv2.rectangle(canvas, (0, 0), (CELL_W, 22), (0, 0, 0), -1)
    cv2.putText(canvas, text, (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
    return canvas


def graph(m, t, seg, title):
    g = np.full((TH, GRAPH_W, 3), 255, np.uint8)
    top = np.nanpercentile(m, 99) * 1.1 or 1.0
    xs = (t / t[-1] * (GRAPH_W - 10) + 5).astype(int)
    ys = (TH - 10 - np.clip(m / top, 0, 1) * (TH - 40)).astype(int)
    ty = int(TH - 10 - min(seg.threshold / top, 1) * (TH - 40))
    cv2.line(g, (5, ty), (GRAPH_W - 5, ty), (170, 170, 170), 1)
    cv2.polylines(g, [np.stack([xs, ys], 1).reshape(-1, 1, 2)], False, (200, 90, 0), 2)
    for sec, col in ((seg.start_sec, (0, 160, 0)), (seg.end_sec, (0, 0, 220))):
        x = int(sec / t[-1] * (GRAPH_W - 10) + 5)
        cv2.line(g, (x, 24), (x, TH - 5), col, 2)
    cv2.rectangle(g, (0, 0), (GRAPH_W, 22), (0, 0, 0), -1)
    cv2.putText(g, title, (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    return g


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--person", required=True)
    args = parser.parse_args()

    labels = pd.read_csv(LABEL_FILE, keep_default_na=False).set_index("sample_id")
    segs = pd.read_csv(SEGMENT_FILE, keep_default_na=False)
    segs = segs[segs.person_id == args.person].reset_index(drop=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    lines = []
    for seg in segs.itertuples():
        df = pd.read_csv(SEQ_ROOT / args.person / f"{seg.sample_id}_sequence.csv", encoding="utf-8-sig")
        m, fps, _ = motion_curve(df)
        t = df.time_sec.to_numpy()
        cap = cv2.VideoCapture(str(ROOT / labels.loc[seg.sample_id, "video_file"]))
        first, last = int(df.frame_id.iloc[0]), int(df.frame_id.iloc[-1])
        mid = (seg.start_frame + seg.end_frame) // 2
        cells = [
            cell(read_frame(cap, first), f"{seg.sample_id} 0.0s"),
            cell(read_frame(cap, seg.start_frame), f"START {seg.start_sec:.1f}s", (120, 255, 120)),
            cell(read_frame(cap, mid), f"mid {mid / fps:.1f}s"),
            cell(read_frame(cap, seg.end_frame), f"END {seg.end_sec:.1f}s", (120, 120, 255)),
            cell(read_frame(cap, last), f"last {t[-1]:.1f}s"),
        ]
        cap.release()
        cells.append(graph(m, t, seg, f"motion  act={seg.active_sec:.1f}s / {seg.duration_sec:.1f}s"))
        line = np.hstack(cells)
        lines.append(cv2.copyMakeBorder(line, 2, 2, 0, 0, cv2.BORDER_CONSTANT, value=(80, 80, 80)))

    for k in range(0, len(lines), ROWS_PER_IMAGE):
        path = OUT_DIR / f"{args.person}_segments_{k // ROWS_PER_IMAGE + 1}.jpg"
        cv2.imwrite(str(path), np.vstack(lines[k: k + ROWS_PER_IMAGE]), [cv2.IMWRITE_JPEG_QUALITY, 80])
        print(f"[SAVED] {path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
