"""Human QA·동작 구간 확인용 사진 만들기 (Skeleton 포함).

영상 1개 = 한 줄: [처음] [시작] [가운데] [끝] [마지막] + 움직임 그래프
- 관절은 Keypoint CSV의 Pixel 좌표로 다시 그린다 (YOLO를 다시 돌리지 않음)
- 관절 색: 초록 = Confidence >= 0.5, 빨강 = 0.5 미만 (2일차 VALID / LOW 기준)
- 그래프: 초록 선 = 시작, 빨간 선 = 끝, 회색 선 = 기준값

사용 예:
    python scripts/team03/team_review_sheets.py --subject S03
    python scripts/team03/team_review_sheets.py --clips S04_A010_T01,S05_A016_T01 --name check
"""
import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.pose_utils import COCO_KEYPOINT_NAMES, COCO_SKELETON  # noqa: E402
from team_action_segments import motion_curve  # noqa: E402

TEAM = ROOT / "data" / "team_projects" / "team03"
OUT_DIR = ROOT / "reports" / "team_projects" / "team03" / "review_sheets"
TH, CELL_W, GRAPH_W, ROWS_PER_IMAGE = 170, 300, 420, 12


def read_frame(cap, idx):
    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
    ok, frame = cap.read()
    return frame if ok else None


def draw_pose(frame, row):
    if frame is None or int(row.pose_detected) != 1:
        return frame
    pts = [(row[f"{n}_x_px"], row[f"{n}_y_px"], row[f"{n}_conf"]) for n in COCO_KEYPOINT_NAMES]
    thick = max(2, frame.shape[0] // 250)
    for a, b in COCO_SKELETON:
        cv2.line(frame, (int(pts[a][0]), int(pts[a][1])), (int(pts[b][0]), int(pts[b][1])), (255, 200, 0), thick)
    for x, y, c in pts:
        cv2.circle(frame, (int(x), int(y)), thick * 2, (0, 200, 0) if c >= 0.5 else (0, 0, 255), -1)
    return frame


def cell(frame, text, color=(255, 255, 255)):
    canvas = np.full((TH, CELL_W, 3), 255, np.uint8)
    if frame is not None:
        h, w = frame.shape[:2]
        s = min(TH / h, CELL_W / w)
        f = cv2.resize(frame, (int(w * s), int(h * s)))
        x0 = (CELL_W - f.shape[1]) // 2
        canvas[: f.shape[0], x0: x0 + f.shape[1]] = f
    cv2.rectangle(canvas, (0, 0), (CELL_W, 20), (0, 0, 0), -1)
    cv2.putText(canvas, text, (4, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)
    return canvas


def graph(m, t, seg, title):
    g = np.full((TH, GRAPH_W, 3), 255, np.uint8)
    top = float(np.nanpercentile(m, 99)) * 1.1 or 1.0
    xs = (t / t[-1] * (GRAPH_W - 10) + 5).astype(int)
    ys = (TH - 10 - np.clip(np.nan_to_num(m) / top, 0, 1) * (TH - 40)).astype(int)
    ty = int(TH - 10 - min(seg.threshold / top, 1) * (TH - 40))
    cv2.line(g, (5, ty), (GRAPH_W - 5, ty), (170, 170, 170), 1)
    cv2.polylines(g, [np.stack([xs, ys], 1).reshape(-1, 1, 2)], False, (200, 90, 0), 2)
    for sec, col in ((seg.start_sec, (0, 160, 0)), (seg.end_sec, (0, 0, 220))):
        x = int(sec / t[-1] * (GRAPH_W - 10) + 5)
        cv2.line(g, (x, 24), (x, TH - 5), col, 2)
    cv2.rectangle(g, (0, 0), (GRAPH_W, 20), (0, 0, 0), -1)
    cv2.putText(g, title, (4, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
    return g


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--subject", default=None)
    parser.add_argument("--clips", default=None, help="쉼표로 구분한 clip_id 목록")
    parser.add_argument("--name", default=None, help="출력 파일 이름 앞부분")
    args = parser.parse_args()

    manifest = pd.read_csv(TEAM / "manifests" / "keypoint_dataset_manifest.csv")
    segs = pd.read_csv(TEAM / "manifests" / "action_segments.csv").set_index("clip_id")
    if args.clips:
        manifest = manifest[manifest.clip_id.isin(args.clips.split(","))]
    if args.subject:
        manifest = manifest[manifest.subject == args.subject]
    name = args.name or args.subject or "clips"
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    lines = []
    for item in manifest.itertuples():
        df = pd.read_csv(ROOT / item.keypoint_csv)
        seg = segs.loc[item.clip_id]
        m, fps, _ = motion_curve(df)
        t = df.time_sec.to_numpy(float)
        cap = cv2.VideoCapture(str(ROOT / item.video_path))
        idx_map = {int(f): i for i, f in enumerate(df.frame_id)}
        mid = (int(seg.start_frame) + int(seg.end_frame)) // 2
        picks = [(int(df.frame_id.iloc[0]), f"{item.clip_id} 0.0s", (255, 255, 255)),
                 (int(seg.start_frame), f"START {seg.start_sec:.1f}s", (120, 255, 120)),
                 (mid, f"mid {mid / fps:.1f}s", (255, 255, 255)),
                 (int(seg.end_frame), f"END {seg.end_sec:.1f}s", (120, 120, 255)),
                 (int(df.frame_id.iloc[-1]), f"last {t[-1]:.1f}s", (255, 255, 255))]
        cells = []
        for frame_id, text, color in picks:
            frame = read_frame(cap, frame_id)
            frame = draw_pose(frame, df.iloc[idx_map[frame_id]])
            cells.append(cell(frame, text, color))
        cap.release()
        cells.append(graph(m, t, seg, f"{item.action_label} act={seg.active_sec:.1f}/{seg.duration_sec:.1f}s"))
        lines.append(cv2.copyMakeBorder(np.hstack(cells), 2, 2, 0, 0, cv2.BORDER_CONSTANT, value=(80, 80, 80)))

    for k in range(0, len(lines), ROWS_PER_IMAGE):
        path = OUT_DIR / f"{name}_{k // ROWS_PER_IMAGE + 1}.jpg"
        cv2.imwrite(str(path), np.vstack(lines[k: k + ROWS_PER_IMAGE]), [cv2.IMWRITE_JPEG_QUALITY, 80])
        print(f"[SAVED] {path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
