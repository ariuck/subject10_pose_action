"""3팀 측면(T02) Keypoint를 좌우 반전해 증강 Clip을 만든다.

측면 카메라는 사람이 한쪽 방향만 보고 찍혔다. 반대쪽에서 찍은 것처럼 보이도록
영상을 다시 Pose 추론하지 않고 Keypoint 단계에서 뒤집는다.

- x 좌표: x_px → 영상 너비 − x_px, x_norm → 1 − x_norm
- 왼쪽·오른쪽 관절 이름을 서로 바꾼다 (반전하면 왼팔이 오른팔 자리에 오므로)
- BBox: x1, x2 → 너비 − x2, 너비 − x1
- 동작 구간(in_action)은 움직임 크기로 정했으므로 원본 구간을 그대로 쓴다

증강 Clip은 Train(S01~S03)에만 학습용으로 쓴다. Validation·Test 사람의 반전 Clip은
"반대 방향 측면" 평가용으로만 만들고 학습에는 넣지 않는다.

출력: data/team_projects/team03/augmented/hflip/
  keypoints/<S>/<label>/<clip_id>_hflip.csv, sequences/, manifests/

사용: python scripts/team03/team_augment_hflip.py
"""
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
TEAM = ROOT / "data" / "team_projects" / "team03"
OUT = TEAM / "augmented" / "hflip"
REL_OUT = OUT.relative_to(ROOT).as_posix()

COCO = ["nose", "left_eye", "right_eye", "left_ear", "right_ear", "left_shoulder", "right_shoulder",
        "left_elbow", "right_elbow", "left_wrist", "right_wrist", "left_hip", "right_hip",
        "left_knee", "right_knee", "left_ankle", "right_ankle"]
SUFFIXES = ["x_px", "y_px", "x_norm", "y_norm", "conf"]
SPLIT_OF = {"train": "train", "val": "val", "test": "test"}


def flip_keypoints(df: pd.DataFrame, width: float) -> pd.DataFrame:
    out = df.copy()
    for name in COCO:
        for col, fn in ((f"{name}_x_px", lambda v: width - v), (f"{name}_x_norm", lambda v: 1.0 - v)):
            v = pd.to_numeric(df[col], errors="coerce")
            out[col] = fn(v).round(6)
    # 왼쪽 ↔ 오른쪽 관절 이름 바꾸기
    for name in COCO:
        if name.startswith("left_"):
            other = "right_" + name[len("left_"):]
            for s in SUFFIXES:
                a, b = f"{name}_{s}", f"{other}_{s}"
                out[a], out[b] = out[b].copy(), out[a].copy()
    x1 = pd.to_numeric(df["box_x1"], errors="coerce")
    x2 = pd.to_numeric(df["box_x2"], errors="coerce")
    out["box_x1"], out["box_x2"] = (width - x2).round(2), (width - x1).round(2)
    return out


def main() -> None:
    kp_manifest = pd.read_csv(TEAM / "manifests" / "keypoint_dataset_manifest.csv").set_index("clip_id")
    segments = pd.read_csv(TEAM / "manifests" / "action_segments.csv").set_index("clip_id")

    rows, seg_rows = [], []
    for split in ("train", "val", "test"):
        manifest = pd.read_csv(TEAM / "splits" / f"{split}.csv")
        for item in manifest[manifest["take"] == "T02"].itertuples():
            width = float(kp_manifest.loc[item.clip_id, "width"])
            src = pd.read_csv(ROOT / item.keypoint_csv)
            flipped = flip_keypoints(src, width)
            clip = f"{item.clip_id}_hflip"
            flipped["clip_id"] = clip
            dst = OUT / "keypoints" / item.subject / item.action_label / f"{clip}.csv"
            dst.parent.mkdir(parents=True, exist_ok=True)
            flipped.to_csv(dst, index=False, encoding="utf-8-sig")

            row = item._asdict()
            row.pop("Index")
            row.update({"clip_id": clip, "keypoint_csv": dst.relative_to(ROOT).as_posix(),
                        "source_clip_id": item.clip_id, "augmentation": "hflip", "split": split})
            rows.append(row)
            seg = segments.loc[item.clip_id].to_dict()
            seg_rows.append({"clip_id": clip, **seg, "method": f"{seg.get('method', '')}+copied_from_source"})

    (OUT / "manifests").mkdir(parents=True, exist_ok=True)
    manifest_path = OUT / "manifests" / "hflip_manifest.csv"
    segment_path = OUT / "manifests" / "hflip_action_segments.csv"
    pd.DataFrame(rows).to_csv(manifest_path, index=False, encoding="utf-8-sig")
    pd.DataFrame(seg_rows).to_csv(segment_path, index=False, encoding="utf-8-sig")
    table = pd.DataFrame(rows)
    print(f"[SAVED] {manifest_path.relative_to(ROOT)}  clips={len(table)}")
    print(table.groupby("split").size().to_string())

    # 원본과 같은 코드로 38-D + 팀 Feature Sequence 계산
    subprocess.run([sys.executable, str(ROOT / "scripts" / "team03" / "team_build_feature_sequences.py"),
                    "--manifest-path", manifest_path.relative_to(ROOT).as_posix(),
                    "--segments-path", segment_path.relative_to(ROOT).as_posix(),
                    "--output-dir", f"{REL_OUT}/sequences",
                    "--sequence-manifest", f"{REL_OUT}/manifests/hflip_sequence_manifest.csv"],
                   check=True, cwd=ROOT)


if __name__ == "__main__":
    main()
