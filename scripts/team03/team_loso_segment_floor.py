"""동작 구간 기준선에 최소값(floor)을 두면 좋아지는지 5명 교차 평가로 비교한다.

카메라 영상은 서 있는 시간이 길어 기준선이 낮게 잡히고, 서 있을 때 흔들림(0.3~0.5)까지 동작 구간이 됐다.
학습 데이터 구간도 같은 규칙으로 다시 잘라 학습·예측을 맞춘다. 규칙은 '처음~마지막' 그대로 두고 기준선만 바꾼다.

출력: reports/team_projects/team03/day06/loso/segment_floor_summary.csv (Git 제외 CSV)
      /tmp/vap_seg/floor_<값>.joblib (5명 전체로 학습한 비교용 모델)
"""
import math
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
import team_day06_baseline as tb  # noqa: E402
from team_action_segments import find_main_segment  # noqa: E402
from team_loso_segment_v2 import load_clips  # noqa: E402
from team_train_hflip_aug import AUG  # noqa: E402

SUBJECTS = ["S01", "S02", "S03", "S04", "S05"]
FLOORS = [0.0, 0.4, 0.5, 0.6, 0.8]


def rows_for_floor(clips, floor, steps, min_valid):
    rows = []
    for c in clips:
        seg = find_main_segment(c["motion"], c["fps"], len(c["motion"]), math.inf, floor)
        f0, f1 = c["kp_frames"][seg["start_idx"]], c["kp_frames"][seg["end_idx"]]
        keep = (c["frame_ids"] >= f0) & (c["frame_ids"] <= f1)
        if keep.sum() < min_valid:
            continue
        values = tb.flatten_sequence(tb.resample_sequence(c["matrix"][keep], c["frame_ids"][keep], target_steps=steps))
        item = c["item"]
        row = {k: item[k] for k in ("clip_id", "subject", "action_id", "action_label", "take", "video_path")}
        row["valid_frames"] = int(keep.sum())
        row.update(zip([f"t{t:02d}_{n}" for t in range(steps) for n in c["names"]], map(float, values)))
        rows.append((row, (f1 - f0) / c["fps"]))
    return pd.DataFrame([r for r, _ in rows]), np.array([s for _, s in rows])


def main() -> None:
    config = tb.load_config()
    steps, min_valid = int(config["sequence_steps"]), int(config["min_valid_frames"])
    master = pd.concat([pd.read_csv(tb.SPLIT_DIR / f"{s}.csv") for s in ("train", "val", "test")], ignore_index=True)
    orig = load_clips(master, tb.SEQ_DIR)
    aug = load_clips(pd.read_csv(AUG / "manifests" / "hflip_manifest.csv"), AUG / "sequences")
    base_len = None
    out, per_subject = [], []
    for floor in FLOORS:
        data, seg_len = rows_for_floor(orig, floor, steps, min_valid)
        adata, _ = rows_for_floor(aug, floor, steps, min_valid)
        if base_len is None:
            base_len = seg_len
        cols = [c for c in data.columns if c not in tb.META]
        accs, f1s, kicks, confs = [], [], [], []
        for s in SUBJECTS:
            model, _ = tb.train_model(pd.concat([data[data.subject != s], adata[adata.subject != s]], ignore_index=True), config)
            te = data[data.subject == s]
            proba = model.predict_proba(te[cols].to_numpy(float))
            pred = np.asarray(model.classes_)[proba.argmax(1)]
            y = te.action_label.astype(str).to_numpy()
            accs.append(accuracy_score(y, pred))
            f1s.append(f1_score(y, pred, average="macro", zero_division=0))
            kicks.append(float((pred[y == "front_kick"] == "front_kick").mean()))
            confs.append(float(proba.max(1).mean()))
            per_subject.append({"floor": floor, "subject": s, "acc": accs[-1]})
        model, _ = tb.train_model(pd.concat([data, adata], ignore_index=True), config)
        Path("/tmp/vap_seg").mkdir(exist_ok=True)
        joblib.dump(model, f"/tmp/vap_seg/floor_{floor}.joblib")
        out.append({"floor": floor, "acc": np.mean(accs), "acc_min": np.min(accs), "f1": np.mean(f1s),
                    "kick": np.mean(kicks), "conf": np.mean(confs), "seg_median_sec": np.median(seg_len),
                    "changed_clips": int((np.abs(seg_len - base_len) > 0.05).sum())})
    df = pd.DataFrame(out).round(4)
    print(df.to_string(index=False))
    print("\n사람별:")
    print(pd.DataFrame(per_subject).pivot(index="floor", columns="subject", values="acc").round(3).to_string())
    (tb.REPORT_DIR / "loso").mkdir(parents=True, exist_ok=True)
    df.to_csv(tb.REPORT_DIR / "loso" / "segment_floor_summary.csv", index=False, encoding="utf-8-sig")


if __name__ == "__main__":
    main()
