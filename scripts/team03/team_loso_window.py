"""예측할 때만 여러 길이의 창(Window)으로 나눠 보고, 가장 확신 높은 창의 답을 쓰는 방법 검증.

학습은 그대로(Clip마다 '처음~마지막' 동작 구간). 카메라 8초 안에 발차기 여러 번·걷기가 섞여도,
발차기 하나만 들어간 창은 학습 Clip과 모양이 같아져서 제대로 맞힐 것이라는 생각.
이 방법이 학습 영상(5명 교차 평가)에서 성능을 떨어뜨리지 않는지 확인한다.

창: 전체 + 길이 2·3·4·6초를 0.5초 간격으로. 창 안에서도 학습과 같은 규칙으로 동작 구간을 자른다.
"""
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import team_day06_baseline as tb  # noqa: E402
from team_action_segments import find_main_segment  # noqa: E402
from team_loso_segment_v2 import load_clips, rows_for_rule  # noqa: E402
from team_train_hflip_aug import AUG  # noqa: E402

SUBJECTS = ["S01", "S02", "S03", "S04", "S05"]
WIN_SEC = [2.0, 3.0, 4.0, 6.0]
STEP_SEC = 0.5


def window_vectors(c: dict, steps: int, min_valid: int):
    """Clip 하나 → [(창 설명, 1,350 벡터)]"""
    n, fps = len(c["motion"]), c["fps"]
    spans = [(0, n)]
    for w in WIN_SEC:
        wl = int(round(w * fps))
        if wl >= n:
            continue
        for a in range(0, n - wl + 1, max(1, int(round(STEP_SEC * fps)))):
            spans.append((a, a + wl))
    out = []
    for a, b in spans:
        sub = c["motion"][a:b]
        seg = find_main_segment(sub, fps, len(sub), math.inf)
        f0, f1 = c["kp_frames"][a + seg["start_idx"]], c["kp_frames"][a + seg["end_idx"]]
        keep = (c["frame_ids"] >= f0) & (c["frame_ids"] <= f1)
        if keep.sum() < min_valid:
            continue
        vec = tb.flatten_sequence(tb.resample_sequence(c["matrix"][keep], c["frame_ids"][keep], target_steps=steps))
        out.append(((a / fps, b / fps), vec))
    return out


def main() -> None:
    config = tb.load_config()
    steps, min_valid = int(config["sequence_steps"]), int(config["min_valid_frames"])
    master = pd.concat([pd.read_csv(tb.SPLIT_DIR / f"{s}.csv") for s in ("train", "val", "test")], ignore_index=True)
    orig = load_clips(master, tb.SEQ_DIR)
    aug = load_clips(pd.read_csv(AUG / "manifests" / "hflip_manifest.csv"), AUG / "sequences")
    data = rows_for_rule(orig, math.inf, steps, min_valid).drop(columns=["seg_sec"])
    adata = rows_for_rule(aug, math.inf, steps, min_valid).drop(columns=["seg_sec"])
    cols = [c for c in data.columns if c not in tb.META]

    res = []
    for s in SUBJECTS:
        train = pd.concat([data[data.subject != s], adata[adata.subject != s]], ignore_index=True)
        model, _ = tb.train_model(train, config)
        classes = np.asarray(model.classes_)
        for c in orig:
            if c["item"]["subject"] != s:
                continue
            wins = window_vectors(c, steps, min_valid)
            proba = model.predict_proba(np.stack([v for _, v in wins]))
            whole = classes[proba[0].argmax()]
            best = int(proba.max(1).argmax())
            res.append({"subject": s, "true": c["item"]["action_label"], "whole": whole,
                        "window_max": classes[proba[best].argmax()], "n_windows": len(wins),
                        "whole_conf": proba[0].max(), "win_conf": proba[best].max()})
    df = pd.DataFrame(res)
    for col in ("whole", "window_max"):
        df[f"{col}_ok"] = df[col] == df["true"]
    print(f"전체 하나로 예측 (지금):      {df.whole_ok.mean():.4f}  Confidence {df.whole_conf.mean():.3f}")
    print(f"창 중 가장 확신 높은 것:     {df.window_max_ok.mean():.4f}  Confidence {df.win_conf.mean():.3f}")
    print("\n사람별:")
    print(df.groupby("subject")[["whole_ok", "window_max_ok"]].mean().round(3).T.to_string())
    per = df.groupby("true")[["whole_ok", "window_max_ok"]].mean().round(3)
    per["차이"] = per.window_max_ok - per.whole_ok
    print("\n동작별 (차이 큰 순):")
    print(per.sort_values("차이").to_string())


if __name__ == "__main__":
    main()
