"""동작 구간 규칙 비교: '처음~마지막'(지금) vs '가장 큰 움직임 덩어리'(merge 간격별).

카메라 8초 녹화에서 걸어 들어오기·발차기 여러 번이 한 구간으로 묶여 앞차기가 '물건 던지기'로 나왔다.
규칙을 바꾸면 학습 데이터 구간도 같이 바뀌므로, 학습·예측을 같은 규칙으로 맞춰 5명 교차 평가로 비교한다.

구성: 45-D, 정면·측면·후면 + 측면 반전 (지금 서비스 모델과 같음)
출력: reports/team_projects/team03/day06/loso/segment_v2_summary.csv (Git 제외 CSV)
      /tmp/vap_seg/<rule>.joblib  (5명 전체로 학습한 비교용 모델, 카메라 영상 확인용 임시 파일)
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
from team_action_segments import find_main_segment, motion_curve  # noqa: E402
from team_train_hflip_aug import AUG  # noqa: E402

ROOT = tb.ROOT
OUT = tb.REPORT_DIR / "loso"
SUBJECTS = ["S01", "S02", "S03", "S04", "S05"]
RULES = {"first_last": math.inf, "main_gap0.5": 0.5, "main_gap1.0": 1.0, "main_gap2.0": 2.0}


def load_clips(manifest: pd.DataFrame, seq_dir: Path) -> list[dict]:
    """Clip마다 Frame Feature(45-D, 전체)·frame_id·움직임 곡선을 한 번만 계산해 둔다."""
    out = []
    old = tb.SEQ_DIR
    tb.SEQ_DIR = seq_dir
    try:
        for _, item in manifest.iterrows():
            matrix, frame_ids, names = tb.clip_frames(item, {"team_features": True})
            kp = pd.read_csv(ROOT / item["keypoint_csv"])
            m, fps, _ = motion_curve(kp)
            out.append({"item": item, "matrix": matrix, "frame_ids": np.asarray(frame_ids), "names": names,
                        "motion": m, "fps": fps, "kp_frames": kp["frame_id"].to_numpy()})
    finally:
        tb.SEQ_DIR = old
    return out


def rows_for_rule(clips: list[dict], gap: float, steps: int, min_valid: int) -> pd.DataFrame:
    rows = []
    for c in clips:
        seg = find_main_segment(c["motion"], c["fps"], len(c["motion"]), gap)
        f0, f1 = c["kp_frames"][seg["start_idx"]], c["kp_frames"][seg["end_idx"]]
        keep = (c["frame_ids"] >= f0) & (c["frame_ids"] <= f1)
        matrix, frame_ids = c["matrix"][keep], c["frame_ids"][keep]
        if len(matrix) < min_valid:
            continue
        values = tb.flatten_sequence(tb.resample_sequence(matrix, frame_ids, target_steps=steps))
        cols = [f"t{t:02d}_{n}" for t in range(steps) for n in c["names"]]
        item = c["item"]
        row = {k: item[k] for k in ("clip_id", "subject", "action_id", "action_label", "take", "video_path")}
        row["valid_frames"] = len(matrix)
        row["seg_sec"] = (f1 - f0) / c["fps"]
        row.update(zip(cols, map(float, values)))
        rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    config = tb.load_config()
    steps, min_valid = int(config["sequence_steps"]), int(config["min_valid_frames"])
    master = pd.concat([pd.read_csv(tb.SPLIT_DIR / f"{s}.csv") for s in ("train", "val", "test")], ignore_index=True)
    orig = load_clips(master, tb.SEQ_DIR)
    aug = load_clips(pd.read_csv(AUG / "manifests" / "hflip_manifest.csv"), AUG / "sequences")
    tmp = Path("/tmp/vap_seg")
    tmp.mkdir(exist_ok=True)

    results = []
    for rule, gap in RULES.items():
        data = rows_for_rule(orig, gap, steps, min_valid)
        adata = rows_for_rule(aug, gap, steps, min_valid)
        cols = [c for c in data.columns if c not in tb.META and c != "seg_sec"]
        for s in SUBJECTS:
            train = pd.concat([data[data.subject != s], adata[adata.subject != s]], ignore_index=True)
            model, _ = tb.train_model(train.drop(columns=["seg_sec"]), config)
            te = data[data.subject == s]
            proba = model.predict_proba(te[cols].to_numpy(float))
            pred = np.asarray(model.classes_)[proba.argmax(1)]
            y = te.action_label.astype(str).to_numpy()
            results.append({"rule": rule, "subject": s, "accuracy": accuracy_score(y, pred),
                            "macro_f1": f1_score(y, pred, average="macro", zero_division=0),
                            "kick_acc": float((pred[y == "front_kick"] == "front_kick").mean()),
                            "conf": float(proba.max(1).mean())})
        full = pd.concat([data, adata], ignore_index=True).drop(columns=["seg_sec"])
        model, _ = tb.train_model(full, config)
        joblib.dump(model, tmp / f"{rule}.joblib")
        print(f"{rule:12s} 학습 구간 길이 중간값 {data.seg_sec.median():.2f}초 (원래 Clip 대비 줄어든 Clip "
              f"{int((data.seg_sec < pd.Series([o['kp_frames'][-1] / o['fps'] for o in orig]).median() * 0.5).sum())}개)")

    df = pd.DataFrame(results)
    summary = df.groupby("rule").agg(acc=("accuracy", "mean"), acc_min=("accuracy", "min"), f1=("macro_f1", "mean"),
                                     kick=("kick_acc", "mean"), conf=("conf", "mean")).round(4)
    summary = summary.loc[list(RULES)]
    print()
    print(summary.to_string())
    print("\n사람별 정확도:")
    print(df.pivot(index="rule", columns="subject", values="accuracy").loc[list(RULES)].round(3).to_string())
    OUT.mkdir(parents=True, exist_ok=True)
    summary.to_csv(OUT / "segment_v2_summary.csv", encoding="utf-8-sig")


if __name__ == "__main__":
    main()
