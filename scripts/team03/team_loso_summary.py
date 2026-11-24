"""결정적인 순간 특징을 더하면 좋아지는지 5명 교차 평가로 비교한다.

비교 (모두 지금 서비스와 같은 동작 구간: 기준선 최소값 0.4, 45-D, 후면 + 측면 반전)
- seq          : 30 Step 시간표 1,350개 (지금 서비스)
- seq+summary  : 시간표 + 결정적인 순간 225개 = 1,575개
- summary      : 결정적인 순간 225개만
출력: reports/team_projects/team03/day06/loso/summary_features.csv (Git 제외), /tmp/vap_seg/summary_<방식>.joblib
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
from team_summary_features import summary_feature_names, summary_features  # noqa: E402
from team_train_hflip_aug import AUG  # noqa: E402

SUBJECTS = ["S01", "S02", "S03", "S04", "S05"]
FLOOR = 0.4


def build(clips, steps, min_valid, floor=FLOOR):
    rows = []
    for c in clips:
        seg = find_main_segment(c["motion"], c["fps"], len(c["motion"]), math.inf, floor)
        f0, f1 = c["kp_frames"][seg["start_idx"]], c["kp_frames"][seg["end_idx"]]
        keep = (c["frame_ids"] >= f0) & (c["frame_ids"] <= f1)
        if keep.sum() < min_valid:
            continue
        m, ids = c["matrix"][keep], c["frame_ids"][keep]
        seq = tb.flatten_sequence(tb.resample_sequence(m, ids, target_steps=steps))
        summ = summary_features(m, c["names"])
        item = c["item"]
        row = {k: item[k] for k in ("clip_id", "subject", "action_id", "action_label", "take", "video_path")}
        row["valid_frames"] = int(keep.sum())
        row.update(zip([f"t{t:02d}_{n}" for t in range(steps) for n in c["names"]], map(float, seq)))
        row.update(zip(summary_feature_names(c["names"]), map(float, summ)))
        rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    config = tb.load_config()
    steps, min_valid = int(config["sequence_steps"]), int(config["min_valid_frames"])
    master = pd.concat([pd.read_csv(tb.SPLIT_DIR / f"{s}.csv") for s in ("train", "val", "test")], ignore_index=True)
    data = build(load_clips(master, tb.SEQ_DIR), steps, min_valid)
    adata = build(load_clips(pd.read_csv(AUG / "manifests" / "hflip_manifest.csv"), AUG / "sequences"), steps, min_valid)
    seq_cols = [c for c in data.columns if c.startswith("t") and c[1:3].isdigit()]
    sum_cols = [c for c in data.columns if c.startswith(("sum_", "peak_"))]
    variants = {"seq": seq_cols, "seq+summary": seq_cols + sum_cols, "summary": sum_cols}
    out, per = [], []
    for name, cols in variants.items():
        accs, f1s, kicks, throws, confs = [], [], [], [], []
        for s in SUBJECTS:
            train = pd.concat([data[data.subject != s], adata[adata.subject != s]], ignore_index=True)
            model, _ = tb.train_model(train[tb.META[:5] + ["valid_frames", "video_path"] + cols], config)
            te = data[data.subject == s]
            proba = model.predict_proba(te[cols].to_numpy(float))
            pred = np.asarray(model.classes_)[proba.argmax(1)]
            y = te.action_label.astype(str).to_numpy()
            accs.append(accuracy_score(y, pred))
            f1s.append(f1_score(y, pred, average="macro", zero_division=0))
            kicks.append(float((pred[y == "front_kick"] == "front_kick").mean()))
            throws.append(float((pred[y == "throw_one_hand"] == "throw_one_hand").mean()))
            confs.append(float(proba.max(1).mean()))
            per.append({"variant": name, "subject": s, "acc": accs[-1]})
            for a in sorted(set(y)):
                per.append({"variant": name, "subject": "action:" + a, "acc": float((pred[y == a] == a).mean())})
        full = pd.concat([data, adata], ignore_index=True)
        model, _ = tb.train_model(full[tb.META[:5] + ["valid_frames", "video_path"] + cols], config)
        joblib.dump({"model": model, "cols": cols}, f"/tmp/vap_seg/summary_{name}.joblib")
        out.append({"variant": name, "features": len(cols), "acc": np.mean(accs), "acc_min": np.min(accs),
                    "f1": np.mean(f1s), "kick": np.mean(kicks), "throw": np.mean(throws), "conf": np.mean(confs)})
    df = pd.DataFrame(out).round(4)
    print(df.to_string(index=False))
    p = pd.DataFrame(per)
    subj = p[~p.subject.str.startswith("action:")].pivot(index="variant", columns="subject", values="acc")
    print("\n사람별:\n" + subj.round(3).to_string())
    act = p[p.subject.str.startswith("action:")].groupby(["variant", "subject"]).acc.mean().unstack(0)
    act.index = act.index.str.replace("action:", "")
    act["차이(seq+summary - seq)"] = act["seq+summary"] - act["seq"]
    print("\n동작별 (5명 평균):\n" + act.round(3).sort_values("차이(seq+summary - seq)").to_string())
    df.to_csv(tb.REPORT_DIR / "loso" / "summary_features.csv", index=False, encoding="utf-8-sig")


if __name__ == "__main__":
    main()
