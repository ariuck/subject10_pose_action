"""동작 구간만 잘라 쓰면 좋아지는지 5명 교차 평가(LOSO)로 확인한다.

서비스에서 카메라 8초에 대기 시간이 섞여 동작이 묻히는 문제가 있어 '동작 구간 자르기'를 검토했다.
6일차 실험(38-D)에서는 구간만 쓰면 좋아졌지만, 지금 서비스 모델(45-D, 영상 전체로 학습)에서는 확인한 적이 없다.

비교 (학습 구성은 최종 모델과 같은 3view_hflip, 45-D)
- A: 전체로 학습 → 전체로 예측   (지금 서비스)
- B: 전체로 학습 → 구간만 예측   (서비스에서 예측할 때만 자르기)
- C: 구간으로 학습 → 구간만 예측 (모델을 다시 학습)
동작 구간 = action_segments.csv의 in_action (관절 움직임 크기로 자동 계산, 앞뒤 0.2초 여유)

출력: reports/team_projects/team03/day06/loso/segment_folds.csv, segment_summary.csv
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
import team_day06_baseline as tb  # noqa: E402
from team_train_hflip_aug import AUG, build_rows  # noqa: E402

ROOT = tb.ROOT
OUT = tb.REPORT_DIR / "loso"
SUBJECTS = ["S01", "S02", "S03", "S04", "S05"]
tb.CONDITIONS["team_45d_segment"] = {"team_features": True, "segment_only": True}


def main() -> None:
    config = tb.load_config()
    OUT.mkdir(parents=True, exist_ok=True)
    master = pd.concat([pd.read_csv(tb.SPLIT_DIR / f"{s}.csv") for s in ("train", "val", "test")],
                       ignore_index=True)
    aug_m = pd.read_csv(AUG / "manifests" / "hflip_manifest.csv")

    data = {}
    for key, name in (("whole", "team_45d"), ("segment", "team_45d_segment")):
        cond = tb.CONDITIONS[name]
        data[key] = (build_rows(master, cond, config, tb.SEQ_DIR), build_rows(aug_m, cond, config, AUG / "sequences"))
    cols = [c for c in data["whole"][0].columns if c not in tb.META]

    plans = {"A_whole_whole": ("whole", "whole"), "B_whole_segment": ("whole", "segment"),
             "C_segment_segment": ("segment", "segment")}
    rows = []
    for test_s in SUBJECTS:
        models = {}
        for key in ("whole", "segment"):
            orig, aug = data[key]
            train = pd.concat([orig[orig.subject != test_s], aug[aug.subject != test_s]], ignore_index=True)
            models[key], mcols = tb.train_model(train, config)
            assert mcols == cols
        for plan, (train_key, test_key) in plans.items():
            te = data[test_key][0]
            te = te[te.subject == test_s]
            proba = models[train_key].predict_proba(te[cols].to_numpy(float))
            pred = np.asarray(models[train_key].classes_)[proba.argmax(1)]
            y = te["action_label"].astype(str).to_numpy()
            row = {"plan": plan, "test_subject": test_s, "test_clips": len(te),
                   "accuracy": round(accuracy_score(y, pred), 4),
                   "macro_f1": round(f1_score(y, pred, average="macro", zero_division=0), 4),
                   "mean_conf": round(float(proba.max(1).mean()), 4)}
            rows.append(row)
            print(f"{plan:18s} test={test_s}  acc {row['accuracy']:.3f}  f1 {row['macro_f1']:.3f}  conf {row['mean_conf']:.3f}")

    folds = pd.DataFrame(rows)
    folds.to_csv(OUT / "segment_folds.csv", index=False, encoding="utf-8-sig")
    summary = (folds.groupby("plan")
               .agg(acc_mean=("accuracy", "mean"), acc_std=("accuracy", "std"),
                    f1_mean=("macro_f1", "mean"), acc_min=("accuracy", "min"), conf=("mean_conf", "mean"))
               .round(4).reset_index())
    summary.to_csv(OUT / "segment_summary.csv", index=False, encoding="utf-8-sig")
    print()
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
