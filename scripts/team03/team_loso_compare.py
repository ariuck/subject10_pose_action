"""5명을 한 명씩 Test로 빼서(Leave-One-Subject-Out) 학습 데이터 구성을 비교한다.

사람 한 명(34~51개)으로 낸 Validation·Test 점수는 사람에 따라 크게 흔들렸다.
그래서 5번 돌려 평균을 내고, 어느 구성이 사람과 상관없이 나은지 확인한다.

- 회차마다 Train = 나머지 4명, Test = 빠진 1명의 원본 51개 (정면·측면·후면)
- 비교 구성: 2view(T01+T02) / 3view(+T03) / 3view_hflip(+Train 사람의 T02 반전)
- 반전 Clip은 Train 사람 것만 학습에 넣는다 (Test 사람의 반전본은 사용하지 않음)
- 조건: 교재 38-D, 3팀 45-D / 모델 설정은 6일차와 같다

출력: reports/team_projects/team03/day06/loso/loso_folds.csv, loso_summary.csv

사용: python scripts/team03/team_loso_compare.py
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
CONDS = ["baseline_38d", "team_45d"]
SETS = ["2view", "3view", "3view_hflip"]


def main() -> None:
    config = tb.load_config()
    OUT.mkdir(parents=True, exist_ok=True)
    master = pd.concat([pd.read_csv(tb.SPLIT_DIR / f"{s}.csv") for s in ("train", "val", "test")],
                       ignore_index=True)
    aug_m = pd.read_csv(AUG / "manifests" / "hflip_manifest.csv")

    rows = []
    for name in CONDS:
        cond = tb.CONDITIONS[name]
        data = build_rows(master, cond, config, tb.SEQ_DIR)          # 원본 255개 한 번만 계산
        aug = build_rows(aug_m, cond, config, AUG / "sequences")      # 반전 85개
        cols = [c for c in data.columns if c not in tb.META]
        for test_s in SUBJECTS:
            tr_all = data[data.subject != test_s]
            te = data[data.subject == test_s]
            sets = {
                "2view": tr_all[tr_all["take"].isin(["T01", "T02"])],
                "3view": tr_all,
                "3view_hflip": pd.concat([tr_all, aug[aug.subject != test_s]], ignore_index=True),
            }
            for label in SETS:
                model, mcols = tb.train_model(sets[label], config)
                assert mcols == cols
                proba = model.predict_proba(te[cols].to_numpy(float))
                pred = np.asarray(model.classes_)[proba.argmax(1)]
                y = te["action_label"].astype(str).to_numpy()
                row = {"condition": name, "train_set": label, "test_subject": test_s,
                       "train_clips": len(sets[label]), "test_clips": len(te),
                       "accuracy": round(accuracy_score(y, pred), 4),
                       "macro_f1": round(f1_score(y, pred, average="macro", zero_division=0), 4),
                       "mean_conf": round(float(proba.max(1).mean()), 4)}
                for take, view in (("T01", "front"), ("T02", "side"), ("T03", "back")):
                    m = te["take"].to_numpy() == take
                    row[f"{view}_acc"] = round(float((pred[m] == y[m]).mean()), 4)
                rows.append(row)
                print(f"{name:13s} {label:12s} test={test_s}  acc {row['accuracy']:.3f}  f1 {row['macro_f1']:.3f}")

    folds = pd.DataFrame(rows)
    folds.to_csv(OUT / "loso_folds.csv", index=False, encoding="utf-8-sig")
    summary = (folds.groupby(["condition", "train_set"])
               .agg(acc_mean=("accuracy", "mean"), acc_std=("accuracy", "std"),
                    f1_mean=("macro_f1", "mean"), f1_std=("macro_f1", "std"),
                    acc_min=("accuracy", "min"), acc_max=("accuracy", "max"),
                    front=("front_acc", "mean"), side=("side_acc", "mean"), back=("back_acc", "mean"),
                    conf=("mean_conf", "mean"))
               .round(4).reset_index())
    summary.to_csv(OUT / "loso_summary.csv", index=False, encoding="utf-8-sig")
    print()
    print(summary.to_string(index=False))
    print(f"\n[SAVED] {(OUT / 'loso_summary.csv').relative_to(ROOT)}")


if __name__ == "__main__":
    main()
