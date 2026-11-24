"""후면(T03)을 추가하고 측면(T02) 좌우 반전까지 넣었을 때 성능을 비교한다.

학습 데이터 구성 (Train 사람 S01~S03, 동작마다 사람 1명당)
- 2view        : 정면 T01 + 측면 T02                       = 2개  (102 Clip)
- 3view        : 정면 T01 + 측면 T02 + 후면 T03             = 3개  (153 Clip)
- 3view_hflip  : 3view + 측면 좌우 반전 T02_hflip            = 4개  (204 Clip)

평가는 원본 영상만 사용한다 (반전 Clip은 학습에만 넣음).
- Validation S04 51개 (T01·T02·T03), Test S05 51개
- 방향별 Test 정답 수 (정면·측면·후면 각 17개)
- 조건 선택 기준(Test 전에 정함): Validation Macro F1이 가장 높은 조합

모델 설정은 6일차와 같다 (Random Forest 300 Tree, class_weight balanced, 30 Step).

사용: python scripts/team03/team_train_3view_hflip.py
"""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import team_day06_baseline as tb  # noqa: E402
from team_train_hflip_aug import AUG, build_rows, save_model  # noqa: E402

ROOT = tb.ROOT
OUT = tb.REPORT_DIR / "three_view"
CONDS = ["baseline_38d", "team_45d"]
VIEWS = {"T01": "front", "T02": "side", "T03": "back"}


def main() -> None:
    config = tb.load_config()
    aug_manifest = pd.read_csv(AUG / "manifests" / "hflip_manifest.csv")
    aug_train_m = aug_manifest[aug_manifest["split"] == "train"]
    OUT.mkdir(parents=True, exist_ok=True)

    results, models = [], {}
    for name in CONDS:
        cond = tb.CONDITIONS[name]
        train = tb.build_split("train", cond, config)
        val = tb.build_split("val", cond, config)
        test = tb.build_split("test", cond, config)
        aug_train = build_rows(aug_train_m, cond, config, AUG / "sequences")
        sets = {
            "2view": train[train["take"].isin(["T01", "T02"])],
            "3view": train,
            "3view_hflip": pd.concat([train, aug_train], ignore_index=True),
        }
        for label, tr in sets.items():
            model, cols = tb.train_model(tr, config)
            tag = f"{name}_{label}"
            v = tb.evaluate(model, cols, val, tag, "validation", OUT)
            t = tb.evaluate(model, cols, test, tag, "test", OUT)
            row = {"condition": name, "train_set": label, "train_clips": len(tr),
                   "val_acc": v["accuracy"], "val_f1": v["macro_f1"], "val_correct": v["clips"] - v["wrong"],
                   "test_acc": t["accuracy"], "test_f1": t["macro_f1"], "test_correct": t["clips"] - t["wrong"],
                   "test_mean_conf": round(float(t["pred"]["confidence"].mean()), 4)}
            pred = t["pred"]
            for take, view in VIEWS.items():
                p = pred[pred["take"] == take]
                row[f"test_{view}_correct"] = int(p["correct"].sum())
            # 이전 실험과 비교용: 정면·측면만 Test (34개)
            p2 = pred[pred["take"].isin(["T01", "T02"])]
            row["test_front_side_correct"] = int(p2["correct"].sum())
            results.append(row)
            models[(name, label)] = (model, cols, tr, len(aug_train) if label.endswith("hflip") else 0)
            print(f"{name:13s} {label:12s} train={len(tr):3d}  val {row['val_correct']}/51 F1 {row['val_f1']:.4f}  "
                  f"test {row['test_correct']}/51 F1 {row['test_f1']:.4f}  "
                  f"(front {row['test_front_correct']} / side {row['test_side_correct']} / back {row['test_back_correct']})")

    table = pd.DataFrame(results)
    table.to_csv(OUT / "three_view_comparison.csv", index=False, encoding="utf-8-sig")
    best = table.sort_values(["val_f1", "val_acc"], ascending=False).iloc[0]
    print(f"\nValidation 기준 최고: {best['condition']} + {best['train_set']} (val F1 {best['val_f1']:.4f})")

    for name in CONDS:
        model, cols, tr, n_aug = models[(name, "3view_hflip")]
        path = save_model(model, cols, tr, config, name, n_aug, folder=f"{name}_3view_hflip")
        print(f"[SAVED] {path.relative_to(ROOT)}")
    print(f"[SAVED] {(OUT / 'three_view_comparison.csv').relative_to(ROOT)}")


if __name__ == "__main__":
    main()
