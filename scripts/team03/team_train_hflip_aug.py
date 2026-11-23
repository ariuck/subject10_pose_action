"""측면(T02) 좌우 반전 증강을 Train에 넣었을 때 성능이 어떻게 바뀌는지 비교한다.

- 비교: 원본 Train 102개  vs  원본 102개 + 반전 51개 (Train 사람 S01~S03의 측면만)
- 조건: 교재 38-D, 3팀 45-D (모델 설정은 6일차와 같음)
- 평가: Validation(S04 34개), Test(S05 34개),
        반전 Test(S05 측면 17개를 뒤집은 것 = 반대쪽에서 찍은 측면을 가정)
- 모델 저장: artifacts/team03/day06/<조건>_hflip/ (교재 35번과 같은 세 파일)

사용: python scripts/team03/team_train_hflip_aug.py
"""
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import team_day06_baseline as tb  # noqa: E402

ROOT = tb.ROOT
AUG = ROOT / "data" / "team_projects" / "team03" / "augmented" / "hflip"
OUT = tb.REPORT_DIR / "augment_hflip"
CONDS = ["baseline_38d", "team_45d"]


def build_rows(manifest: pd.DataFrame, cond: dict, config: dict, seq_dir: Path) -> pd.DataFrame:
    """tb.build_split과 같은 계산을 주어진 Manifest에 적용한다 (증강 Clip은 Sequence 폴더만 다름)."""
    steps = int(config["sequence_steps"])
    rows = []
    old = tb.SEQ_DIR
    tb.SEQ_DIR = seq_dir
    try:
        for _, item in manifest.iterrows():
            matrix, frame_ids, names = tb.clip_frames(item, cond)
            if len(matrix) < int(config["min_valid_frames"]):
                print(f"  [SKIP] {item['clip_id']} valid={len(matrix)}")
                continue
            values = tb.flatten_sequence(tb.resample_sequence(matrix, frame_ids, target_steps=steps))
            cols = [f"t{t:02d}_{n}" for t in range(steps) for n in names]
            row = {k: item[k] for k in ("clip_id", "subject", "action_id", "action_label", "take", "video_path")}
            row["valid_frames"] = len(matrix)
            row.update(zip(cols, map(float, values)))
            rows.append(row)
    finally:
        tb.SEQ_DIR = old
    return pd.DataFrame(rows)


def save_model(model, cols, train: pd.DataFrame, config: dict, name: str, n_aug: int) -> Path:
    out = tb.ARTIFACT_DIR / f"{name}_hflip"
    out.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, out / "rf_action_baseline.joblib")
    (out / "rf_feature_columns.json").write_text(json.dumps(cols, indent=2), encoding="utf-8")
    cond = tb.CONDITIONS[name]
    info = {
        "train_clips": int(len(train)),
        "subjects": sorted(train["subject"].unique().tolist()),
        "classes": [str(c) for c in model.classes_],
        "feature_count": len(cols),
        "sequence_steps": int(config["sequence_steps"]),
        "min_valid_frames": int(config["min_valid_frames"]),
        "n_estimators": int(config["n_estimators"]),
        "max_depth": config["max_depth"],
        "random_state": int(config["random_state"]),
        "condition": name,
        "frame_feature_count": len(cols) // int(config["sequence_steps"]),
        "team_features": tb.TEAM_FEATURES if cond.get("team_features") else [],
        "class_weight": "balanced",
        "views": sorted(train["take"].unique().tolist()),
        "augmentation": "hflip of T02 (side) train clips",
        "augmented_clips": n_aug,
        "action_classes_csv": "configs/team03_action_classes.csv",
    }
    (out / "rf_training_info.json").write_text(json.dumps(info, indent=2, ensure_ascii=False), encoding="utf-8")
    return out


def main() -> None:
    config = tb.load_config()
    aug_manifest = pd.read_csv(AUG / "manifests" / "hflip_manifest.csv")
    aug_train_m = aug_manifest[aug_manifest["split"] == "train"]
    aug_test_m = aug_manifest[aug_manifest["split"] == "test"]
    OUT.mkdir(parents=True, exist_ok=True)

    results = []
    for name in CONDS:
        cond = tb.CONDITIONS[name]
        train = tb.build_split("train", cond, config)
        val = tb.build_split("val", cond, config)
        test = tb.build_split("test", cond, config)
        aug_train = build_rows(aug_train_m, cond, config, AUG / "sequences")
        test_flip = build_rows(aug_test_m, cond, config, AUG / "sequences")
        test_side = test[test["take"] == "T02"]

        for label, tr in (("original", train), ("hflip_aug", pd.concat([train, aug_train], ignore_index=True))):
            model, cols = tb.train_model(tr, config)
            tag = f"{name}_{label}"
            res = {
                "condition": name, "train": label, "train_clips": len(tr),
                "val": tb.evaluate(model, cols, val, tag, "validation", OUT),
                "test": tb.evaluate(model, cols, test, tag, "test", OUT),
                "test_side": tb.evaluate(model, cols, test_side, tag, "test_side", OUT),
                "test_side_flipped": tb.evaluate(model, cols, test_flip, tag, "test_side_flipped", OUT),
            }
            row = {"condition": name, "train": label, "train_clips": len(tr)}
            for key in ("val", "test", "test_side", "test_side_flipped"):
                row[f"{key}_acc"] = res[key]["accuracy"]
                row[f"{key}_f1"] = res[key]["macro_f1"]
                row[f"{key}_correct"] = res[key]["clips"] - res[key]["wrong"]
                row[f"{key}_clips"] = res[key]["clips"]
                row[f"{key}_mean_conf"] = round(float(res[key]["pred"]["confidence"].mean()), 4)
            results.append(row)
            print(f"{name:14s} {label:10s} train={len(tr):3d}  "
                  f"val {row['val_acc']:.4f}/{row['val_f1']:.4f}  "
                  f"test {row['test_acc']:.4f}/{row['test_f1']:.4f}  "
                  f"test_side {row['test_side_correct']}/{row['test_side_clips']}  "
                  f"test_side_flipped {row['test_side_flipped_correct']}/{row['test_side_flipped_clips']}")
            if label == "hflip_aug":
                path = save_model(model, cols, tr, config, name, len(aug_train))
                print(f"  [SAVED] {path.relative_to(ROOT)}")

    table = pd.DataFrame(results)
    table.to_csv(OUT / "hflip_comparison.csv", index=False, encoding="utf-8-sig")
    print(f"\n[SAVED] {(OUT / 'hflip_comparison.csv').relative_to(ROOT)}")


if __name__ == "__main__":
    main()
