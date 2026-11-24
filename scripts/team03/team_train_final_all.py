"""서비스용 최종 모델을 5명 전체 데이터로 학습해 저장한다.

구성: 45-D, 사람·동작마다 정면 T01 + 측면 T02 + 측면 좌우 반전 + 후면 T03 (4개)
- 원본 255개 + 측면 반전 85개 = 340개로 학습
- 이 모델은 따로 Test할 사람이 남지 않는다. 처음 보는 사람에 대한 성능은
  5명 교차 평가 평균(team_loso_compare.py, 같은 구성 0.753)으로 추정한다.
- 학습 데이터 확인용으로 Train 정확도만 함께 출력한다 (성능 지표 아님).

출력: artifacts/team03/day06/team_45d_3view_hflip_all5/
  rf_action_baseline.joblib · rf_feature_columns.json · rf_training_info.json

사용: python scripts/team03/team_train_final_all.py
"""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import team_day06_baseline as tb  # noqa: E402
from team_train_hflip_aug import AUG, build_rows, save_model  # noqa: E402

NAME = "team_45d"
FOLDER = "team_45d_3view_hflip_all5"
# --segment: 동작 구간만으로 학습 (team_loso_segment.py에서 LOSO 0.753 → 0.784)
tb.CONDITIONS["team_45d_segment"] = {"team_features": True, "segment_only": True}


def train_with_floor(floor: float) -> None:
    """동작 구간 기준선 최소값(floor)을 둔 서비스용 모델 (team_loso_segment_floor.py에서 0.4가 정확도 유지)."""
    import json
    import math  # noqa: F401
    from team_loso_segment_floor import rows_for_floor
    from team_loso_segment_v2 import load_clips

    config = tb.load_config()
    steps, min_valid = int(config["sequence_steps"]), int(config["min_valid_frames"])
    master = pd.concat([pd.read_csv(tb.SPLIT_DIR / f"{s}.csv") for s in ("train", "val", "test")], ignore_index=True)
    data, _ = rows_for_floor(load_clips(master, tb.SEQ_DIR), floor, steps, min_valid)
    aug, _ = rows_for_floor(load_clips(pd.read_csv(AUG / "manifests" / "hflip_manifest.csv"), AUG / "sequences"),
                            floor, steps, min_valid)
    train = pd.concat([data, aug], ignore_index=True)
    model, cols = tb.train_model(train, config)
    folder = f"team_45d_segment_floor{floor}_3view_hflip_all5"
    path = save_model(model, cols, train, config, "team_45d_segment", len(aug), folder=folder)
    info = json.loads((path / "rf_training_info.json").read_text(encoding="utf-8"))
    info["segment_floor"] = floor
    (path / "rf_training_info.json").write_text(json.dumps(info, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Train clips : {len(train)} (원본 {len(data)} + 측면 반전 {len(aug)}), segment_floor={floor}")
    print(f"[SAVED] {path.relative_to(tb.ROOT)}")


def train_with_summary(floor: float) -> None:
    """동작 구간(기준선 최소값 floor) + 30 Step 시간표 + 결정적인 순간 특징 225개 (team_loso_summary.py 0.784 → 0.835)."""
    import json
    from team_loso_segment_v2 import load_clips
    from team_loso_summary import build

    config = tb.load_config()
    steps, min_valid = int(config["sequence_steps"]), int(config["min_valid_frames"])
    master = pd.concat([pd.read_csv(tb.SPLIT_DIR / f"{s}.csv") for s in ("train", "val", "test")], ignore_index=True)
    data = build(load_clips(master, tb.SEQ_DIR), steps, min_valid, floor)
    aug = build(load_clips(pd.read_csv(AUG / "manifests" / "hflip_manifest.csv"), AUG / "sequences"), steps, min_valid, floor)
    train = pd.concat([data, aug], ignore_index=True)
    model, cols = tb.train_model(train, config)
    folder = f"team_45d_segment_floor{floor}_summary_3view_hflip_all5"
    path = save_model(model, cols, train, config, "team_45d_segment", len(aug), folder=folder)
    info = json.loads((path / "rf_training_info.json").read_text(encoding="utf-8"))
    info.update({"segment_floor": floor, "summary_features": True, "summary_feature_count": 225,
                 "frame_feature_count": 45, "feature_count": len(cols)})
    (path / "rf_training_info.json").write_text(json.dumps(info, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Train clips : {len(train)}, Features: {len(cols)} (시간표 {steps * 45} + 결정적 순간 225)")
    print(f"[SAVED] {path.relative_to(tb.ROOT)}")


def main() -> None:
    global NAME, FOLDER
    if "--summary" in sys.argv:
        train_with_summary(float(sys.argv[sys.argv.index("--floor") + 1]) if "--floor" in sys.argv else 0.4)
        return
    if "--floor" in sys.argv:
        train_with_floor(float(sys.argv[sys.argv.index("--floor") + 1]))
        return
    if "--segment" in sys.argv:
        NAME, FOLDER = "team_45d_segment", "team_45d_segment_3view_hflip_all5"
    config = tb.load_config()
    cond = tb.CONDITIONS[NAME]
    master = pd.concat([pd.read_csv(tb.SPLIT_DIR / f"{s}.csv") for s in ("train", "val", "test")],
                       ignore_index=True)
    data = build_rows(master, cond, config, tb.SEQ_DIR)
    aug = build_rows(pd.read_csv(AUG / "manifests" / "hflip_manifest.csv"), cond, config, AUG / "sequences")
    train = pd.concat([data, aug], ignore_index=True)
    model, cols = tb.train_model(train, config)
    path = save_model(model, cols, train, config, NAME, len(aug), folder=FOLDER)

    print(f"Train clips : {len(train)} (원본 {len(data)} + 측면 반전 {len(aug)})")
    print(f"Subjects    : {sorted(train.subject.unique())}")
    print(f"Views       : {train['take'].value_counts().to_dict()}")
    print(f"Classes     : {len(model.classes_)}  Features: {len(cols)}")
    train_acc = (model.predict(train[cols].to_numpy(float)) == train['action_label'].to_numpy()).mean()
    print(f"Train acc   : {train_acc:.4f} (학습 데이터 확인용, 성능 지표 아님)")
    print(f"[SAVED] {path.relative_to(tb.ROOT)}")


if __name__ == "__main__":
    main()
