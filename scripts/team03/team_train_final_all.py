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


def main() -> None:
    global NAME, FOLDER
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
