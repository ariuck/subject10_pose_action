"""3팀 Feature를 단계별로 늘려 가며 Validation(S04) 성능 변화를 기록한다.

38-D (교재)
→ +자세 3개   (상체 기울기, BBox 비율, 골반-바닥 높이)          = 41-D
→ +발목 속도 2개                                                 = 43-D
→ +손목 속도 2개                                                 = 45-D (강사님 추천 +7)
비교용: 38-D + 속도 4개만 (자세 Feature 없이)                    = 42-D

모델·Sequence 설정은 6일차 Baseline과 같다 (team_day06_baseline.py의 함수 재사용).
Test(S05)는 사용하지 않는다.

출력: reports/team_projects/team03/day06/feature_steps/
  - feature_steps.csv           단계별 Accuracy·Macro F1
  - per_class_f1.csv            단계별 동작 F1
  - prediction_changes.csv      38-D → 45-D에서 예측이 바뀐 Clip
  - team_feature_importance.csv 45-D 모델에서 Feature 종류별 중요도 합
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
from team_day06_baseline import (  # noqa: E402
    REPORT_DIR,
    ROOT,
    TEAM_FEATURES,
    build_split,
    evaluate,
    load_config,
    train_model,
)

POSTURE = ["torso_tilt_deg", "bbox_aspect_ratio", "hip_floor_height"]
ANKLE = ["left_ankle_speed", "right_ankle_speed"]
WRIST = ["left_wrist_speed", "right_wrist_speed"]
STEPS = [
    ("38d_textbook", []),
    ("41d_+posture", POSTURE),
    ("43d_+ankle_speed", POSTURE + ANKLE),
    ("45d_+wrist_speed", POSTURE + ANKLE + WRIST),
    ("42d_speed_only", ANKLE + WRIST),
]
OUT = REPORT_DIR / "feature_steps"


def main() -> None:
    config = load_config()
    OUT.mkdir(parents=True, exist_ok=True)
    rows, per_class, preds, models = [], {}, {}, {}
    for name, feats in STEPS:
        cond = {"team_features": feats} if feats else {}
        train = build_split("train", cond, config)
        val = build_split("val", cond, config)
        model, cols = train_model(train, config)
        res = evaluate(model, cols, val, name, "validation", OUT)
        labels = list(model.classes_)
        f1s = f1_score(res["pred"].true_label, res["pred"].pred_label, labels=labels, average=None, zero_division=0)
        per_class[name] = dict(zip(labels, np.round(f1s, 3)))
        preds[name] = res["pred"].set_index("clip_id")
        models[name] = (model, cols)
        rows.append({"step": name, "frame_features": 38 + len(feats), "sequence_features": len(cols),
                     "added": "+".join(feats) or "-", "accuracy": res["accuracy"], "macro_f1": res["macro_f1"],
                     "wrong": res["wrong"]})
        print(f"{name:18s} {38 + len(feats)}-D  features={len(cols):5d}  acc={res['accuracy']:.4f}  "
              f"f1={res['macro_f1']:.4f}  wrong={res['wrong']}")

    pd.DataFrame(rows).to_csv(OUT / "feature_steps.csv", index=False, encoding="utf-8-sig")
    pc = pd.DataFrame(per_class)
    pc["diff_45_vs_38"] = (pc["45d_+wrist_speed"] - pc["38d_textbook"]).round(3)
    pc.to_csv(OUT / "per_class_f1.csv", encoding="utf-8-sig")
    print("\n[동작별 F1]")
    print(pc.sort_values("diff_45_vs_38").to_string())

    a, b = preds["38d_textbook"], preds["45d_+wrist_speed"]
    changed = a.join(b, lsuffix="_38", rsuffix="_45")
    changed = changed[changed.pred_label_38 != changed.pred_label_45][
        ["true_label_38", "pred_label_38", "confidence_38", "pred_label_45", "confidence_45"]]
    changed.to_csv(OUT / "prediction_changes.csv", encoding="utf-8-sig")
    print("\n[38-D → 45-D 예측이 바뀐 Clip]")
    print(changed.to_string())

    model, cols = models["45d_+wrist_speed"]
    imp = pd.Series(model.feature_importances_, index=cols)
    kind = imp.index.str.replace(r"^t\d\d_", "", regex=True)
    group = np.where(kind.isin(TEAM_FEATURES), kind,
                     np.where(kind.str.endswith("_angle"), "joint_angles(4)", "normalized_coords(34)"))
    by_group = imp.groupby(group).sum().sort_values(ascending=False).round(4)
    by_group.to_csv(OUT / "team_feature_importance.csv", header=["importance_sum"], encoding="utf-8-sig")
    print("\n[45-D 모델 Feature 종류별 중요도 합 (전체 = 1)]")
    print(by_group.to_string())
    print(f"\n[SAVED] {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
