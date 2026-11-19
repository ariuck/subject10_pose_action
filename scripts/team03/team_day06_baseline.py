"""3팀 170개 Dataset으로 6일차 흐름(34~40번)을 다시 수행한다 — 6일차 Team Mini Challenge.

교재 6일차 코드는 경로와 "6개 Action / Clip 6개"가 코드 안에 고정되어 있어 그대로 쓸 수 없다.
그래서 교재와 **같은 공통 함수**(src/sequence_utils.py)와 **같은 모델 설정**(configs/day06_baseline.json,
Random Forest + class_weight="balanced")을 그대로 불러와 3팀 데이터에 적용한다.

입력
- 5일차 흐름으로 만든 3팀 Split: data/team_projects/team03/splits/{train,val,test}.csv
  (Train S01·S02·S03 102개 / Validation S04 34개 / Test S05 34개)
- 3팀 Keypoint CSV (원본 좌표·Confidence)
- 3팀 45-D Sequence CSV (팀 Feature 7개, 동작 구간 in_action)

출력: reports/team_projects/team03/day06/, artifacts/team03/day06/

사용 예:
    python scripts/team03/team_day06_baseline.py validate   # Baseline + 한 가지 조건 비교 실험 (Validation만)
    python scripts/team03/team_day06_baseline.py test --condition baseline_38d   # 마지막에 Test 1회
"""
import argparse
import json
import sys
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.ensemble import RandomForestClassifier  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.sequence_utils import (  # noqa: E402
    FRAME_FEATURE_NAMES,
    build_frame_feature_matrix,
    flatten_sequence,
    middle_frame_feature,
    resample_sequence,
)

TEAM = ROOT / "data" / "team_projects" / "team03"
SPLIT_DIR = TEAM / "splits"
SEQ_DIR = TEAM / "sequences"
CONFIG_PATH = ROOT / "configs" / "day06_baseline.json"
REPORT_DIR = ROOT / "reports" / "team_projects" / "team03" / "day06"
ARTIFACT_DIR = ROOT / "artifacts" / "team03" / "day06"
TEAM_FEATURES = ["torso_tilt_deg", "bbox_aspect_ratio", "hip_floor_height",
                 "left_ankle_speed", "right_ankle_speed", "left_wrist_speed", "right_wrist_speed"]
ANGLES = ["left_elbow_angle", "right_elbow_angle", "left_knee_angle", "right_knee_angle"]

# 한 번에 한 가지 조건만 바꾼 실험 (Baseline = 교재 조건: 38-D, 30 Step, Clip 전체, 정면+측면, 300 Tree)
CONDITIONS = {
    "baseline_38d": {},
    "static_middle_frame": {"static": True},
    "coordinates_only": {"drop_angles": True},
    "team_45d": {"team_features": True},
    "action_segment_only": {"segment_only": True},
    "front_only": {"view": "T01"},
    "side_only": {"view": "T02"},
}


def load_config() -> dict:
    with CONFIG_PATH.open(encoding="utf-8") as f:
        return json.load(f)


def clip_frames(item, cond: dict):
    """Clip 하나 → (Frame Feature 행렬, frame_ids, Feature 이름). 교재 build_frame_feature_matrix 사용."""
    kp = pd.read_csv(ROOT / item["keypoint_csv"])
    matrix, frame_ids = build_frame_feature_matrix(kp)
    names = list(FRAME_FEATURE_NAMES)
    if cond.get("segment_only") or cond.get("team_features"):
        seq = pd.read_csv(SEQ_DIR / f"{item['clip_id']}_sequence.csv").set_index("frame_id")
        if cond.get("segment_only"):
            keep = seq.loc[frame_ids, "in_action"].to_numpy() == 1
            matrix, frame_ids = matrix[keep], frame_ids[keep]
        if cond.get("team_features"):
            extra = (seq[TEAM_FEATURES].apply(pd.to_numeric, errors="coerce")
                     .interpolate(limit_direction="both").loc[frame_ids].to_numpy(np.float32))
            matrix = np.hstack([matrix, extra])
            names += TEAM_FEATURES
    if cond.get("drop_angles"):
        keep_cols = [i for i, n in enumerate(names) if n not in ANGLES]
        matrix, names = matrix[:, keep_cols], [names[i] for i in keep_cols]
    return matrix, frame_ids, names


def build_split(split: str, cond: dict, config: dict) -> pd.DataFrame:
    manifest = pd.read_csv(SPLIT_DIR / f"{split}.csv")
    if cond.get("view"):
        manifest = manifest[manifest["take"] == cond["view"]]
    steps = int(config["sequence_steps"])
    rows = []
    for _, item in manifest.iterrows():
        matrix, frame_ids, names = clip_frames(item, cond)
        if len(matrix) < int(config["min_valid_frames"]):
            print(f"  [SKIP] {item['clip_id']} valid={len(matrix)}")
            continue
        if cond.get("static"):
            values, cols = middle_frame_feature(matrix), names
        else:
            values = flatten_sequence(resample_sequence(matrix, frame_ids, target_steps=steps))
            cols = [f"t{t:02d}_{n}" for t in range(steps) for n in names]
        row = {"clip_id": item["clip_id"], "subject": item["subject"], "action_id": item["action_id"],
               "action_label": item["action_label"], "take": item["take"], "valid_frames": len(matrix),
               "video_path": item["video_path"]}
        row.update(zip(cols, map(float, values)))
        rows.append(row)
    return pd.DataFrame(rows)


META = ["clip_id", "subject", "action_id", "action_label", "take", "valid_frames", "video_path"]


def train_model(train: pd.DataFrame, config: dict):
    cols = [c for c in train.columns if c not in META]
    model = RandomForestClassifier(n_estimators=int(config["n_estimators"]), max_depth=config["max_depth"],
                                   random_state=int(config["random_state"]), class_weight="balanced", n_jobs=-1)
    model.fit(train[cols].to_numpy(float), train["action_label"].astype(str).to_numpy())
    return model, cols


def evaluate(model, cols, data: pd.DataFrame, name: str, split_label: str, out_dir: Path) -> dict:
    labels = list(model.classes_)
    y_true = data["action_label"].astype(str).to_numpy()
    proba = model.predict_proba(data[cols].to_numpy(float))
    y_pred = np.asarray(labels)[proba.argmax(1)]
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)
    out_dir.mkdir(parents=True, exist_ok=True)
    pred = data[["clip_id", "subject", "take", "action_label", "video_path"]].rename(
        columns={"action_label": "true_label"}).assign(pred_label=y_pred, confidence=proba.max(1).round(4),
                                                       correct=y_true == y_pred)
    pred.to_csv(out_dir / f"{split_label}_predictions_{name}.csv", index=False, encoding="utf-8-sig")
    report = classification_report(y_true, y_pred, labels=labels, zero_division=0)
    (out_dir / f"{split_label}_metrics_{name}.txt").write_text(
        f"{split_label} Accuracy: {acc:.4f}\n{split_label} Macro F1: {f1:.4f}\n\n{report}", encoding="utf-8")
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    fig, ax = plt.subplots(figsize=(11, 10))
    ConfusionMatrixDisplay(cm, display_labels=labels).plot(ax=ax, xticks_rotation=90, colorbar=False)
    ax.set_title(f"Team03 {split_label} Confusion Matrix ({name})")
    plt.tight_layout()
    plt.savefig(out_dir / f"{split_label}_confusion_{name}.png", dpi=120)
    plt.close()
    return {"condition": name, "clips": len(data), "feature_count": len(cols),
            "accuracy": round(acc, 4), "macro_f1": round(f1, 4), "wrong": int((y_true != y_pred).sum()),
            "report": report, "pred": pred}


def run_validate() -> None:
    config = load_config()
    out = REPORT_DIR / "validation"
    results = []
    for name, cond in CONDITIONS.items():
        train = build_split("train", cond, config)
        val = build_split("val", cond, config)
        model, cols = train_model(train, config)
        res = evaluate(model, cols, val, name, "validation", out)
        if name == "baseline_38d":
            ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
            joblib.dump(model, ARTIFACT_DIR / "rf_team03_baseline.joblib")
            (ARTIFACT_DIR / "rf_team03_feature_columns.json").write_text(json.dumps(cols), encoding="utf-8")
            print(f"Train clips   : {len(train)} ({sorted(train.subject.unique())})")
            print(f"Feature count : {len(cols)}")
            print(f"Classes       : {len(model.classes_)}")
            print(res["report"])
            wrong = res["pred"][~res["pred"].correct]
            wrong.to_csv(out / "validation_misclassified_baseline_38d.csv", index=False, encoding="utf-8-sig")
            print(wrong[["clip_id", "true_label", "pred_label", "confidence"]].to_string(index=False))
            print()
        results.append({k: v for k, v in res.items() if k not in ("report", "pred")}
                       | {"train_clips": len(train)})
        print(f"{name:22s} train={len(train):3d} val={res['clips']:3d} features={res['feature_count']:5d} "
              f"acc={res['accuracy']:.4f} f1={res['macro_f1']:.4f} wrong={res['wrong']}")
    table = pd.DataFrame(results)
    table.to_csv(out / "condition_comparison.csv", index=False, encoding="utf-8-sig")
    print(f"\n[SAVED] {(out / 'condition_comparison.csv').relative_to(ROOT)}")


def run_test(condition: str) -> None:
    config = load_config()
    cond = CONDITIONS[condition]
    train = build_split("train", cond, config)
    test = build_split("test", cond, config)
    model, cols = train_model(train, config)
    res = evaluate(model, cols, test, condition, "test", REPORT_DIR / "test")
    print(f"Test Subject : {sorted(test.subject.unique())}  clips={len(test)}  condition={condition}")
    print(f"Test Accuracy: {res['accuracy']:.4f}")
    print(f"Test Macro F1: {res['macro_f1']:.4f}")
    print(res["report"])
    wrong = res["pred"][~res["pred"].correct]
    print(wrong[["clip_id", "true_label", "pred_label", "confidence"]].to_string(index=False))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["validate", "test"])
    parser.add_argument("--condition", default="baseline_38d", choices=list(CONDITIONS))
    args = parser.parse_args()
    if args.mode == "validate":
        run_validate()
    else:
        run_test(args.condition)


if __name__ == "__main__":
    main()
