"""지금 서비스 모델 구성(45-D, 동작 구간, 후면 + 측면 반전)의 5명 교차 평가를 동작별로 쪼개 본다.

어떤 동작을 잘 맞히고 어떤 동작끼리 헷갈리는지 알아야, 무엇을 더 찍고 무엇을 고쳐야 할지 정할 수 있다.

출력: reports/team_projects/team03/day06/loso/per_action_accuracy.csv, confusion_pairs.csv (Git 제외 CSV)
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import team_day06_baseline as tb  # noqa: E402
from team_train_hflip_aug import AUG, build_rows  # noqa: E402

OUT = tb.REPORT_DIR / "loso"
SUBJECTS = ["S01", "S02", "S03", "S04", "S05"]
GROUPS = {"kiosk_tap": "타격·파손", "swing": "타격·파손", "shotgun": "타격·파손", "throw_one_hand": "던지기",
          "front_kick": "차기", "grab_shake": "밀기·흔들기", "push": "밀기·흔들기", "body_slam": "밀기·흔들기",
          "sit_floor": "앉기", "sit_on_object": "앉기", "crouch_take_item": "앉기", "vomit": "숙이기",
          "lie_forward": "눕기·쓰러짐", "lie_backward": "눕기·쓰러짐", "fall_sideways": "눕기·쓰러짐",
          "fall_backward": "눕기·쓰러짐", "fall_forward": "눕기·쓰러짐"}


def main() -> None:
    config = tb.load_config()
    tb.CONDITIONS["team_45d_segment"] = {"team_features": True, "segment_only": True}
    cond = tb.CONDITIONS["team_45d_segment"]
    master = pd.concat([pd.read_csv(tb.SPLIT_DIR / f"{s}.csv") for s in ("train", "val", "test")], ignore_index=True)
    data = build_rows(master, cond, config, tb.SEQ_DIR)
    aug = build_rows(pd.read_csv(AUG / "manifests" / "hflip_manifest.csv"), cond, config, AUG / "sequences")
    cols = [c for c in data.columns if c not in tb.META]

    preds = []
    for s in SUBJECTS:
        train = pd.concat([data[data.subject != s], aug[aug.subject != s]], ignore_index=True)
        model, _ = tb.train_model(train, config)
        te = data[data.subject == s]
        proba = model.predict_proba(te[cols].to_numpy(float))
        classes = np.asarray(model.classes_)
        order = np.argsort(proba, axis=1)[:, ::-1]
        for i, (_, row) in enumerate(te.iterrows()):
            p = proba[i]
            group_p = pd.Series(p, index=classes).groupby(lambda c: GROUPS[c]).sum()
            preds.append({"clip_id": row["clip_id"], "subject": s, "take": row["take"], "true": row["action_label"],
                          "pred": classes[order[i, 0]], "conf": p[order[i, 0]],
                          "in_top3": row["action_label"] in classes[order[i, :3]],
                          "true_group": GROUPS[row["action_label"]], "pred_group": group_p.idxmax(),
                          "group_conf": group_p.max()})
    df = pd.DataFrame(preds)
    df["correct"] = df.true == df.pred
    df["group_correct"] = df.true_group == df.pred_group

    print(f"전체 정확도 {df.correct.mean():.3f} | 후보 3개 안 {df.in_top3.mean():.3f} | "
          f"묶음 정확도 {df.group_correct.mean():.3f}")
    print(f"평균 Confidence: 맞힘 {df[df.correct].conf.mean():.3f} / 틀림 {df[~df.correct].conf.mean():.3f} | "
          f"묶음 Confidence {df.group_conf.mean():.3f}")
    print("방향별:", df.groupby("take").correct.mean().round(3).to_dict())

    per = (df.groupby("true").agg(acc=("correct", "mean"), top3=("in_top3", "mean"), n=("correct", "size"),
                                  conf=("conf", "mean"))
           .sort_values("acc").round(3))
    per["자주 틀린 답"] = [df[(df.true == a) & ~df.correct].pred.value_counts().head(2).to_dict() for a in per.index]
    print("\n동작별 (정확도 낮은 순):")
    print(per.to_string())

    pairs = (df[~df.correct].groupby(["true", "pred"]).size().sort_values(ascending=False).head(12))
    print("\n가장 많이 헷갈린 쌍 (정답 → 예측):")
    print(pairs.to_string())

    by_subject = df.groupby("subject").correct.mean().round(3)
    print("\n사람별:", by_subject.to_dict())
    OUT.mkdir(parents=True, exist_ok=True)
    per.to_csv(OUT / "per_action_accuracy.csv", encoding="utf-8-sig")
    pairs.to_csv(OUT / "confusion_pairs.csv", encoding="utf-8-sig")


if __name__ == "__main__":
    main()
