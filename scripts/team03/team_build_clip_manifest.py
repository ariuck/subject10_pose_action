"""3팀 RGB 영상 255개의 Clip Manifest 만들기 (4일차 RGB 문서 STEP 1).

입력 : data/team_projects/team03/rgb/videos/S0x/S0x_A0nn_T0t.mp4
       configs/team03_action_classes.csv
출력 : configs/team03_clip_manifest.csv

파일명 규칙 (5일차와 같은 Subject_Action_Take)
- Subject : S01~S05 (실명 대신 가명 ID, 실명 연결표는 Git에 올리지 않음)
- Action  : A001~A017 (configs/team03_action_classes.csv)
- Take    : T01 = 정면(front), T02 = 측면(side), T03 = 후면(back) — 같은 동작을 다른 방향에서 따로 촬영
"""
import csv
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
VIDEO_ROOT = ROOT / "data" / "team_projects" / "team03" / "rgb" / "videos"
ACTION_FILE = ROOT / "configs" / "team03_action_classes.csv"
OUTPUT = ROOT / "configs" / "team03_clip_manifest.csv"
PATTERN = re.compile(r"^(S\d{2})_(A\d{3})_(T\d{2})\.mp4$")
VIEW = {"T01": "front", "T02": "side", "T03": "back"}
EXPECTED_SUBJECTS = 5
EXPECTED_ACTIONS = 17
EXPECTED_TAKES = 3


def main() -> None:
    with ACTION_FILE.open(encoding="utf-8-sig") as f:
        actions = {r["action_id"]: r for r in csv.DictReader(f)}

    rows, errors = [], []
    for path in sorted(VIDEO_ROOT.rglob("*.mp4")):
        m = PATTERN.match(path.name)
        if m is None:
            errors.append(f"파일명 규칙 오류: {path.name}")
            continue
        subject, action_id, take = m.groups()
        if path.parent.name != subject:
            errors.append(f"Subject 폴더 불일치: {path}")
        if action_id not in actions or take not in VIEW:
            errors.append(f"정의되지 않은 Action/Take: {path.name}")
            continue
        rows.append({
            "clip_id": path.stem,
            "subject_id": subject,
            "action_id": action_id,
            "action_label": actions[action_id]["action_label"],
            "take_id": take,
            "view": VIEW[take],
            "behavior_type": actions[action_id]["behavior_type"],
            "video_file": path.relative_to(ROOT).as_posix(),
            "source_type": "RGB",
            "review_status": "PENDING",
        })

    subjects = sorted({r["subject_id"] for r in rows})
    expected_total = EXPECTED_SUBJECTS * EXPECTED_ACTIONS * EXPECTED_TAKES
    if len(rows) != expected_total:
        errors.append(f"영상 수 {len(rows)} != {expected_total}")
    for s in subjects:
        for a in actions:
            for t in VIEW:
                if not any(r["clip_id"] == f"{s}_{a}_{t}" for r in rows):
                    errors.append(f"누락: {s}_{a}_{t}")

    if errors:
        print("\n".join(errors))
        sys.exit(1)

    with OUTPUT.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    print(f"Subjects: {subjects}")
    print(f"Actions : {len(actions)}  Takes: {list(VIEW)}")
    print(f"Clips   : {len(rows)}")
    print(f"[SAVED] {OUTPUT.relative_to(ROOT)}")
    print("Clip manifest: OK")


if __name__ == "__main__":
    main()
