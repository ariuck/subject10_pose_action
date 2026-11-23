"""3팀 7일차 Mini Challenge용 실행기.

교재 코드(42번 저장영상 Runtime, 44번 IP Webcam Runtime)는 그대로 두고
아래 세 가지만 3팀 것으로 바꿔서 실행한다.

1. 모델: artifacts/team03/day06/baseline_38d/ (3팀 17개 동작, 38-D, 30 Step)
2. Action 이름: configs/team03_action_classes.csv (A001~A017)
3. 출력 폴더: reports/day07/challenge/team03/

45-D 모델은 실시간 코드에 팀 Feature 7개 계산이 없어서 이번에는 38-D 모델만 사용한다.

사용 예:
  python scripts/team03/team_day07_runtime.py video --source data/day07/challenge/team03/videos/S93_A005_T01.mp4
  python scripts/team03/team_day07_runtime.py video --source ... --confidence 0.20
  python scripts/team03/team_day07_runtime.py webcam --url "http://192.168.40.115:4747/video"
"""

import csv
import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR = ROOT / "artifacts" / "team03" / "day06" / "baseline_38d"
CLASSES_CSV = ROOT / "configs" / "team03_action_classes.csv"
CHALLENGE_DIR = ROOT / "reports" / "day07" / "challenge" / "team03"

TEXTBOOK_SCRIPTS = {
    "video": ROOT / "scripts" / "42_realtime_action_video.py",
    "webcam": ROOT / "scripts" / "44_ip_webcam_action_realtime.py",
}
OUTPUT_DIRS = {
    "video": CHALLENGE_DIR / "saved_video",
    "webcam": CHALLENGE_DIR / "ip_webcam",
}

# IP Webcam에서 Target을 고르는 키. 0(Target 해제)·R(Buffer 초기화)·Q(종료)는 교재 그대로 쓰므로 피한다.
TARGET_KEYS = list("123456789abcdefgh")


def load_team_actions() -> dict:
    with CLASSES_CSV.open(encoding="utf-8") as f:
        return {row["action_id"]: row["action_label"] for row in csv.DictReader(f)}


def load_textbook_module(mode: str):
    path = TEXTBOOK_SCRIPTS[mode]
    spec = importlib.util.spec_from_file_location(f"textbook_{mode}", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> None:
    if len(sys.argv) < 2 or sys.argv[1] not in TEXTBOOK_SCRIPTS:
        print("사용: team_day07_runtime.py {video|webcam} [교재 42·44번 옵션]")
        sys.exit(1)

    mode = sys.argv.pop(1)
    actions = load_team_actions()
    module = load_textbook_module(mode)

    # 교재 코드의 6일차 모델 경로를 3팀 모델로 교체
    module.MODEL_PATH = MODEL_DIR / "rf_action_baseline.joblib"
    module.TRAIN_INFO_PATH = MODEL_DIR / "rf_training_info.json"
    module.DEFAULT_OUTPUT_DIR = OUTPUT_DIRS[mode]

    if mode == "video":
        # 파일명 S93_A005_T01.mp4 → A005 → push
        module.ACTION_MAP = actions
    else:
        module.KEY_TO_ACTION = {
            ord(key): label for key, label in zip(TARGET_KEYS, actions.values())
        }
        print("[3팀 Target 키] (아래에 나오는 교재 1~6 목록 대신 이 표를 사용)")
        for key, (action_id, label) in zip(TARGET_KEYS, actions.items()):
            print(f"  {key} → {action_id} {label}")
        print("  0 Target 해제 / R Buffer 초기화 / Q 종료(Log 저장)")
        print()

    print(f"[3팀 모델] {module.MODEL_PATH.relative_to(ROOT)}")
    module.main()


if __name__ == "__main__":
    main()
