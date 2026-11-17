"""team3_all_actions.csv에서 교재 22번 코드가 읽는 형식의 Action 설정 파일 만들기.

교재 22번은 Sequence CSV 이름을 action_label로 만들기 때문에
action_id / action_label 자리에 겹치지 않는 sample_id(예: P3_A09_side)를 넣는다.
동작 이름·방향은 sample_id로 team3_all_actions.csv와 다시 연결할 수 있다.

사용 예:
    python scripts/team3/team_make_sequence_config.py --person P3
    python scripts/team3/team_make_sequence_config.py            # 170개 전부
"""
import argparse
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
LABEL_FILE = ROOT / "configs" / "team3_all_actions.csv"
VIDEO_ROOT = "data/team3_videos/"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--person", default=None, help="P1~P5 (생략하면 전체)")
    args = parser.parse_args()

    labels = pd.read_csv(LABEL_FILE, keep_default_na=False)
    if args.person:
        labels = labels[labels.person_id == args.person]
        if labels.empty:
            raise SystemExit(f"{args.person}에 해당하는 영상이 없습니다.")

    out = pd.DataFrame({
        "action_id": labels.sample_id,
        "action_label": labels.sample_id,
        "type": "dynamic",
        "video_file": labels.video_file.str.removeprefix(VIDEO_ROOT),
        "description": labels.person + " " + labels.action_id.astype(str) + "번 "
                       + labels.action_name + " (" + labels.view + ")",
    })
    name = f"team3_sequence_actions_{args.person or 'all'}.csv"
    path = ROOT / "configs" / name
    out.to_csv(path, index=False, encoding="utf-8")
    print(f"[SAVED] {path.relative_to(ROOT)} ({len(out)} rows)")


if __name__ == "__main__":
    main()
