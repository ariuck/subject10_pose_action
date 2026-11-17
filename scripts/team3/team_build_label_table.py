"""3팀 5명 x 34개 영상(170개) 라벨표 만들기.

규칙
- 사람별 파일을 시간순으로 정렬해 2개씩 묶으면 1~17번 동작 순서와 같다.
- 각 쌍은 앞 파일이 정면(front), 뒤 파일이 측면(side)이다.
- 예외: 김석범·지혜성 9번은 측면을 먼저 찍었다 (대표 장면으로 확인).

출력: configs/team3_all_actions.csv
"""
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
VIDEO_ROOT = ROOT / "data" / "team3_videos"
OUTPUT = ROOT / "configs" / "team3_all_actions.csv"

PEOPLE = [
    ("P1", "강동연"),
    ("P2", "김동훈"),
    ("P3", "김석범"),
    ("P4", "이후영"),
    ("P5", "지혜성"),
]

ACTIONS = [
    (1, "kiosk_strike", "키오스크 내려치기", "타격·파손"),
    (2, "throw_object", "한팔로 물건 던지기", "던지기"),
    (3, "freezer_kick", "냉동고 앞차기", "차기"),
    (4, "shake_claw_machine", "뽑기기계 흔들기", "밀기·흔들기"),
    (5, "push_two_hands", "양손 밀치기", "밀기·흔들기"),
    (6, "swing_strike", "양손으로 휘둘러 타격하기", "타격·파손"),
    (7, "sit_on_mat", "매트 위에 앉기", "앉기"),
    (8, "sit_on_chair_phone", "의자 위에 점프해서 앉아 폰보기", "앉기"),
    (9, "lie_prone_phone", "앞으로 누워서 폰보기", "눕기·쓰러짐"),
    (10, "lie_back_side", "뒤로 눕고 옆으로 눕기", "눕기·쓰러짐"),
    (11, "body_bump", "몸통박치기", "밀기·흔들기"),
    (12, "kneel_claw_machine", "무릎꿇고 앉아 인형빼기", "앉기"),
    (13, "table_strike", "테이블 내려치기", "타격·파손"),
    (14, "vomit", "구토하기", "숙이기"),
    (15, "sleep_against_wall", "벽에 기대어 잠자기", "눕기·쓰러짐"),
    (16, "faint_back", "뒤로 기절하기", "눕기·쓰러짐"),
    (17, "trip_fall", "장애물 걸려 앞으로 넘어지기", "눕기·쓰러짐"),
]

# (사람, 동작번호) -> 쌍 안에서 측면을 먼저 찍은 경우
SIDE_FIRST = {("김석범", 9), ("지혜성", 9)}


def main():
    rows = []
    for person_id, person in PEOPLE:
        files = sorted((VIDEO_ROOT / person).glob("*.mp4"))
        if len(files) != 34:
            raise SystemExit(f"{person}: 영상이 34개가 아닙니다 ({len(files)}개)")
        for (action_id, slug, name, group), pair in zip(ACTIONS, zip(files[0::2], files[1::2])):
            views = ("side", "front") if (person, action_id) in SIDE_FIRST else ("front", "side")
            for view, path in zip(views, pair):
                rows.append({
                    "sample_id": f"{person_id}_A{action_id:02d}_{view}",
                    "person_id": person_id,
                    "person": person,
                    "action_id": action_id,
                    "action_label": slug,
                    "action_name": name,
                    "group": group,
                    "view": view,
                    "video_file": path.relative_to(ROOT).as_posix(),
                    "note": "측면을 먼저 촬영" if (person, action_id) in SIDE_FIRST else "",
                })
    df = pd.DataFrame(rows)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT, index=False, encoding="utf-8")
    print(f"[SAVED] {OUTPUT.relative_to(ROOT)} ({len(df)} rows)")


if __name__ == "__main__":
    main()
