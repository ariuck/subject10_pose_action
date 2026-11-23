# Day 05 Mini Challenge

## 1. 선택한 Action

- Action ID: A016 (3팀 동작 정의표 `configs/team03_action_classes.csv`)
- Action Label: `fall_backward` (뒤로 넘어지기 — 서 있다가 뒤로 쓰러져 누운 채 움직이지 않음)
- 우리 팀(3팀 무인 상점 이상행동) 17개 동작 중 하나를 골랐다. 모션캡처 촬영 때 팀원들이 같은 동작을 같은 장소·같은 측면 카메라로 찍은 영상이다.

## 2. 데이터

- S01 파일: `S01_A016_T02.mp4` (측면)
- S02 파일: `S02_A016_T02.mp4` (측면)
- S03 파일: `S03_A016_T02.mp4` (측면)
- 위치: `data/day05/challenge/videos/S0x/` (팀 영상 폴더 `data/team_projects/team03/rgb/videos/`를 가리키는 링크)
- 3팀 영상은 동작마다 정면(T01)·측면(T02)을 따로 찍었기 때문에 Take 번호를 방향으로 사용했다. 이번에는 세 사람 모두 측면(T02)을 사용했다.

## 3. 자동 QA 결과

| Subject | QA Status | Pose Ratio | Single Person Ratio | Mean KP Conf | Low Critical Ratio |
|---|---|---:|---:|---:|---:|
| S01 | AUTO_PASS | 1.000 | 0.994 | 0.808 | 0.000 |
| S02 | AUTO_PASS | 1.000 | 1.000 | 0.809 | 0.000 |
| S03 | AUTO_PASS | 1.000 | 1.000 | 0.822 | 0.000 |

- 중요 관절: 어깨·골반·무릎·발목 (쓰러진 자세의 상체 기울기와 골반-바닥 높이를 계산하는 관절)
- 교재 26·27번은 Action 기준표가 교재 6개 동작으로 고정돼 있어서, 같은 처리 흐름에 3팀 동작표만 바꾼 사본(`scripts/team03/team_extract_keypoint_dataset.py`, `team_keypoint_qa.py`)으로 실행했다.

## 4. Human QA

- CHECK Clip: 없음
- 필수 Human QA: 없음
- 다만 교재 28·29번 코드는 모든 Clip의 `human_decision`이 입력되어야 통과하므로, 세 영상을 직접 보고 PASS로 기록했다 (서 있다가 뒤로 쓰러져 누운 뒤 일어남, 전신이 화면 안, 다른 사람 없음).
- 재촬영 여부: 없음

## 5. Approved Dataset

- Approved Clip 수: 3 (S01 1개, S02 1개, S03 1개)

## 6. Subject-wise Split

- Train: S01 / `S01_A016_T02`
- Validation: S02 / `S02_A016_T02`
- Test: S03 / `S03_A016_T02`
- 설정: `configs/day05_challenge_subject_split.json`

## 7. Leakage 검사

- Subject overlap: 없음
- Clip overlap: 없음
- Leakage check: OK

## 8. 한 줄 정리

- 같은 동작이라도 사람마다 평균 Confidence가 0.808 / 0.809 / 0.822로 조금씩 달랐다. 측면에서 쓰러지는 동작이라 교재 영상(0.95 안팎)보다 전체적으로 낮았지만, 중요 관절의 저신뢰도 비율은 세 사람 모두 0이었다.
- Random Frame Split을 하면 같은 사람·같은 영상의 거의 같은 Frame이 Train과 Test에 함께 들어가 성능이 부풀려지므로, 처음 보는 사람을 평가하려면 사람 단위로 나누는 Subject-wise Split을 써야 한다.

## 참고 — 3팀 전체 Dataset (170개)에도 같은 흐름 적용

- 5명 × 17동작 × 정면·측면 = 170개 → 자동 QA AUTO_PASS 166 / CHECK 4 → Human QA PASS 170 → Approved 170
- Split: Train S01·S02·S03 (102개) / Validation S04 (34개) / Test S05 (34개), Leakage check: OK
- 자세한 내용: `reports/team_projects/team03/team03_data_preparation_report.md`
