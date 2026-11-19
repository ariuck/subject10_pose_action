# Day 05 Summary

## 1. Dataset

- Subjects: S01, S02, S03
- Actions: 6 (A001 bend_return ~ A006 wave, 4일차 `configs/action_classes.csv` 그대로 사용)
- Take: Action당 1
- Total Clips: 18
- 입력 검사 (`25_check_day05_data.py`): 18개 모두 640x480, 25fps, `Day05 video data: READY`

## 2. Keypoint Dataset

- 생성된 Keypoint CSV: 18개 (`data/day05/keypoints/S0x/<action>/S0x_A00n_T01.csv`)
- COCO Keypoint: 17개
- 한 Frame = 기본 정보 10개 + Keypoint 85개 = 95개 Column
- 저장 값:
  - Pixel 좌표
  - Image Normalized 좌표 (예: nose_x 343.5 / 640 = 0.5367)
  - Confidence
- 18개 Clip 모두 pose_ratio = 1.0. walk_turn_walk(A003)만 평균 Confidence가 0.888 / 0.914 / 0.894로 상대적으로 낮았다.

## 3. 자동 QA

- AUTO_PASS: 13
- CHECK: 5
- Pose Ratio: 18개 모두 1.0
- Low Critical Ratio: 18개 모두 0.0
- CHECK 주요 원인:
  - S02 A001~A005의 single_person_ratio (0.0000 / 0.0000 / 0.0581 / 0.2987 / 0.0083)

## 4. Human QA

- CHECK 5개: 원본 영상을 보니 화면 오른쪽 **옷걸이에 걸린 검은 외투**를 YOLO가 두 번째 사람으로 검출한 것이었다. 촬영 대상은 한 명이고 대표 인물(BBox conf 0.92~0.93)도 올바르게 선택되어 5개 모두 PASS.
- AUTO_PASS 13개: 교재에 실린 28·29번 코드는 모든 Clip의 `human_decision`이 입력되어야 통과하고 PASS만 승인하므로(24.8 처음 실행 시 Approved 5), 교재 24.9처럼 13개도 시작·25%·50%·75%·끝 장면으로 확인하고 PASS로 기록했다.
- Human PASS: 18개
- RESHOOT: 0개
- EXCLUDE: 0개

## 5. Approved Dataset

- AUTO_PASS → Human PASS: 13개
- CHECK → Human PASS: 5개
- Approved Clips: 18개 (Subject별 6개, Action별 3개)

## 6. Subject-wise Split

- Train: S01 / 6 Clips
- Validation: S02 / 6 Clips
- Test: S03 / 6 Clips
- 각 Split에 A001~A006 모두 존재
- 설정: `configs/day05_subject_split.json`

## 7. Leakage 검사

- Subject overlap: 없음
- Clip overlap: 없음
- Action Coverage: 정상
- Result: Leakage check: OK

## 8. Random Frame Split 비교

- Train Frames: 1946
- Test Frames: 487
- Subject overlap: S01, S02, S03
- Clip overlap: 18

Random Frame Split에서는 같은 사람과 같은 Clip의 Frame이
Train과 Test에 함께 들어갈 수 있음을 확인했다.

## 9. 현재 Dataset의 한계

현재 데이터는:

3 Subjects × 6 Actions × 1 Take = 18 Clips

의 최소 교육용 Dataset이다.

Dataset 구축·QA·Subject-wise Split과
Action Recognition Pipeline을 배우기에는 적절하지만
현업 배포 성능이나 충분한 일반화 성능을 판단하기에는 작다.

## 10. Mini Challenge

- 선택한 Action: 3팀 A016 `faint_back` (뒤로 기절하기), 측면
- Approved Clip 수: 3
- CHECK Clip: 없음 (3개 모두 AUTO_PASS)
- Human QA 결과: 3개 PASS (교재 28번 코드가 전체 입력을 요구해 직접 확인 후 기록)
- Train: S01
- Validation: S02
- Test: S03
- Leakage check: OK
- 자세한 내용: `reports/day05/challenge/mini_challenge_result.md`

## 11. 실습 중 확인한 점

- 교재 26·27번은 Action 기준(A001~A006)과 중요 관절 표가 코드에 고정되어 있어, 3팀 동작에는 그대로 쓸 수 없었다. 교재 원본은 그대로 두고 3팀 사본(`scripts/team03/`)을 만들었다.
- 교재 26번은 `--video-root`에 상대 경로(예: `data/day05/challenge/videos`)를 주면 `relative_to(ROOT)`에서 ValueError가 난다. 3팀 사본에서는 ROOT 기준 경로로 바꿔서 해결했다.

## 12. 오늘의 핵심

- Pose Dataset은 학습 전에 품질을 검사해야 한다.
- 자동 QA의 CHECK는 탈락이 아니라 "사람이 먼저 볼 영상"이다. 이번 CHECK 5개도 원인은 배경 물체 오검출이었다.
- 평가용 데이터는 사람 기준으로 분리하여 Leakage를 막아야 한다.
