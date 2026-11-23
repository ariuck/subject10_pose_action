# 팀별 RGB 데이터 준비 결과 — 3팀 무인 상점 이상행동 인식

4일차 `RGB영상_전처리_Sequence_Dataset` 문서의 팀 확장 작업과 5일차 Dataset QA·Subject-wise Split을
3팀 영상 170개에 적용한 결과다. 영상은 모션캡처 촬영 때 팀원들이 직접 찍었다.

## 1. 프로젝트와 Action

- 팀 번호·프로젝트명: 3팀 — 무인 상점 이상행동 인식
- Action 수: 17 (`configs/team03_action_classes.csv`)
- 촬영 Subject 수: 5 (S01~S05, 실명 대신 가명 ID)
- 전체 MP4 수: 170 = 5명 × 17동작 × 2방향 (T01 정면 / T02 측면)
- 교재 가정(5명 × 15동작 × 5회 = 375개)과 달리 Take 대신 **촬영 방향**을 두 개로 나눴다.
- 2026-11-23 팀에서 사람 ID 순서와 동작 이름을 통일했다. 영상·Keypoint·라벨·QA·Split 파일 이름과 내용을 모두 새 기준으로 바꿨고, Keypoint 값은 바꾸기 전과 같은 것을 확인했다. 동작 번호(A001~A017)는 그대로다.

### 이상행동 정의표 (자세형 / 시간 변화형)

강사님 안내대로 동작을 **단순한 자세**와 **시간 변화가 필요한 행동**으로 나눴다.

| 구분 | 동작 | 판단에 필요한 정보 |
|---|---|---|
| 자세형 (posture) | A007 바닥에 앉기, A008 위에 앉기, A009 앞으로 눕기, A010 뒤로 눕기, A012 쪼그려 물건 빼내기, A014 구토, A015 옆으로 쓰러지기 | 어떤 자세를 얼마나 오래 유지했는지 (상체 기울기, 골반-바닥 높이) |
| 시간 변화형 (temporal) | A001 키오스크 두드리기, A002 한손 물건 던지기, A003 앞차기, A004 잡고 흔들기, A005 밀치기, A006 휘두르기, A011 몸통 박치기, A013 샷건, A016 뒤로 넘어지기, A017 앞으로 넘어지기 | 손·발·골반이 얼마나 빠르게 움직였는지, 자세가 어떻게 바뀌었는지 |

- 17개 모두 **이상행동**이다. 비교할 **정상행동 영상은 아직 없다** (7장 참고).
- A015·A016·A017은 쓰러지는 순간(시간 변화) 뒤에 누운·주저앉은 자세가 이어지는 혼합형이다.

## 2. 데이터 검사

- 정상 재생 MP4 수: 170 / 170 (모두 열리고 첫 Frame 읽힘)
- 해상도·FPS: 세로 1080×1920 145개, 가로 1920×1080 25개 / 59.7~60.0fps
- 파일 누락·중복·Label 오류: 없음 (`scripts/team03/team_build_clip_manifest.py` → `configs/team03_clip_manifest.csv`)
- 파일명 규칙: `S0x_A0nn_T0t.mp4` (5일차와 같은 Subject_Action_Take)
- 대표 Frame 검토 내용:
  - 사람별 34개 영상의 대표 장면을 모아 17개 동작 순서와 정면/측면을 확인했다 (팀원 확인).
  - S01·S04의 A009(앞으로 눕기)는 측면을 먼저 찍어서 파일 순서가 반대였다. 장면을 보고 바로잡았다.
- 영상 정보의 Frame 수(70,621)보다 실제로 읽힌 Frame(70,159)이 적다. 휴대폰 영상에서 흔한 현상이라 실제로 읽힌 Frame만 사용했다.

## 3. Pose / Feature / Sequence

- 사용 Pose 모델: `yolo11n-pose.pt` (COCO 17, `configs/project.json` 그대로)
- Keypoint Dataset: 5일차 26번 방식 + 선택 인물의 BBox 좌표 4개 (`scripts/team03/team_extract_keypoint_dataset.py`)
  - Clip당 CSV 1개, 99개 Column (기본 10 + BBox 4 + Keypoint 85)
  - 원본 Pixel 좌표, Image 정규화 좌표, 관절별 Confidence를 모두 보존 (RGB 문서: "원본 화면 좌표와 정규화 좌표를 함께 저장")

| Subject | 읽은 Frame | Pose 성공 | 비율 |
|---|---:|---:|---:|
| S01 | 12,447 | 12,447 | 100% |
| S02 | 17,510 | 17,302 | 98.8% |
| S03 | 13,015 | 13,015 | 100% |
| S04 | 12,873 | 12,848 | 99.8% |
| S05 | 14,314 | 14,314 | 100% |
| 합계 | 70,159 | 69,926 | 99.7% |

- 대표 오류 또는 누락 관절:
  - Pose 실패는 7개 Clip에서만 나왔고, 모두 바닥에 눕거나 엎드린 장면(A010·A012·A016·A017)이었다. 가장 많은 것은 S02_A010_T01(171 Frame)이다.
  - 측면은 반대쪽 팔다리가 가려져 평균 Confidence 중앙값이 0.831로 정면(0.953)보다 낮았다.

### 기본 38-D 사용 여부와 팀별 추가 Feature (45-D)

- 기본 38-D: 교재와 같은 계산 (골반 중심 + 어깨 폭 정규화 좌표 34개 + 팔꿈치·무릎 각도 4개)
- 추가 7개: 강사님 3팀 추천 Feature(+7 → 45-D)를 우리 영상에 맞췄다 (`scripts/team03/team_build_feature_sequences.py`)

| # | Feature | 계산 | 단위 | 강사님 표와의 관계 |
|---|---|---|---|---|
| 39 | `torso_tilt_deg` | 골반 중심→어깨 중심 벡터와 화면 위쪽 사이 각도 | ° (0 서기, 90 수평) | 상체 기울기 1개 |
| 40 | `bbox_aspect_ratio` | 사람 BBox 가로 / 세로 | 비율 | 사람 BBox 종횡비 1개 |
| 41 | `hip_floor_height` | (BBox 아래 끝 y − 골반 중심 y) / 상체 길이 | 상체 길이 배수 | 골반과 바닥의 상대 높이 1개 (바닥 = 몸의 가장 낮은 점) |
| 42 | `left_ankle_speed` | 0.1초 동안 왼발목 이동 거리 / 상체 길이 | 상체 길이/초 | 양발목 속도 2개 |
| 43 | `right_ankle_speed` | 오른발목, 같은 방식 | 상체 길이/초 | |
| 44 | `left_wrist_speed` | 왼손목, 같은 방식 | 상체 길이/초 | **양발과 문 사이 거리 2개 대신** |
| 45 | `right_wrist_speed` | 오른손목, 같은 방식 | 상체 길이/초 | |

- 선정 이유
  - 상체 기울기·BBox 비율·골반-바닥 높이: 눕기·앉기·쓰러짐처럼 **몸이 수직인지 수평인지**를 직접 표현한다.
  - 발목 속도: 차기·몸통박치기처럼 **다리·몸 이동이 빠른 동작**을 표현한다.
  - 손목 속도: 우리 17개 동작에는 문 차기가 없고, 촬영 장소에 키오스크·냉동고·문 같은 실제 물체가 없어 거리를 계산할 수 없었다. 대신 내려치기·던지기·휘두르기·밀치기처럼 **손을 빠르게 쓰는 이상행동이 6개**라서 강사님 표의 "좌·우 손목 속도 +2"를 넣었다.
- 단위·결측 처리
  - 길이는 모두 **상체 길이**로 나눴다. 측면에서는 두 어깨가 겹쳐 어깨 폭이 0에 가까워지기 때문이다 (3일차 미니챌린지에서 확인).
  - 속도는 60fps에서 바로 앞 Frame과 비교하면 좌표 떨림이 크게 잡혀서 **0.1초(6 Frame) 간격**으로 계산했다. 그래서 Clip마다 처음 6 Frame은 속도가 비어 있다.
  - Pose 실패 Frame도 지우지 않고 `pose_ok=0`으로 남겼다. `feature_ok`(38-D)와 `feature45_ok`(45-D)를 따로 표시했다.

- Sequence CSV 생성 수: 170개 (`data/team_projects/team03/sequences/<clip_id>_sequence.csv`)
  - 70,159 Frame 중 38-D 성공 69,926, 45-D 성공 68,813

### 동작 시작·끝 구간

- 영상은 자르지 않고 구간만 기록했다: `data/team_projects/team03/manifests/action_segments.csv`, Sequence의 `in_action` 열
- 방법: 몸 관절 12개의 0.1초 이동 거리(상체 길이/초)를 0.5초 평균으로 부드럽게 한 뒤, 영상마다 정한 기준값을 넘은 첫 시점과 마지막 시점에 앞뒤 0.2초 여유를 둠 (`scripts/team03/team_action_segments.py`)
- 결과: 앞에서 뺀 시간 중앙값 0.57초, 뒤에서 뺀 시간 중앙값 0.11초, 동작 구간 중앙값 4.62초 (동작 구간 Frame 60,533개)
- 확인: 영상마다 처음·시작·가운데·끝·마지막 장면에 Skeleton을 그린 사진으로 확인했다 (`scripts/team03/team_review_sheets.py`)

### 팀 Feature로 확인한 점 (동작 구간, 사람 5명 중앙값 — `reports/team_projects/team03/tables/feature_by_action.csv`)

- **측면에서는 눕기·쓰러짐이 잘 구분됐다.** A009·A010·A016 측면은 BBox 비율 2.4~3.2, 골반-바닥 높이 0.3~0.5, 팀 자세 규칙의 lie 비율 0.42~0.55였다. 서서 하는 동작(BBox 비율 약 0.5, 골반-바닥 높이 약 1.6)과 차이가 컸다.
- **정면에서는 누운 자세가 잘 안 잡혔다.** 사람이 카메라 쪽으로 길게 누우면 화면에서는 몸이 짧아 보인다. A009·A010 정면은 BBox 비율이 0.75·0.83으로 넓어지지 않았고 lie 비율은 0.15·0.00이었다. A009 정면은 머리가 골반보다 화면 아래에 보여 상체 기울기가 178°까지 나왔다. 2D 한 방향만으로는 한계가 있어 **정면·측면을 함께 쓰는 것**이 필요하다.
- 손목 속도는 던지기(7.5~9.5)·샷건(5.7~7.7)·키오스크 두드리기 측면(8.4)에서 높았고, 밀치기·잡고 흔들기(2.5~4.3)는 낮았다.
- 발목 속도는 앞차기 측면에서 6.6으로 다른 서서 하는 동작(약 0.8~2.5)보다 높았다. 몸통 박치기 측면도 몸 전체가 움직여 4.7로 높았다. 정면에서는 다리가 카메라 쪽으로 움직여 2.6에 그쳤다.
- 위에 앉기(A008)는 정면에서 무릎이 펴져 보여 sit 규칙이 0%였고, 측면에서 42%였다.

## 4. QA 결과

자동 QA는 5일차 27번 기준(Pose 비율·한 사람 비율·평균 Confidence·중요 관절 저신뢰도)을 그대로 쓰고, 중요 관절만 3팀 동작에 맞게 다시 정했다 (`scripts/team03/team_keypoint_qa.py`).
판단을 돕는 참고 열 3개를 추가했다: 연속 Pose 누락 최대 시간, 정규화 기준 관절(어깨·골반) 저신뢰도 비율, 더 잘 보이는 쪽 관절의 저신뢰도 비율.

- AUTO_PASS: 166
- CHECK: 4 → Human QA

| Clip | CHECK 원인 | 원본 확인 결과 | 결정 |
|---|---|---|---|
| S03_A002_T02 던지기 측면 | 중요 관절 저신뢰도 0.207 (> 0.2) | 측면이라 반대쪽 팔이 가려짐. 보이는 쪽 팔은 저신뢰도 0% | PASS |
| S03_A009_T02 앞으로 눕기 측면 | 한 사람 비율 0.866 | 누운 사람을 다리·상체 BBox 2개로 중복 검출. 대표 인물은 전신 BBox | PASS |
| S02_A010_T01 뒤로 눕기 정면 | Pose 비율 0.887, 한 사람 비율 0.887 | 카메라에서 먼 곳에 웅크려 누운 구간 171 Frame Pose 누락. 사람은 화면 안 | PASS (재촬영 후보) |
| S04_A017_T02 앞으로 넘어지기 측면 | 한 사람 비율 0.714 | 화면 오른쪽 아래 끝에 다른 사람 일부가 잡힘(conf 0.30) | PASS (재촬영 후보) |

- PASS: 170
- REVIEW: 위 4개 (CHECK → 원본 확인 후 PASS)
- RESHOOT: 0 (재촬영 후보 2개는 팀 판단 필요)
- EXCLUDE: 0
- 처리 결정과 이유: 4개 모두 촬영 대상이 동작을 끝까지 수행했고 대표 인물 선택이 올바라서 사용했다. 다만 S02_A010_T01은 누운 구간 일부가 비어 있고, S04_A017_T02는 화면 끝에 다른 사람이 보여 다시 찍으면 더 좋다.
- AUTO_PASS 166개도 사람별 대표 장면 모음과 구간 확인 사진으로 동작 수행·전신 포함을 확인했다.

## 5. 데이터 분할

- Train Subject / MP4 수: S01, S02, S03 / 102
- Validation Subject / MP4 수: S04 / 34
- Test Subject / MP4 수: S05 / 34
- 같은 Subject·원본의 분할 중복 여부: 없음 (`31_check_split_leakage.py` → Leakage check: OK, 17개 Action 모두 각 Split에 존재)
- 같은 사람의 정면·측면 영상은 항상 같은 Split에 들어간다 (사람 단위로 나눴기 때문).
- 설정: `configs/team03_subject_split.json`, 결과: `data/team_projects/team03/splits/`

## 6. VICON 활용 계획

- VICON 원본 관절·손 Bone 확인 결과: 아직 확인하지 않았다 (VICON 촬영 목록·FBX 파일 필요).
- RGB와의 관절 매핑·정규화 기준: RGB는 COCO 17 + 골반 중심·어깨 폭(38-D) / 상체 길이(팀 Feature) 기준. VICON은 같은 관절만 골라 2D로 투영한 뒤 같은 기준을 적용할 예정이다.
- VICON 데이터와 RGB 데이터의 구분 방식: `source_type` 열(RGB / VICON)로 구분한다. VICON에는 YOLO Confidence가 없으므로 1.0으로 채우지 않고 비워 둔다.

## 7. 다음 단계

- 개선할 데이터 품질 문제:
  - 재촬영 후보 2개 (S02_A010_T01, S04_A017_T02)
  - 정면에서 누운 자세가 잘 안 잡히는 문제 → 카메라 높이·각도 조정 또는 정면·측면 결합
- **필요한 자료 (아직 없음)**
  1. **정상행동 비교 영상**: 강사님 안내의 "문 앞에서 발만 들고 차지 않은 정상 행동" 같은 오탐 비교 영상. 예) 바닥 물건 줍기(구토·숙이기와 비교), 의자에 앉아 쉬기(앉기와 비교), 신발끈 묶기(무릎꿇기와 비교), 손 흔들기(내려치기와 비교)
  2. **문·바닥·물체 위치(ROI)**: 실제 매장처럼 키오스크·냉동고·문이 화면에 있는 영상, 또는 그 위치를 표시한 정보. 지금 영상은 강의실이라 물체와의 거리 Feature를 만들 수 없다.
  3. **VICON 촬영 목록과 FBX**: VICON 동작 코드와 A001~A017의 연결표
  4. 가능하면 사람·동작마다 **Take 추가** (지금은 방향별 1개뿐)
- 5일차 Dataset QA·Split에서 확인할 사항: 완료 (Approved 170, Leakage OK)
- 6일차 모델 학습 전 준비할 사항: 교재 38-D와 3팀 45-D를 같은 Split(S01~S03 / S04 / S05)으로 비교, 정면·측면을 따로 쓸지 함께 쓸지 결정

### 6일차 Baseline 결과 (이 Dataset으로 학습·평가)

- Random Forest (교재 설정), 30 Step Sequence, Train S01~S03 / Validation S04 / Test S05
- Validation: 38-D 0.6176 / Macro F1 0.6235 → 45-D 0.7353 / 0.7196
- Test (S05, 처음 보는 사람): 45-D **0.6471 / Macro F1 0.6020** (38-D 0.5882 / 0.5652)
- 남은 혼동: 물체 없이 팔로 하는 동작끼리(밀치기 ↔ 잡고 흔들기 ↔ 샷건, 휘두르기 → 몸통 박치기) → 물체 위치(ROI) 정보가 필요하다는 근거
- Validation과 Test 차이가 크다 (0.74 vs 0.65) → 사람 한 명으로 낸 점수는 흔들리므로 5명을 돌아가며 평가할 필요가 있다
- 자세한 내용: `reports/day06/challenge/mini_challenge_result.md`
- Feature를 38-D에서 45-D까지 늘린 과정과 단계별 효과: `reports/team_projects/team03/feature_expansion_log.md`

## 8. 완료 체크리스트

- [x] 170개 MP4의 고유 `clip_id`가 있고 Label·Subject·Take(방향)를 확인했다.
- [x] 17개 Action별 영상 수(각 10개)와 누락·중복을 확인했다.
- [x] 모든 MP4의 재생 상태·FPS·Frame 수를 자동 검사했다.
- [x] 대표 Frame과 이상 사례의 실제 영상 내용을 확인했다.
- [x] 4일차 코드를 Clip별 고유 저장이 가능하도록 확장했다.
- [x] 각 Frame의 Pose·Confidence·정규화·Feature 결과와 실패 여부를 기록했다.
- [x] 3일차에서 정한 팀별 추가 Feature를 구현했다 (45-D).
- [x] 영상별 QA 통계와 PASS / REVIEW / RESHOOT / EXCLUDE를 정리했다.
- [x] RGB 원본 좌표와 정규화 Feature를 구분했다. (VICON 데이터는 아직 없음)
- [x] Train / Validation / Test를 Subject 기준으로 나누었다.
- [x] 좌우 반전 등 증강은 아직 하지 않았다 (Train 분리 후 필요할 때 적용).
- [x] 팀별 결과 보고서와 다음 실습에서 처리할 문제를 정리했다.

## 산출물 위치

| 산출물 | 위치 |
|---|---|
| ① Action 정의표 | `configs/team03_action_classes.csv` |
| ② Clip Manifest | `configs/team03_clip_manifest.csv` |
| ③ 영상 검사표 | `data/team_projects/team03/manifests/keypoint_dataset_manifest.csv` (FPS·Frame·해상도) |
| ④ Keypoint / Sequence CSV | `data/team_projects/team03/keypoints/`, `data/team_projects/team03/sequences/` |
| ⑤ Sequence QA 표 | `data/team_projects/team03/qa/keypoint_clip_qa.csv`, `keypoint_clip_review.csv` |
| ⑥ 팀별 Feature 명세 | 이 문서 3장 |
| ⑦ Dataset Split 표 | `data/team_projects/team03/splits/train.csv`, `val.csv`, `test.csv` |
| ⑧ 결과 보고서 | 이 문서 |

영상·CSV·사진은 용량이 크고 사람 얼굴이 담긴 데이터라 Git에 올리지 않았다. 실명과 가명 ID의 연결표도 `data/` 안에만 보관한다.
