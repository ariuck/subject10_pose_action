# Subject 10 - Pose Estimation & Action Analysis

YOLO Pose로 사람의 관절을 찾고, 관절 좌표의 시간 변화를 Random Forest로 분류해 동작을 인식하는 프로젝트입니다.
교재 실습(6개 동작)과 함께 **3팀 무인 상점 이상행동 17개 동작**(5명 × 17동작 × 정면·측면 = 170개 영상)에 같은 Pipeline을 적용했습니다.

- 진행하면서 겪은 문제와 해결 과정, 개선 결과: [`reports/improvement_log.md`](reports/improvement_log.md)

## Day 01 - Pose 추론 환경과 입력

```text
JPG / MP4 / IP Webcam → Frame → YOLO Pose → Person BBox → COCO 17 Keypoint → Skeleton
```

- 환경: Python 3.12, PyTorch 2.15 + CUDA, RTX 3060 (`reports/day01/environment.txt`)
- COCO8-Pose 이미지 8장, Pose Demo MP4(30 FPS, 60 Frame)에 Pose 추론
- Mini Challenge (`reports/day01/challenge/`)
  - COCO8-Pose 추가 이미지: 049번은 잘 됨, 110번(8명 겹침)은 불안정
  - 직접 촬영한 4가지 자세: 옆으로 서기, 쪼그려 앉기, 엎드리기, 벽에 붙어 웅크리기
  - 가장 잘 된 자세는 쪼그려 앉기, 가장 불안정한 자세는 벽에 붙어 웅크리기(사람 검출 실패)

```text
scripts/
├── 00_check_environment.py
├── 01_image_input.py
├── 02_video_input.py
├── 03_pose_image.py
├── 04_pose_video.py
└── 05_ip_webcam_capture.py
```

## Day 02 - Keypoint 좌표와 Pose 품질 분석

```text
이미지 · MP4 → YOLO Pose → BBox + 17 Keypoint → Pixel / Normalized 좌표
→ Keypoint Confidence → Threshold → 조건별 · Frame별 Pose 품질 비교
```

**조건별 이미지 비교**
- 정면: 평균 Keypoint Confidence 0.975, LOW 관절 0개
- 측면: 평균 0.754, LOW 관절 4개 (카메라 반대쪽 눈·귀·팔)
- 거리: BBox 면적이 0.300 → 0.149로 줄어도 Confidence는 0.975 → 0.971로 유지됐다.

**직접 촬영한 동작 영상 36개**
- 1080×1920, 60 FPS, 같은 동작을 정면·측면 카메라 2대로 촬영했다.
- 정면 vs 측면

  | 카메라 | 평균 Confidence | LOW 관절 |
  |---|---|---|
  | 정면 | 0.96~0.98 | 거의 없음 |
  | 측면 | 0.82~0.86 | 거의 모든 Frame에서 4개 (반대쪽 눈·귀·손목) |

- 바닥에 눕거나 엎드린 동작이 가장 불안정했다. 평균 0.724, LOW 관절 최대 11개.
- 미검출: 벽에 붙어 바닥에 웅크린 장면에서 1,512 Frame 중 171 Frame
- 오검출: 화면 구석의 물체를 `person 0.35`로 검출했다. Person Confidence 기준 0.25를 넘었기 때문이다.

**Team Mini Challenge (정면 vs 측면)**
- 정면은 평균 0.979 / 0.976, LOW 관절 0개였다.
- 측면은 평균 0.794 / 0.785, LOW 관절 4개였다.
- Threshold 0.30에서는 가려진 팔이 앞쪽 팔에 겹쳐 그려졌다. 0.50에서는 제외됐다.

```text
src/pose_utils.py
scripts/
├── 06_check_day02_data.py
├── 07_pose_image_detail.py
├── 08_draw_skeleton_quality.py
├── 09_compare_pose_images.py
└── 10_pose_video_quality.py
```

## Day 03 - Joint Angle, Normalization & Frame Feature

```text
COCO 17 Keypoint + Confidence → 관절각도 (Confidence 낮으면 SKIP)
→ Hip Center 이동 + Shoulder Width 정규화
→ Normalized 좌표 34개 + 관절각도 4개 = 38-D Frame Feature → 영상의 각 Frame에 반복
```

**정규화 효과 (직접 촬영한 Near / Far, 같은 사람·같은 자세)**

| 비교 | 평균 관절 거리 |
|---|---|
| Pixel 좌표 | 163.60 |
| Normalized 좌표 | 0.086 |

- 좌우 반전: x 부호를 바꾸고 Left / Right 관절 번호도 바꿨다. 왼팔 올리기와 오른팔 올리기의 차이가 0.609에서 0.121로 줄었다.

**문제 해결: Confidence가 높은데 관절 위치가 틀림**
- 상황: `bend_pose.jpg`에서 왼팔꿈치·손목이 실제 왼팔이 아니라 오른팔 위에 겹쳐 찍혔다.
- 원인: Confidence가 0.817 / 0.884로 높아서 Confidence 검사로는 걸러지지 않았다.
- 해결: **겹쳐 찍힌 팔·다리 검사**를 추가했다. 한쪽 관절이 반대쪽 팔다리 선에 거의 붙어 있으면(overlap < 0.05) Confidence가 낮은 쪽 각도를 SKIP한다.
- 결과: 틀린 각도 1개를 SKIP했다. (`reports/day03/bend_pose_left_right_check.md`)

**Team Mini Challenge - 3팀 17개 동작 (정면·측면 36개 영상)**

| 확인 항목 | 결과 |
|---|---|
| 관절각도 | 무릎은 서기 161~173°, 눕기 159~164°로 둘 다 펴져 있어서 구분이 안 됨 |
| 어깨 폭 정규화 (측면 눕기 Near / Far) | 측면에서 두 어깨가 겹쳐 어깨 폭이 18.99px까지 작아짐 → 차이 `8.199` |
| 상체 길이 정규화 | 눕기 Near / Far `0.438`, 서기 Near / Far `0.167`, 눕기 vs 서기 `1.209` |
| 영상 (뒤로 기절하기, 808 Frame) | 상체 기울기: 서기 약 7° → 누움 77~93° (3.3~10.6초 유지) → 서기 약 1° |

- 눕기·앉기·숙이기처럼 자세가 유지되는 동작은 상체 기울기와 골반-발목 높이로 구분됐다.
- 동작마다 잘 보인 Feature가 달랐다.
  - 내려치기: 팔 올림
  - 던지기: 손목 속도
  - 앞차기: 발목 속도
  - 몸통박치기: 골반 속도
- 흔들기·밀치기처럼 물체와 닿는 동작은 Pose만으로 부족해서 객체 검출이 필요하다.
- 결과: `reports/day03/challenge/feature_design.md`, `reports/day03/challenge/team3_action_summary.md`

```text
src/feature_utils.py, src/team_features.py
scripts/11_check_day03_data.py ~ 18_video_frame_features.py
scripts/team3/   (자세 규칙, 팀 Feature, 상체 길이 정규화, 36개 영상 요약)
```

## Day 04 - Action Clip & Sequence

```text
Action Clip → Frame → YOLO Pose → 17 Keypoint → Normalization + Joint Angle
→ 38-D Frame Feature → 시간 순서 유지 → Sequence → Action Label
```

- 교재 6개 동작(824 Frame): Pose 성공 824 / 824, Feature 생성 824 / 824
- 동작별로 잘 변한 Feature

  | 동작 | Feature 변화 |
  |---|---|
  | wave | 오른쪽 Wrist y `-0.03 → -3.05 → -0.04` |
  | sit_stand | Knee `160° → 178°` |
  | walk_turn_walk | Hip Center x `472 → 279 → 462` |

**Mini Challenge - 3팀 17개 동작 (정면·측면 36개 영상)**
- 18,950 Frame 중 Pose 성공 18,742 (98.9%). 실패 Frame은 모두 바닥에서 몸을 말거나 엎드린 장면이었다.
- 같은 이름의 정면·측면 결과가 서로 덮어써져서, Label에 `_front` / `_side`를 붙여 36개를 따로 저장했다.
- 휴대폰 영상은 영상 정보의 Frame 수가 실제보다 1~6개 많았다. 그래서 실제로 읽히는 Frame 기준으로 대표 Frame을 뽑는 사본을 만들었다(108장).
- 측면에서는 어깨 폭이 작아져 정규화 좌표가 튀었다. 인접 Row 변화량 중앙값이 정면 0.227, 측면 4.475였다.
- 뒤로 기절하기 vs 앞으로 넘어지기
  - 시작·끝 자세가 같아서 중간의 시간 변화로만 구분됐다.
  - 기절은 골반이 1초 안에 약 190px 내려가 머물렀다.
  - 넘어지기는 골반이 1초에 약 380px 앞으로 이동했다.

**팀 확장 - 3팀 RGB 영상 170개, 45-D Sequence**

```text
S0x_A0nn_T0t.mp4 (5명 × 17동작 × 정면·측면, 실명 대신 S01~S05)
→ Clip Manifest → Keypoint Dataset (Pixel 좌표 + Confidence + BBox)
→ 자동 QA (AUTO_PASS 166 / CHECK 4) → Human QA (PASS 170)
→ 동작 시작·끝 구간 (in_action 표시)
→ 38-D + 3팀 7개 = 45-D Sequence 170개
```

| 추가 Feature (+7) | 의미 |
|---|---|
| `torso_tilt_deg` | 상체 기울기 (0° 서기, 90° 수평) |
| `bbox_aspect_ratio` | 사람 BBox 가로/세로 비율 |
| `hip_floor_height` | 골반과 바닥(BBox 아래 끝)의 상대 높이 / 상체 길이 |
| `left/right_ankle_speed` | 양발목 속도 (상체 길이/초, 0.1초 간격) |
| `left/right_wrist_speed` | 양손목 속도 (문·물체 위치는 영상에 없어서 대체) |

- 70,159 Frame 중 Pose 성공 69,926 (99.7%), 45-D 성공 68,813
- 측면 눕기·쓰러짐은 서서 하는 동작과 잘 구분됐다.
  - 눕기·쓰러짐: BBox 비율 2.4~3.2, 골반-바닥 높이 0.3~0.5
  - 서서 하는 동작: 약 0.5 / 1.6
- 보고서: `reports/team_projects/team03/team03_data_preparation_report.md`

## Day 05 - Keypoint Dataset QA & Subject-wise Split

```text
MP4 → COCO 17 Keypoint Dataset → 자동 QA → Human QA → Approved Clips
→ Subject-wise Split → Leakage check
```

| | 교재 (6동작 × 3명) | 3팀 (17동작 × 5명 × 정면·측면) |
|---|---|---|
| Approved | 18 | 170 |
| Train / Validation / Test | S01 / S02 / S03 | S01~S03 (102) / S04 (34) / S05 (34) |
| Leakage check | OK | OK |

- 자동 QA CHECK 5개는 원본을 확인해 보니 **옷걸이에 걸린 외투**를 두 번째 사람으로 오검출한 것이었다. 모두 PASS 처리했다.
- Frame을 무작위로 나누면 같은 사람·같은 Clip이 Train과 Test에 함께 들어간다(Subject 3명, Clip 18개 모두 겹침). 그래서 평가는 Subject 기준으로 나눴다.
- 3팀 영상에 쓰려고 동작표가 고정된 교재 26·27번의 3팀 사본을 만들었다. 상대경로 오류도 함께 고쳤다.
- 결과: `reports/day05/challenge/mini_challenge_result.md`

```text
scripts/25_check_day05_data.py ~ 32_demo_frame_split_leakage.py
scripts/team03/team_extract_keypoint_dataset.py, team_keypoint_qa.py
```

## Day 06 - Action Recognition Baseline

```text
Keypoint CSV → 38-D Frame Feature → Sequence Resampling (30 Step) → Sequence Feature (1,140)
→ Random Forest (300 Tree) → Validation → 오분류 분석 → Feature 비교 → Final Test
```

**3팀 17개 이상행동 (Train 102 / Validation 34 / Test 34)**

Baseline에서 한 가지 조건만 바꿔 Validation으로 비교했다.

| 조건 | Feature | Validation Accuracy | Validation Macro F1 |
|---|---:|---:|---:|
| Baseline (38-D) | 1,140 | 0.7353 | 0.7373 |
| Static (가운데 Frame) | 38 | 0.5882 | 0.5941 |
| 좌표만 | 1,020 | 0.6765 | 0.6728 |
| **3팀 45-D** | 1,350 | 0.7647 | **0.7725** |
| 동작 구간만 | 1,140 | 0.7941 | 0.7588 |
| 정면만 / 측면만 | 1,140 | 0.6471 / 0.6471 | 0.5882 / 0.5529 |

- Sequence가 Static보다, 각도를 넣은 것이 좌표만 쓴 것보다 높았다.
- 남은 혼동은 물체 없이 팔로 하는 동작끼리였다. (밀치기 ↔ 흔들기, 내려치기·던지기·휘두르기)
- Feature를 한 단계씩 늘려 본 Validation Macro F1

  | 단계 | Macro F1 |
  |---|---:|
  | 38-D | 0.7373 |
  | + 자세 3개 | 0.7647 |
  | + 발목 속도 | 0.7922 |
  | + 손목 속도 | 0.7725 |

  늘릴수록 항상 좋아지지는 않았다.
- 최종 조건은 45-D로 정했다. Test를 보기 전에 정한 기준(Validation Macro F1 최고)으로 골랐다.
- **Final Test (S05)**: Accuracy 0.7647 / Macro F1 0.7529 (26 / 34). 같은 Test에서 38-D Baseline은 0.6765 / 0.6630이었다.
- 교재 데이터(18 Clip): Test S03 6 / 6. Clip 수가 매우 적어서 참고용 수치다.
- 결과: `reports/day06/challenge/mini_challenge_result.md`, `reports/team_projects/team03/feature_expansion_log.md`

```text
configs/day06_baseline.json
src/sequence_utils.py
scripts/33_check_day06_inputs.py ~ 40_evaluate_test.py
scripts/team03/team_day06_baseline.py   (3팀 조건 비교 · 모델 저장)
```

## Day 07 - Sliding Window Realtime Action Recognition

학습한 Random Forest를 저장영상과 스마트폰 카메라 Stream(DroidCam, 1280×720)의 연속 Frame에 연결했다.

```text
Saved MP4 / Phone Camera Stream
→ YOLO Pose → 38-D Frame Feature
→ Timed Sliding Window (최근 5초, 80% 이상 차면 예측)
→ 30 Step Resampling → Random Forest (0.25초마다)
→ Confidence Threshold (미만이면 UNKNOWN)
→ Stabilization (최근 5개 중 3표) → Stable Action
```

**교재 모델 저장영상 실험 (S02)**

| 실험 | Prediction | UNKNOWN 비율 | 평균 Confidence |
|---|---:|---:|---:|
| wave, Window 5초 | 0 | - | - |
| wave, Window 3초 / 2초 | 5 / 8 | 1.0 / 1.0 | 0.249 / 0.260 |
| walk_turn_walk, 기본 | 21 | 1.0 | 0.231 |
| walk_turn_walk, Threshold 0.40 / 0.70 | 21 / 21 | 1.0 / 1.0 | 0.231 |
| walk_turn_walk, Buffer 3 / 7 | 21 / 21 | 1.0 / 1.0 | 0.231 |

- 첫 예측은 항상 4초에 나왔다. 5초 Window의 80%를 채워야 하기 때문이다.
- Confidence가 0.2대여서 기본 Threshold(0.55)에서는 모두 UNKNOWN이었다.

**3팀 모델 실시간 실행 - A007 매트 위에 앉기**
- 교재 실시간 코드는 그대로 두고, 모델 경로와 동작 이름표만 3팀 것으로 바꿔 실행하는 파일을 만들었다.
- 입력은 학습에 쓰지 않은 S05 영상(10.2초)이다. Threshold 하나만 바꿔 비교했다.

| Threshold | UNKNOWN 비율 | Stable 일치 비율 | Raw 정답 |
|---:|---:|---:|---:|
| 0.55 | 1.00 | 0.00 | 23 / 25 |
| 0.20 | 0.32 | 0.64 | 23 / 25 |

- 17개 Class 모델은 Confidence가 원래 낮다(0.16~0.33). 그래서 6개 Class용 기준으로는 맞는 예측도 전부 보류됐다.
- Threshold는 모델의 판단을 바꾸지 않고 내보낼지만 정한다. 그래서 기준값은 모델과 Class 수에 맞춰 정해야 한다.
- 결과: [`mini_challenge_summary.md`](reports/day07/challenge/team03/mini_challenge_summary.md), [`runtime_comparison.md`](reports/day07/runtime_comparison.md), [`fps_test.md`](reports/day07/fps_test.md), [`day07_summary.md`](reports/day07/day07_summary.md)

```text
configs/day07_realtime.json
src/realtime_utils.py
scripts/41_check_day07_inputs.py ~ 45_analyze_ip_webcam_log.py
scripts/team03/team_day07_runtime.py   (3팀 모델로 저장영상 · 카메라 실시간 실행)
scripts/team03/team_record_clip.py     (폰 Stream으로 평가용 영상 녹화)
```
