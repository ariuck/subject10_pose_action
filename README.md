# Subject 10 - Pose Estimation & Action Analysis

교과 10 자세추정 이론·실습 프로젝트입니다.

## Day 01 - Pose 전체 구조·개발환경·실제 데이터 입력

### 사용한 입력

- COCO8-Pose
- Ultralytics Pose Demo MP4 (`solution_ci_pose_demo.mp4`)
- 스마트폰 IP Webcam 캡처 이미지 (촬영 예정)

### 학습한 내용

- Pose Estimation과 Action Recognition의 차이
- WSL 프로젝트와 Python 가상환경 구성
- OpenCV / PyTorch / Ultralytics 확인
- GPU / CUDA 확인
- COCO8-Pose 이미지 입력
- YOLO Pose 추론
- Person Bounding Box
- COCO 17 Keypoint
- Skeleton
- 실제 MP4 Frame 처리
- MP4 Frame별 Pose
- IP Webcam 연결 및 Frame 저장
- 직접 촬영 이미지 Pose 추론

### Day 01 Pipeline

```text
JPG / MP4 / IP Webcam
        ↓
Frame
        ↓
YOLO Pose
        ↓
Person Bounding Box
        ↓
COCO 17 Keypoint
        ↓
Skeleton
```

### 주요 실행 파일

```text
scripts/
├── 00_check_environment.py
├── 01_image_input.py
├── 02_video_input.py
├── 03_pose_image.py
├── 04_pose_video.py
└── 05_ip_webcam_capture.py
```

### 주요 결과

- `reports/day01/environment.txt` (Python 3.12, PyTorch 2.15 + CUDA, RTX 3060)
- COCO8-Pose 이미지 8장 Pose 결과 (`reports/day01/pose_*.jpg/.txt`)
- `reports/day01/pose_demo_output.mp4` (30 FPS, 60 Frame, 640 x 360)
- IP Webcam 촬영 이미지 Pose 결과 (촬영 후 추가)
- Mini Challenge (`reports/day01/challenge/`)
  - COCO8-Pose 추가 이미지 분석: 049번 잘 됨, 110번(8명 겹침) 불안정
  - 직접 촬영 영상에서 캡처한 4가지 조건: 옆으로 서기, 쪼그려 앉기, 엎드리기, 벽에 붙어 웅크리기
  - 가장 잘 된 결과: 쪼그려 앉기 / 가장 불안정: 벽에 붙어 웅크리기 (사람 검출 실패)

### 다음 수업

2일차에는 Keypoint 좌표와 Confidence를 자세히 읽고,
촬영 조건에 따라 Pose 품질이 어떻게 달라지는지 분석합니다.

## Day 02 - Keypoint 좌표와 Pose 품질 분석

### 오늘 사용한 입력

- 강사가 준비한 조건별 이미지 6장
  - `front.jpg`
  - `arms_up.jpg`
  - `side.jpg`
  - `partial_body.jpg`
  - `occlusion.jpg`
  - `distance_far.jpg`
- 1일차에서 사용한 실제 MP4
- 추가: 모션캡처 촬영 때 직접 찍은 동작 영상 36개 (`20261113_*.mp4`)
- Team Mini Challenge: 직접 촬영 영상에서 캡처한 정면 2장, 측면 2장

### 오늘 학습한 내용

- Person BBox Confidence와 Keypoint Confidence 구분
- COCO 17 Keypoint Pixel 좌표 확인
- Normalized 좌표 이해
- Keypoint Confidence와 Threshold 이해
- Confidence 기준에 따른 Skeleton 표시
- 여러 이미지의 Pose 품질 CSV 비교
- BBox Area Ratio 비교
- 평균 Keypoint Confidence와 LOW 관절 확인
- MP4의 Frame별 Pose 품질 분석
- 불안정한 Frame을 숫자로 찾고 실제 영상과 연결
- 촬영 조건 하나를 바꾸어 Pose 품질 비교

### 실제 실습에서 확인한 결과

- `front.jpg`
  - 평균 Keypoint Confidence: `0.9754`
  - LOW 관절: `0`
  - 정면 기준 이미지에서 전체적으로 안정적
  - Threshold 0.70 → 17개 모두 / 0.90 → right_ear 제외 / 0.95 → 양쪽 귀 제외

- `side.jpg`
  - 평균 Keypoint Confidence: `0.7541`
  - LOW 관절: `4` (right_eye, right_ear, right_elbow, right_wrist)
  - 이번 6장 중 가장 불안정한 결과

- `front.jpg` VS `distance_far.jpg`
  - BBox Area Ratio `0.300 → 0.149` (사람 크기 약 절반)
  - 평균 Keypoint Confidence는 `0.9754 → 0.9710`
  - LOW 관절은 두 이미지 모두 `0`
  - 이번 거리 범위에서는 Pose 품질이 잘 유지됨

- MP4 Frame 분석 (`pose_quality_demo.mp4`)
  - 사람은 60개 Frame 모두 검출됨
  - 약 `0.80초`의 25~26번 Frame에서 얼굴 주변 Keypoint Confidence가 낮아짐
  - 실제 영상에서 바닥 자세와 얼굴 방향이 함께 나타난 것을 확인

- Team Mini Challenge (방향: 정면 vs 측면)
  - 정면: 평균 Keypoint Confidence `0.979 / 0.976`, LOW 관절 `0 / 0`
  - 측면: 평균 `0.794 / 0.785`, LOW 관절 `4 / 4` (전부 카메라 반대쪽 눈·귀·팔)
  - Threshold 0.30에서는 가려진 팔이 앞쪽 팔과 겹쳐 그려지고, 0.50에서는 제외됨

### 추가 실습 - 직접 촬영한 동작 영상 36개

교재의 `10_pose_video_quality.py`를 수정 없이 사용하고, 입력만 직접 찍은 영상으로 바꾸어 실행했습니다.

```bash
for f in data/day02/videos/20261113_*.mp4; do
  python scripts/10_pose_video_quality.py --source "$f" --max-frames 0
done
```

- 영상: 1080 x 1920(일부 1920 x 1080), 60 FPS, 약 3~27초, 같은 동작을 정면·측면 카메라 2대로 촬영
- 영상별 요약: `reports/day02/tables/huyoung_video_summary.csv`

확인한 결과

- 정면 카메라 vs 측면 카메라 (같은 동작)
  - 정면: 평균 Keypoint Confidence 약 `0.96~0.98`, LOW 관절이 있는 Frame 거의 없음
  - 측면: 평균 약 `0.82~0.86`, 거의 모든 Frame에서 LOW 관절 4개 (반대쪽 눈·귀·손목)
  - 교재의 `side.jpg` 결과와 같은 경향
- 바닥에 눕거나 엎드린 동작이 가장 불안정
  - `20261113_161722`: 평균 `0.724`, LOW 관절 최대 `11`
  - `20261113_161202`: 평균 `0.744`, 엎드린 측면 자세라 얼굴 Keypoint가 거의 보이지 않음
- 사람 자체를 못 찾은 Frame
  - `20261113_161645`: 1512 Frame 중 171 Frame 미검출 (약 15~17초)
    → 벽에 붙어 바닥에 웅크리고 엎드린 장면, 얼굴과 팔다리가 거의 보이지 않음
  - `20261113_161722`(21 Frame), `20261113_162425`(15 Frame)도 바닥 동작 중 미검출
- 오검출
  - `20261113_162450`: 67 Frame에서 사람 2명 검출 → 실제로는 화면 구석의 물체를
    `person 0.35`로 잘못 검출 (Person Confidence 기준 0.25를 넘었기 때문)

### Day 02 Pipeline

```text
이미지 · MP4
        ↓
YOLO Pose
        ↓
Person BBox
+
COCO 17 Keypoint
        ↓
Pixel / Normalized 좌표
        ↓
Keypoint Confidence
        ↓
Threshold
        ↓
조건별 Pose 품질 비교
        ↓
Frame별 Pose 품질 분석
        ↓
숫자와 실제 이미지·영상 연결
```

### 주요 실행 파일

```text
scripts/
├── 06_check_day02_data.py
├── 07_pose_image_detail.py
├── 08_draw_skeleton_quality.py
├── 09_compare_pose_images.py
└── 10_pose_video_quality.py

src/
└── pose_utils.py
```

### 주요 결과 파일

```text
reports/day02/tables/
├── front_keypoints.csv
├── image_quality_summary.csv
├── pose_quality_demo_frame_quality.csv
├── pose_quality_demo_keypoints.csv
├── 20261113_*_frame_quality.csv   (직접 촬영 영상 36개)
├── 20261113_*_keypoints.csv
└── huyoung_video_summary.csv

reports/day02/videos/
├── pose_quality_demo_pose.mp4
└── 20261113_*_pose.mp4

reports/day02/challenge/
└── challenge_notes.md
```

### 다음 수업

3일차에는 2일차에서 확인한 Keypoint를 이용하여
관절각도와 정규화된 Frame Feature를 만듭니다.

```text
Keypoint 좌표
+ Confidence
        ↓
신뢰할 수 있는 관절 확인
        ↓
관절각도
+ 정규화
        ↓
Frame Feature
```

## Day 03 - Joint Angle, Normalization & Frame Feature

> 진행 범위: 교재 1~40번 전체 (관절각도, 정규화, Near / Far 비교, Person A / B 비교, 좌우 반전, 38-D Frame Feature, 영상 Frame Feature, Team Mini Challenge)
> 14~15번은 모션캡처 촬영 때 직접 찍은 영상에서 캡처한 장면을, 16~23번과 26~28번은 교재 제공 데이터를 사용했습니다.
> 17번 코드부터는 우리 팀(3팀 — 무인 상점 이상행동 인식) 주제에 맞게 바꿔 진행했습니다.
> 모션캡처 촬영 때 17개 이상행동을 정면·측면으로 찍은 영상 36개(`configs/team3_actions.csv`)를 사용했습니다.

### 오늘 배운 내용

- COCO 17 Keypoint와 Confidence 확인
- COCO Left / Right가 사람 본인 기준이라는 것
- 세 Keypoint로 두 벡터를 만들어 관절각도 계산
- 낮은 Confidence 관절로 계산한 각도는 SKIP
- Hip Center 기준 몸 중심 이동
- Shoulder Width 기준 크기 정규화
- 정규화 좌표와 관절각도는 서로 다른 Feature
- Near / Far 이미지를 같은 방법으로 정규화
- 정규화 전·후 평균 관절 거리 자동 비교
- 사람이 달라도 비슷한 자세 비교 (Person A / B)
- Keypoint 좌우 반전 (x 부호 반전 + Left / Right 관절 교환)
- 34개 Normalized 좌표 + 4개 관절각도 = 38-D Frame Feature
- 동작을 설명하는 Feature와 상대 위치 Feature
- 팀 17개 동작을 설명하는 Feature 선택(자세 + 움직임)과 규칙 판정
- 영상의 각 Frame에서 38-D Feature 생성
- Feature가 시간에 따라 변하는 과정 확인

### Day 03 핵심 흐름

```text
YOLO Pose
        ↓
COCO 17 Keypoint
+
Confidence
        ↓
관절각도 (팔꿈치·무릎 4개)
        ↓
Hip Center로 몸 중심 이동
        ↓
Shoulder Width로 크기 정규화
        ↓
Normalized Keypoint
        ↓
Near / Far · Person A / B 정규화 전·후 비교
        ↓
좌우 반전 비교
        ↓
Normalized Keypoint 34개
+
Joint Angle 4개
        ↓
38-D Frame Feature
        ↓
동작을 설명하는 Feature 선택 (3팀: 자세 Feature 5개 + 움직임 Feature 3개)
        ↓
영상의 각 Frame에 반복
        ↓
시간에 따른 Feature 변화
        ↓
Sequence의 재료
```

### 오늘 추가한 주요 코드

```text
src/
├── feature_utils.py
└── team_features.py                 (3팀 추가 Feature: 자세 5개 + 움직임 3개, 자세 상태 규칙)

configs/
└── team3_actions.csv                (영상 36개 ↔ 17개 동작·그룹·정면/측면)

scripts/
├── 11_check_day03_data.py
├── 12_joint_angle.py
├── 13_normalize_pose.py
├── 14_compare_normalization.py
├── 15_flip_keypoints.py
├── 16_make_frame_feature.py
├── 17_team_pose_rules.py            (교재 17_arms_up_rule.py를 팀 동작 규칙으로 변경)
├── 18_video_frame_features.py       (38-D + 팀 Feature + BBox 원본 좌표 + 규칙 결과 저장)
├── 18a_team_torso_normalization.py   (팀 추가: 어깨 폭 vs 상체 길이 정규화 비교)
└── 18b_team_action_summary.py        (팀 추가: 36개 영상의 팀 Feature를 동작별로 요약)
```

### 실제 실습에서 확인한 결과

- 데이터 확인 (`11_check_day03_data.py`): 이미지 8장 READY
  - Near / Far 캔버스 크기 같음
  - `arms_up_person_b.jpg`와 `arms_up_near.jpg`는 크기는 같지만 픽셀 단위로 완전히 같지는 않음 (INFO)
- 관절각도 (`12_joint_angle.py`)

| 이미지 | 왼팔꿈치 | 오른팔꿈치 | 왼무릎 | 오른무릎 |
|---|---|---|---|---|
| `arms_up_person_a.jpg` | 132.50° | 125.38° | 174.53° | 173.54° |
| `sit_pose.jpg` | 133.50° | 136.46° | 97.88° | 117.49° |
| `bend_pose.jpg` | **SKIP** | 158.53° | 174.52° | 173.93° |

  - `sit_pose.jpg`는 무릎이 크게 굽혀져 서 있는 자세(약 174°)와 확실히 구분됨
  - 팔꿈치 각도는 서 있는 자세와 앉은 자세에서 비슷함 → 관절각도는 "얼마나 굽혔는지"만 설명함
- 정규화 (`13_normalize_pose.py`, `arms_up_person_a.jpg`)
  - Hip Center `(326.48, 581.34)`, Shoulder Width `132.85 px`
  - 골반 중심이 `(0, 0)`이 되고, 어깨 폭이 `1.0`이 되도록 바뀜
  - 왼손목 `(423.5, 118.8) px` → `(0.730, -3.482)`, 오른손목 `(263.3, 116.4) px` → `(-0.476, -3.500)`
  - 손목 y가 어깨 y(-1.96)보다 작아 손목이 어깨보다 위에 있다는 것을 정규화 좌표로도 확인

### 14~15번 - Near / Far 정규화 비교 (직접 촬영한 자료)

같은 사람이 같은 자세(팔을 내리고 정면으로 선 자세)로 카메라에서 가까이·멀리 선 장면을 사용했다.

| 파일 | 원본 영상 | 시간 | 해상도 |
|---|---|---|---|
| `data/day03/images/stand_near.jpg` | `20261113_152953.mp4` | 0.10초 | 1080 x 1920 |
| `data/day03/images/stand_far.jpg` | `20261113_153253.mp4` | 0.10초 | 1080 x 1920 |

교재의 `arms_up_far.jpg`는 같은 캔버스에서 사람을 작게 만든 이미지지만,
이번에는 실제로 카메라에서 멀리 떨어져 찍은 장면이다.

**14번 - 두 이미지를 같은 방법으로 정규화**

```bash
python scripts/13_normalize_pose.py --source data/day03/images/stand_near.jpg
python scripts/13_normalize_pose.py --source data/day03/images/stand_far.jpg
```

| 항목 | Near | Far |
|---|---|---|
| Shoulder Width | 246.24 px | 133.35 px |
| Hip Center | (510.12, 1078.84) | (525.14, 979.74) |
| Left Wrist Pixel | (681.1, 1055.3) | (611.9, 960.6) |
| Left Wrist Normalized | (0.694, -0.095) | (0.650, -0.143) |
| Left Shoulder Normalized | (0.465, -1.633) | (0.386, -1.693) |

- Far에서는 어깨 폭이 Near의 약 0.54배로 작게 측정되어, 같은 관절의 Pixel 좌표가 크게 달라졌다.
- 정규화 후에는 같은 관절의 좌표가 서로 비슷한 값이 되었다.

**15번 - 정규화 전·후 평균 관절 거리 비교**

```bash
python scripts/14_compare_normalization.py \
  --a data/day03/images/stand_near.jpg \
  --b data/day03/images/stand_far.jpg \
  --output normalization_near_far.csv
```

| 비교 | 평균 관절 거리 |
|---|---|
| Pixel 좌표 | `163.60` |
| Normalized 좌표 | `0.086` |

- 정규화 전에는 화면 속 사람 크기 차이 때문에 17개 관절이 평균 163.60 px 떨어져 있었다.
- Hip Center와 Shoulder Width로 정규화한 뒤에는 평균 차이가 0.086으로 줄었다.
- 관절별로 보면 골반(0.007)과 팔꿈치·손목(0.01~0.07)은 거의 같아졌고,
  오른발목(0.161)과 얼굴 관절(0.13~0.15)에 차이가 가장 많이 남았다.
  Far에서 발 간격이 Near보다 조금 좁고, 멀리서 찍어 얼굴이 작게 보였기 때문이다.
- 정규화가 차이를 0으로 만드는 것은 아니지만, 같은 자세를 더 비슷한 숫자로 표현해 준다는 것을 확인했다.

### 16번 - 사람이 달라도 비슷한 Arms Up 자세 비교

```bash
python scripts/14_compare_normalization.py \
  --a data/day03/images/arms_up_person_a.jpg \
  --b data/day03/images/arms_up_person_b.jpg \
  --output normalization_person_a_b.csv
```

| 항목 | 결과 |
|---|---|
| Person A / B 어깨 폭 | 132.85 px / 160.25 px |
| Pixel 평균 관절 거리 | `69.40` |
| Normalized 평균 관절 거리 | `0.339` |

- 정규화 후 차이가 크게 줄었지만, 같은 사람 Near / Far(0.086)보다는 크게 남았다.
- 사람마다 체형과 팔을 든 각도가 달라서, 정규화로 위치·크기 차이는 줄어도 자세 자체의 차이는 남는다.

### 17~18번 - 좌우 반전 비교

```bash
python scripts/15_flip_keypoints.py \
  --left data/day03/images/left_arm_up.jpg \
  --right data/day03/images/right_arm_up.jpg
```

| 비교 | 평균 관절 거리 |
|---|---|
| 실제 Left Arm Up vs 실제 Right Arm Up | `0.609` |
| 반전한 Left Arm Up vs 실제 Right Arm Up | `0.121` |

| 관절 | 반전 전 | 반전 후 |
|---|---|---|
| left_elbow | 1.342 | 0.188 |
| right_elbow | 1.532 | 0.149 |
| left_wrist | 3.096 | 0.221 |
| right_wrist | 2.994 | 0.357 |
| left_knee | 0.104 | 0.109 |
| left_ankle | 0.135 | 0.221 |

- 팔을 올린 쪽을 가장 직접 보여 주는 손목·팔꿈치에서 차이가 크게 줄었다.
- 무릎·발목은 반전 후 비슷하거나 조금 커졌다. 두 사진이 거울처럼 복사한 것이 아니라 따로 찍은 사진이라 다리 위치가 조금 다르기 때문이다.
- 좌우 반전은 x 좌표 부호만 바꾸는 것이 아니라 Left / Right 관절 번호도 바꿔야 한다.
- 왼팔 올리기를 반전해 오른팔 올리기 데이터로 쓰면 데이터 보강에 사용할 수 있지만, 정답 라벨도 오른팔 올리기로 바꿔야 한다.

### 19~21번 - 38-D Frame Feature

```bash
python scripts/16_make_frame_feature.py
python scripts/16_make_frame_feature.py --source data/day03/images/arms_up_person_b.jpg
python scripts/16_make_frame_feature.py --source data/day03/images/arms_up_far.jpg
```

- `arms_up_person_a.jpg`: 평균 Confidence 0.957, LOW 0개, **Feature count 38**
  - 관절각도: 팔꿈치 132.50° / 125.38°, 무릎 174.53° / 173.54°
  - 팔꿈치는 어느 정도 굽혀져 있고, 무릎은 거의 곧게 펴져 있음
- 한 순간의 자세가 CSV 한 행(34개 정규화 좌표 + 4개 관절각도)으로 저장됨

| 이미지 | 평균 Confidence | LOW Keypoint | 어깨 폭 |
|---|---|---|---|
| Person A | 0.957 | 0 | 132.85 px |
| Person B | 0.958 | 0 | 160.25 px |
| Far | 0.955 | 0 | 103.78 px |

- Person B와 Far는 어깨 폭이 크게 다르지만, 정규화된 17개 Keypoint의 평균 차이는 `0.143`으로 작았다.
- Person A와 B는 `0.338`로 더 크게 차이 났다. 화면 속 크기 차이보다 사람·자세 차이가 Feature에 더 크게 남는다.

### 22~23번 - 동작을 설명하는 Feature

- 38개의 숫자가 모든 동작에서 똑같이 중요한 것은 아니다.
  - Arms Up: 손목·어깨 위치, 팔꿈치 각도
  - Sit: 무릎 각도, 골반과 무릎의 상대 위치
- 상대 위치 Feature 예: `left_wrist_y - left_shoulder_y`
  - 0보다 작으면 손목이 어깨보다 위, 0보다 크면 아래
  - `arms_up_person_a.jpg`: 왼손목 y `-3.482` - 왼어깨 y `-1.956` = `-1.526` → 손목이 어깨보다 위

### 실습 중 확인한 문제 - bend_pose.jpg의 왼팔 (SKIP)

| 관절 | 코드 출력 | 최종 기록 |
|---|---|---|
| left_elbow_angle | 155.81° | **SKIP (틀림)** |
| right_elbow_angle | 158.53° | 158.53° |
| left_knee_angle | 174.52° | 174.52° |
| right_knee_angle | 173.93° | 173.93° |

- 결과 이미지를 확인하니 오른팔(카메라 쪽 팔)은 맞게 찍혔지만,
  **왼팔의 팔꿈치·손목이 실제 왼팔이 아니라 오른팔 위에 겹쳐 찍혔다.**
  실제 왼팔은 몸 반대편에서 아래로 늘어져 손이 허벅지 옆에 있다.
- 따라서 `left_elbow_angle 155.81°`는 실제 왼팔의 각도가 아니므로 SKIP 하고 사용하지 않는다.
- 왼팔꿈치 0.817, 왼손목 0.884로 Confidence가 높아서 원래 코드의 Confidence 검사로는 걸러지지 않았다.
- 그래서 `12_joint_angle.py`에 **겹쳐 찍힌 팔·다리 검사**를 추가했다.
  한쪽 팔(다리)의 관절이 반대쪽 팔(다리) 선 위에 거의 붙어 있으면(overlap < 0.05)
  Confidence가 더 낮은 쪽 각도를 SKIP 한다.
  ```text
  left_elbow_angle    : SKIP (right_elbow와 겹쳐 찍힘, overlap=0.035)
  ```
- 3일차 이미지 8장 중 SKIP 된 것은 bend_pose의 왼팔꿈치 하나뿐이다.
- 측면 자세에서는 반대쪽 관절이 보이는 쪽 팔다리에 겹쳐 찍힐 수 있으므로, 결과 이미지를 직접 확인해야 한다.
- 자세한 내용: `reports/day03/bend_pose_left_right_check.md`

### 3팀 동작 목록 (모션캡처 촬영 영상 36개)

| # | 동작 | 그룹 | # | 동작 | 그룹 |
|---|---|---|---|---|---|
| 1 | 키오스크 내려치기 | 타격·파손 | 10 | 뒤로 눕고 옆으로 눕기 | 눕기·쓰러짐 |
| 2 | 한팔로 물건 던지기 | 던지기 | 11 | 몸통박치기 | 밀기·흔들기 |
| 3 | 냉동고 앞차기 | 차기 | 12 | 무릎꿇고 앉아 인형빼기 | 앉기 |
| 4 | 뽑기기계 흔들기 | 밀기·흔들기 | 13 | 테이블 내려치기 | 타격·파손 |
| 5 | 양손 밀치기 | 밀기·흔들기 | 14 | 구토하기 | 숙이기 |
| 6 | 양손으로 휘둘러 타격하기 | 타격·파손 | 15 | 벽에 기대어 잠자기 | 눕기·쓰러짐 |
| 7 | 매트 위에 앉기 | 앉기 | 16 | 뒤로 기절하기 | 눕기·쓰러짐 |
| 8 | 의자 위에 점프해서 앉아 폰보기 | 앉기 | 17 | 장애물 걸려 앞으로 넘어지기 | 눕기·쓰러짐 |
| 9 | 앞으로 누워서 폰보기 | 눕기·쓰러짐 | | | |

모든 동작을 정면·측면으로 찍었다. 영상 파일과 동작 연결: `configs/team3_actions.csv`

### 24번 - 팀 동작 규칙 (3팀, `17_team_pose_rules.py`)

교재는 "양손목이 어깨보다 위인가?"(Arms Up) 하나를 확인하지만,
우리 팀은 17개 동작을 설명할 수 있도록 Feature를 늘리고 같은 방식으로 규칙을 만들었다. (`src/team_features.py`)

| Feature | 계산 | 설명하는 동작 |
|---|---|---|
| 상체 기울기 `torso_tilt_deg` | 골반 중심→어깨 중심이 세로에서 기운 각도 | 눕기·쓰러짐, 숙이기 |
| 골반-발목 높이 `hip_ankle_dy` | (발목 y − 골반 y) ÷ 상체 길이 | 눕기 / 숙이기 / 앉기 구분 |
| BBox 가로/세로 `bbox_aspect_ratio` | 사람 박스 너비 ÷ 높이 | 눕기 (보조) |
| 팔 올림 `wrist_above_shoulder` | (어깨 y − 손목 y) ÷ 상체 길이 | 내려치기·던지기 준비 |
| 두 발목 높이 차 `ankle_height_diff` | \|왼발목 y − 오른발목 y\| ÷ 상체 길이 | 차기 |
| 손목·발목·골반 속도 (영상) | 앞 Frame과의 이동 ÷ 상체 길이 ÷ 시간 | 던지기·차기·몸통박치기 |

```text
자세 상태
  상체 >= 60° + |골반-발목 높이| < 0.6               → 눕기
  상체 >= 45° + 골반-발목 높이 >= 0.6                → 숙이기
  상체 < 45° + (|골반-발목 높이| < 0.6 또는 무릎 < 120°) → 앉기
  나머지                                             → 서기
팔 올림 : 손목이 어깨보다 상체 길이 0.2배 이상 위
발 들기 : 두 발목 높이 차가 상체 길이 0.5배 이상
```

```bash
python scripts/17_team_pose_rules.py --source data/day03/challenge/images_actions/01_kiosk_strike.jpg
```

17개 동작 대표 장면(`data/day03/challenge/images_actions/`)에 실행한 결과:

| 결과 | 동작 |
|---|---|
| 눕기 (맞음) | 9 앞으로 누워서 폰보기, 10 뒤로 눕고 옆으로 눕기, 16 뒤로 기절하기 |
| 앉기 (맞음) | 7 매트 위에 앉기, 8 의자 위에 앉아 폰보기 |
| 숙이기 | 14 구토하기 (맞음), 12 인형빼기·15 벽 잠자기·17 넘어진 뒤 (손·무릎을 짚은 자세) |
| 팔 올림 True | 1 키오스크 내려치기 (0.532), 2 던지기 (0.305), 13 테이블 내려치기 (0.293), 4 흔들기 (0.257) |
| 놓침 | 3 앞차기: 정면에서 발이 카메라 쪽으로 나와 두 발목 높이 차가 0.225 (기준 0.5 미달) |
| 서기와 구분 안 됨 | 5 양손 밀치기, 6 휘둘러 타격 (팔 올림 0.147), 11 몸통박치기 |

- 상체 기울기 하나만 보면 구토하기(77.8°)·인형빼기·벽 잠자기·넘어지기도 모두 "눕기"가 된다. 교재에서 Sit 사진이 Arms Up = True로 나온 것과 같은 문제다.
- 골반-발목 높이를 함께 보면 실제로 누운 9·10·16번만 눕기로 남았다.
- **Feature 하나만으로는 비슷한 자세를 구분하기 어렵고, 동작의 의미를 설명하는 Feature를 조합해야 한다.**
- 차기·밀치기처럼 순간적으로 일어나는 동작은 사진 한 장의 자세로는 서 있는 사람과 거의 같다.

### 25~28번 - 영상의 각 Frame을 38-D Feature로 (교재 데모 영상)

```bash
cp data/day02/videos/pose_quality_demo.mp4 data/day03/videos/feature_demo.mp4
python scripts/18_video_frame_features.py --max-frames 150
python scripts/18_video_frame_features.py --max-frames 0
```

- Total frames `60`, Valid pose frames `60`, Feature count per frame `38` (교재와 같음)
- CSV 한 행 = 한 Frame. 시간 순서가 CSV 행 순서로 저장됨
- 3팀용으로 사람 BBox 원본 좌표(`bbox_x1~y2`), 팀 Feature 8개, 규칙 결과(`pose_state`, `lie_down`, `arm_raised`, `leg_raised`)도 함께 저장
- 27번 왼팔꿈치 각도: Frame 1 `140.14°` → Frame 10 `138.44°` → Frame 15 `125.91°` → Frame 20 `115.62°` (점점 굽혀짐)
- 28번 `left_wrist_y - left_shoulder_y`: Frame 1 `10.99` → Frame 20 `4.19` → Frame 28 `0.747` (손목이 어깨 높이로 올라옴)
- 데모 영상은 팔굽혀펴기 장면이라 몸이 수평이어서 60 Frame 모두 `lie_down = 1`로 판정했다.
  → 몸이 수평인 운동 자세와 실제로 누운 자세는 한 장으로는 구분되지 않는다. 골반이 바닥에서 떠 있는지, 시간에 따라 움직이는지를 봐야 한다.

### 팀 추가 - 17개 동작 영상 36개 전체 (`18b_team_action_summary.py`)

```bash
for v in data/day02/videos/20261113_*.mp4; do
  python scripts/18_video_frame_features.py --source "$v" --max-frames 0
done
python scripts/18b_team_action_summary.py
```

전체 표: `reports/day03/challenge/team3_action_summary.md`

| 확인한 점 | 근거 (p95, 측면) |
|---|---|
| 내려치기는 팔 올림이 가장 큼 | 키오스크 내려치기 팔 올림 `0.582` (정면 `0.467`), 팔 올림 Frame 40~44% |
| 던지기는 손목 속도가 가장 빠름 | 던지기 손목 속도 `11.96` |
| 앞차기는 발목 속도로 잡힘 | 발목 속도 `10.48` (정면 `5.72`, 다른 서서 하는 동작 정면 1.3~2.7) |
| 몸통박치기는 골반 속도가 큼 | 골반 속도 `3.03` (다른 서서 하는 동작 1.4~2.6) |
| 흔들기·밀치기·휘두르기는 서로 비슷 | 팔 올림 0.07~0.18, 손목 속도 4.7~6.8 |
| 바닥 자세는 측면에서 맞음 | 측면 앉기 7번 `54.4%` / 8번 `37.2%`, 눕기 9·10번 `62%`, 숙이기 14번 `52.2%` |
| 정면에서는 바닥 자세가 틀림 | 10번 정면 눕기 `0%`, 16번 정면은 눕기 대신 앉기 `36.6%`, 8번 정면 앉기 `0%` |

- 카메라 쪽으로 누우면 몸이 화면에서 세로로 짧게 보여 상체 기울기가 작게 나온다. 바닥 자세는 측면 카메라, 팔 동작은 Confidence가 높은 정면 카메라가 더 유리했다.
- 흔들기·밀치기·문 차기처럼 물체와 닿는 동작은 Pose만으로 부족하고, 키오스크·뽑기기계·냉동고 같은 물체 위치(객체 검출)가 필요하다.

### Team Mini Challenge (29~37번) - 바닥에 누운 자세 vs 서기

- 교재처럼 Target / Contrast 한 쌍으로 진행: Target = **바닥에 누운 자세** (9·10·16번 동작의 대표 상태), Contrast = **서기**, 측면 카메라
- 사진 4장과 영상 1개는 모션캡처 촬영 때 직접 찍은 영상에서 캡처
  (`target_near` 16번 165214 9.00초, `target_far` 10번 161722 10.00초, `contrast_near` 9번 161202 0초, `contrast_far` 17번 165930 0초, `challenge_action.mp4` = 16번 뒤로 기절하기 측면 165214 전체)
- 결과 정리: `reports/day03/challenge/feature_design.md`

| 확인 항목 | 결과 |
|---|---|
| 관절각도 | 무릎은 서기 161~173° / 눕기 159~164°로 둘 다 펴져 있어 구분이 안 됨. 팔꿈치 차이는 팔 모양 차이 |
| 정규화 (눕기 Near / Far) | Pixel `198.18` → 어깨 폭 정규화 `8.199` (오히려 커짐) |
| 이유 | 측면에서는 두 어깨가 겹쳐 보여 어깨 폭이 `124.91px` / `18.99px`로 크게 달라짐 |
| 팀 추가 비교 (`18a_team_torso_normalization.py`) | 상체 길이로 나누면 눕기 Near / Far `0.438`, 서기 Near / Far `0.167`, 눕기 vs 서기 `1.209` |
| 38-D Feature | 4장 모두 Feature 38개. Target Far는 평균 Confidence `0.623`, LOW 7개 |
| 선택한 Feature | 상체 기울기, 골반-발목 높이, BBox 가로/세로 비율 (+ 17개 동작용 팔 올림·두 발목 높이 차·속도 3개) |
| 영상 (808 Frame) | 상체 기울기 서기 약 `7°` → 쓰러져 누움 `77~93°` (3.3~10.6초 유지) → 서기 약 `1°` |

- 영상에서 `lie_down`은 쓰러지는 순간 0.2초, 0.03초짜리로 깜빡였다. 한 Frame 판정만으로 알림을 보내면 안 되고, "일정 시간 이상 누워 있음" 같은 시간 조건이 필요하다.
- 누워 있는 동안 38-D의 `nose_y`(어깨 폭 정규화)는 -2.85 ~ -0.30으로 흔들렸지만, 상체 기울기는 안정적으로 유지되었다.

### 오늘 이해한 핵심

관절각도는 세 관절로 "관절이 얼마나 굽혀졌는지"를 표현하고,
정규화 좌표는 골반 중심과 어깨 폭을 기준으로 "관절이 몸 기준 어디에 있는지"를 표현한다.
두 값은 서로 다른 정보를 담고 있어서 함께 사용해야 자세를 더 잘 설명할 수 있다.

또한 Confidence가 높아도 관절 위치가 틀릴 수 있으므로,
특히 측면 자세에서는 결과 이미지를 확인한 뒤 각도를 사용해야 한다.

동작을 구분할 때는 38개의 숫자를 모두 보는 것이 아니라,
그 동작을 설명하는 Feature를 골라야 한다.
우리 팀 17개 동작 중 눕기·앉기·숙이기처럼 자세가 유지되는 동작은 상체 기울기와 골반-발목 높이로,
내려치기·던지기·차기·몸통박치기처럼 순간적인 동작은 팔 올림과 손목·발목·골반 속도로 설명되었다.
측면 카메라에서는 어깨 폭 정규화가 불안정해서, 상체 길이를 기준으로 하는 것이 더 안정적이었다.

영상에서는 같은 Feature를 Frame마다 만들고, 그 값이 시간에 따라 변하는 것을 확인했다.
속도처럼 앞뒤 Frame을 비교해야만 나오는 Feature가 있다는 것도 확인했다.

### 다음 수업

4일차에는 여러 Frame의 Feature를 시간 순서로 묶어
하나의 동작을 표현하는 **Sequence**로 확장합니다.
3팀은 여기에 상태가 유지된 시간, 골반과 바닥의 상대 높이, Confidence가 높은 Frame만 쓴 속도,
키오스크·냉동고·테이블 등 물체 위치와 손목·발목 거리 Feature를 추가할 예정입니다.

## Day 04 - Action Clip & Sequence

> 진행 범위: 교재 1~51번 전체. 본 실습은 교재 제공 영상 6개, Mini Challenge는 우리 팀(3팀) 동작 2개를 모션캡처 촬영 때 직접 찍은 측면 영상으로 진행했습니다.

### 목표

3일차의 38-D Frame Feature를
영상의 시간 순서대로 연결하여
Action Sequence를 만든다.

### 사용한 Action

- bend_return
- leg_raise_lower
- walk_turn_walk
- drink_return
- sit_stand
- wave

### Pipeline

```text
Action Clip
→ Frame
→ YOLO Pose
→ COCO 17 Keypoint
→ Normalization
→ Joint Angle
→ 38-D Frame Feature
→ 시간 순서 유지
→ Sequence
→ Action Label
```

### 주요 코드

```text
configs/
├── action_classes.csv               (6개 Action ID·Label·MP4 연결)
└── day04_challenge_actions.csv      (Mini Challenge: faint_back, trip_fall)

scripts/
├── 19_check_day04_data.py
├── 20_check_action_classes.py
├── 21_extract_clip_keyframes.py     (휴대폰 영상의 마지막 Frame 오류 처리 추가)
├── 22_build_action_sequences.py
├── 23_summarize_action_sequences.py
└── 24_inspect_sequence_stages.py
```

> 3일차 팀 추가 코드 `19_team_torso_normalization.py`, `20_team_action_summary.py`는 4일차 교재 코드 번호(19~24)와 겹치지 않도록 `18a_`, `18b_`로 이름을 바꿨습니다.

### 주요 결과

- 6개 영상 모두 640x480, 25fps, 정상 열림 (`video_metadata.csv`)
- `action_classes.csv`의 6개 Action과 MP4 연결 확인: `Action class check: OK`
- 대표 Frame 18장 (Start / Middle / End)
- 6개의 Action Sequence CSV 생성
- 총 824 Frame 처리
- Pose 성공 824 / 824
- Feature 생성 성공 824 / 824
- Action별 Feature의 시간 변화 확인

| Action | 평균 Confidence | 인접 Row 변화량 | 잘 변한 Feature |
|---|---|---|---|
| wave | 0.9724 | 0.178 | 오른쪽 Wrist y `-0.03 → -3.05 → -0.04`, 오른쪽 Elbow `177° → 116° → 173°` |
| drink_return | 0.9538 | 0.113 | 오른쪽 Wrist y `-0.05 → -2.10 → -0.02`, 오른쪽 Elbow `170° → 61° → 172°` |
| bend_return | 0.9583 | 0.159 | 양쪽 Wrist y `0 → 0.95 → 0`, Hip Center y `249 → 258 → 250` |
| sit_stand | 0.9743 | 0.101 | Knee `160° / 149° → 178° / 173°`, Hip Center y `286 → 250` |
| walk_turn_walk | 0.8883 | 17.478 | Hip Center x `472 → 279 → 462` |
| leg_raise_lower | 0.9678 | 0.217 | 5개 Sample로는 안 보임 → 9개에서 오른쪽 Ankle y `2.4 → 1.81` (1.84초) |

- `sit_stand` 영상은 설명("앉았다가 다시 일어선다")과 달리 앉은 자세로 시작해 일어서는 동작만 있었다. 대표 Frame과 Sequence 값 모두 이렇게 나와 그대로 기록했다.
- `walk_turn_walk`의 큰 변화량(17.478)은 성능 점수가 아니라 방향 전환·정규화 변화·Keypoint 흔들림이 섞인 값이다.
- 상세 결과: `reports/day04/day04_summary.md`

### Mini Challenge - 3팀 동작: 뒤로 기절하기 vs 앞으로 넘어지기

| 항목 | faint_back (16번) | trip_fall (17번) |
|---|---|---|
| 영상 | 측면 1920x1080 60fps, 13.5초 | 측면 1920x1080 60fps, 6.3초 |
| rows / Pose 성공 / Feature 성공 | 808 / 808 / 808 | 374 / 374 / 374 |
| 평균 Confidence / LOW | 0.848 / 1.78 | 0.786 / 3.10 |
| 잘 변한 Feature | Hip Center y: 1초 안에 약 190px 하강 후 8초 이상 유지 | Hip Center x: 1초에 약 380px 앞으로 이동, 무릎 70~100° 유지 |

- 두 영상 모두 시작과 끝은 서 있는 자세라 처음·마지막 Frame만 보면 차이가 없고, 중간의 시간 변화에서만 구분되었다.
- 기절은 골반이 바닥까지 내려가 머물렀고, 넘어지기는 골반이 앞으로 빠르게 이동한 뒤 높이는 약 40px만 내려가 손·무릎으로 버텼다.
- 측면 영상은 어깨 폭 정규화가 불안정해 인접 Row 변화량이 4.2 / 3.2로 크게 나왔다. 그래서 Pixel 좌표(Hip Center)와 관절각도를 중심으로 해석했다.
- 휴대폰 영상은 기록된 Frame 수(810)가 실제(808)보다 커서 21번 코드가 End Frame에서 멈췄다. 실제 마지막 Frame을 쓰도록 고쳤다.
- 결과 정리: `reports/day04/challenge/action_sequence_result.md`

### 오늘 확인한 핵심

- Pose는 한 순간의 자세이다.
- Sequence는 Frame Feature의 시간 흐름이다.
- Action은 한 Frame이 아니라 시간에 따른 자세 변화로 이해해야 한다.
- Action마다 중요하게 변하는 Feature가 다를 수 있다.
- 대표 Sample만으로 동작이 잘 보이지 않으면 더 많은 시점을 확인해야 한다.
- 사람이 화면에서 이동하거나 쓰러지는 동작은 정규화 좌표만이 아니라 Hip Center 같은 원본 위치도 함께 봐야 한다.

### 다음 수업

Day 05에서는 여러 Subject와 반복 Take를 이용하여
Sequence 데이터를 학습용 Dataset 구조로 확장하고
품질검수와 데이터 분리를 진행한다.
