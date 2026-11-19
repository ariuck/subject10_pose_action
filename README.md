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

> 진행 범위: 교재 1~40번 전체 (관절각도, 정규화, Near / Far 비교, Person A / B 비교, 좌우 반전, 38-D Frame Feature, Arms Up 규칙, 영상 Frame Feature, Team Mini Challenge)
> 14~15번은 모션캡처 촬영 때 직접 찍은 영상에서 캡처한 장면을, 나머지 본 실습은 교재 제공 데이터를 사용했습니다.
> Mini Challenge는 우리 팀(3팀 — 무인 상점 이상행동 인식) 17개 동작을 정면·측면으로 찍은 영상으로 진행했습니다.

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
- 동작을 설명하는 Feature 선택
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
관절각도
        ↓
Hip Center
        ↓
Shoulder Width Normalization
        ↓
Normalized Keypoint 34개
+
Joint Angle 4개
        ↓
38-D Frame Feature
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
└── feature_utils.py

scripts/
├── 11_check_day03_data.py
├── 12_joint_angle.py
├── 13_normalize_pose.py
├── 14_compare_normalization.py
├── 15_flip_keypoints.py
├── 16_make_frame_feature.py
├── 17_arms_up_rule.py
└── 18_video_frame_features.py
```

- `18_video_frame_features.py`: 교재 본문에 실린 18번 코드가 17번 코드와 같은 내용으로 잘못 들어가 있어서, 26번의 설명·실행 예시·출력 결과를 기준으로 작성했다. 실행 결과(60 Frame, 38개 Feature)가 교재와 같았다.

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

### 24번 - Arms Up 상대 위치 Feature 규칙 (`17_arms_up_rule.py`)

```bash
python scripts/17_arms_up_rule.py --source data/day03/images/arms_up_person_a.jpg
python scripts/17_arms_up_rule.py --source data/day03/images/sit_pose.jpg
```

| 이미지 | Min required confidence | Left wrist - shoulder | Right wrist - shoulder | Arms Up |
|---|---|---|---|---|
| `arms_up_person_a.jpg` | 0.971 | -1.526 | -1.541 | True |
| `sit_pose.jpg` | 0.956 | -0.883 | -1.081 | True |

- 정규화 좌표에서는 위쪽이 음수이므로 `wrist_y - shoulder_y < 0`이면 손목이 어깨보다 위다.
- `arms_up_person_a`는 사진과 맞게 True가 나왔다.
- `sit_pose`도 양손이 어깨보다 위라서 True가 나왔다. 규칙대로는 맞지만, 서서 팔 올리기와 앉아서 팔 올리기를 구분하지 못한다.
- **Feature 하나만으로는 서로 다른 자세를 완벽하게 구분하기 어렵다.** 무릎 각도 같은 다른 Feature를 함께 봐야 한다.

### 25~28번 - 영상의 각 Frame을 38-D Feature로

```bash
cp data/day02/videos/pose_quality_demo.mp4 data/day03/videos/feature_demo.mp4
python scripts/18_video_frame_features.py --max-frames 150
python scripts/18_video_frame_features.py --max-frames 0
```

- Total frames `60`, Valid pose frames `60`, Feature count per frame `38` (교재와 같음)
- CSV 한 행 = 한 Frame. 시간 순서가 CSV 행 순서로 저장됨
- 27번 왼팔꿈치 각도: Frame 1 `140.14°` → Frame 10 `138.44°` → Frame 15 `125.91°` → Frame 20 `115.62°` (점점 굽혀짐)
- 28번 `left_wrist_y - left_shoulder_y`: Frame 1 `10.99` → Frame 20 `4.19` → Frame 28 `0.747` → Frame 30 `1.549`
  - 손목이 어깨보다 한참 아래에서 어깨 높이 근처까지 올라오는 흐름이 숫자로 보였다.

### Team Mini Challenge (29~37번) - 3팀 17개 동작 (정면·측면)

Mini Challenge는 그날 배운 내용을 직접 해 보는 복습이라, 우리 팀(3팀 무인 상점 이상행동) 17개 동작 영상으로 진행했다.
모션캡처 촬영 때 17개 동작을 정면·측면으로 찍은 영상 36개(`configs/team3_actions.csv`)를 사용했다.
교재 Mini Challenge 형식(Target / Contrast 한 쌍)은 **바닥에 누운 자세 vs 서기**(측면)로 맞췄다.

```text
src/team_features.py                       팀 Feature: 상체 기울기, 골반-발목 높이, BBox 비율, 팔 올림,
                                           두 발목 높이 차, 손목·발목·골반 속도, 자세 상태 규칙
scripts/team3/team_pose_rules.py           사진 한 장에서 서기/앉기/숙이기/눕기, 팔 올림, 발 들기 판정
scripts/team3/team_video_features.py       38-D + 팀 Feature + BBox 원본 좌표를 Frame별로 저장
scripts/team3/team_torso_normalization.py  어깨 폭 정규화 vs 상체 길이 정규화 비교
scripts/team3/team_action_summary.py       36개 영상의 팀 Feature를 동작별로 요약
```

| 확인 항목 | 결과 |
|---|---|
| 관절각도 | 무릎은 서기 161~173° / 눕기 159~164°로 둘 다 펴져 있어 구분이 안 됨 |
| 정규화 (눕기 Near / Far) | Pixel `198.18` → 어깨 폭 정규화 `8.199` (측면에서 두 어깨가 겹쳐 어깨 폭 124.91 / 18.99px) |
| 상체 길이 정규화 | 눕기 Near / Far `0.438`, 서기 Near / Far `0.167`, 눕기 vs 서기 `1.209` |
| 38-D Feature | 4장 모두 Feature 38개 |
| 선택한 Feature | 상체 기울기, 골반-발목 높이, BBox 가로/세로 비율 (+ 팔 올림, 두 발목 높이 차, 속도 3개) |
| 영상 (16 뒤로 기절하기, 808 Frame) | 상체 기울기 서기 약 `7°` → 쓰러져 누움 `77~93°` (3.3~10.6초 유지) → 서기 약 `1°` |

17개 동작 전체 (36개 영상):

- 눕기·앉기·숙이기처럼 자세가 유지되는 동작은 상체 기울기 + 골반-발목 높이로 구분되었다. 정면보다 측면에서 잘 맞았다.
- 상체 기울기 하나만 보면 구토·인형빼기·넘어지기도 "눕기"가 되어, 골반-발목 높이를 함께 봐야 했다. 교재 24번에서 Sit 사진이 Arms Up = True로 나온 것과 같은 문제다.
- 내려치기는 팔 올림, 던지기는 손목 속도, 앞차기는 발목 속도, 몸통박치기는 골반 속도로 가장 잘 보였다.
- 흔들기·밀치기처럼 물체와 닿는 동작은 Pose만으로 부족해 객체 검출이 필요하다.
- 결과 정리: `reports/day03/challenge/feature_design.md`, `reports/day03/challenge/team3_action_summary.md`

### 오늘 이해한 핵심

관절각도는 세 관절로 "관절이 얼마나 굽혀졌는지"를 표현하고,
정규화 좌표는 골반 중심과 어깨 폭을 기준으로 "관절이 몸 기준 어디에 있는지"를 표현한다.
두 값은 서로 다른 정보를 담고 있어서 함께 사용해야 자세를 더 잘 설명할 수 있다.

또한 Confidence가 높아도 관절 위치가 틀릴 수 있으므로,
특히 측면 자세에서는 결과 이미지를 확인한 뒤 각도를 사용해야 한다.

동작을 구분할 때는 38개의 숫자를 모두 보는 것이 아니라,
손목과 어깨의 상대 높이처럼 해당 동작을 설명하는 Feature를 선택하는 것이 중요하다.
영상에서는 같은 38-D Feature를 Frame마다 만들고, Feature가 시간에 따라 변하는 모습을 확인했다.

### 다음 수업

4일차에는 여러 Frame의 Feature를 시간 순서로 묶어
하나의 동작을 표현하는 **Sequence**로 확장합니다.

## Day 04 - Action Clip & Sequence

> 진행 범위: 교재 1~51번 전체. 본 실습은 교재 제공 영상 6개, Mini Challenge는 우리 팀(3팀) 17개 동작을 정면·측면으로 찍은 영상 36개로 진행했습니다.

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
├── action_classes.csv
└── day04_challenge_actions.csv

scripts/
├── 19_check_day04_data.py
├── 20_check_action_classes.py
├── 21_extract_clip_keyframes.py
├── 22_build_action_sequences.py
├── 23_summarize_action_sequences.py
└── 24_inspect_sequence_stages.py
```

- 교재 코드 6개는 교재 원본 그대로 사용했다.

### 주요 결과

- 6개 영상 모두 640x480, 25fps, 정상 열림 (`video_metadata.csv`)
- `action_classes.csv`의 6개 Action과 MP4 연결 확인: `Action class check: OK`
- 대표 Frame 18장 (Start / Middle / End)
- 6개의 Action Sequence CSV 생성
- 총 824 Frame 처리
- Pose 성공 824 / 824
- Feature 생성 성공 824 / 824
- Action별 Feature의 시간 변화 확인
- 직접 촬영한 Action에도 같은 Sequence Pipeline 적용

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

### Mini Challenge - 3팀 17개 동작 (정면·측면 36개 영상)

Mini Challenge는 우리 팀(3팀) 17개 동작을 정면·측면으로 찍은 영상 36개를 같은 Sequence Pipeline에 넣었다.
(`configs/day04_challenge_actions.csv`)

- 22번 코드는 파일 이름을 `action_label`로 만들기 때문에 Label 뒤에 `_front` / `_side`를 붙였다 (예: `kiosk_strike_front`).
- 휴대폰 영상은 영상 정보의 Frame 수가 실제보다 1~6개 많아 교재 21번 원본이 End Frame에서 멈췄다. Mini Challenge는 이 경우만 처리한 사본(`scripts/team3/team_extract_clip_keyframes.py`)으로 대표 Frame 108장을 만들었다.
- 총 18,950 Frame, Pose 성공 18,742 (98.9%). 32개 영상은 pose_ok_ratio = 1.0이었다.
- 실패 Frame은 모두 바닥에서 몸을 말거나 엎드린 장면이었다 (10번 정면 171, 10번 측면 21, 12번 정면 15, 17번 정면 1).

| 구분 | 평균 Confidence 중앙값 | 평균 LOW 중앙값 | 인접 Row 변화량 중앙값 |
|---|---|---|---|
| 정면 18개 | 0.937 | 0.61 | 0.227 |
| 측면 18개 | 0.829 | 2.41 | 4.475 |

- 서서 하는 동작은 정면에서 변화량이 0.10~0.30으로 교재 영상과 비슷했지만, 측면에서는 4.7~17.0으로 뛰었다. 측면에서 어깨 폭이 작아져 정규화 좌표가 튄 영향이다.

비교한 두 Action: 16 뒤로 기절하기 vs 17 앞으로 넘어지기 (같은 측면 카메라)

| 항목 | faint_back (측면) | trip_fall (측면) |
|---|---|---|
| rows / Pose 성공 / Feature 성공 | 808 / 808 / 808 | 374 / 374 / 374 |
| 평균 Confidence / LOW | 0.848 / 1.78 | 0.786 / 3.10 |
| 잘 변한 Feature | Hip Center y: 1초 안에 약 190px 하강 후 8초 이상 유지 | Hip Center x: 1초에 약 380px 앞으로 이동, 무릎 70~100° 유지 |

- 두 영상 모두 시작과 끝은 서 있는 자세라, 처음·마지막 Frame만 보면 차이가 없고 중간의 시간 변화에서만 구분되었다.
- 기절은 골반이 바닥까지 내려가 머물렀고, 넘어지기는 골반이 앞으로 빠르게 이동한 뒤 손·무릎으로 버텼다.
- 결과 정리: `reports/day04/challenge/action_sequence_result.md`

### 팀 확장 - 3팀 RGB 영상 170개 (4일차 `RGB영상_전처리_Sequence_Dataset` 문서)

같은 Pipeline을 3팀 전체 영상(5명 × 17동작 × 정면·측면 = 170개)에 자동 반복 적용하고,
강사님이 제안한 3팀 추가 Feature를 넣어 **45-D Sequence**를 만들었다.

```text
data/team_projects/team03/rgb/videos/S0x/S0x_A0nn_T0t.mp4   (T01 정면, T02 측면, 실명 대신 S01~S05)
→ Clip Manifest (configs/team03_clip_manifest.csv)
→ Keypoint Dataset (원본 Pixel 좌표 + 관절별 Confidence + BBox)
→ 자동 QA (AUTO_PASS 166 / CHECK 4) → Human QA (PASS 170)
→ 동작 시작·끝 구간 (영상은 자르지 않고 in_action으로 표시)
→ 38-D + 3팀 7개 = 45-D Sequence 170개
```

| 추가 Feature (+7) | 의미 |
|---|---|
| `torso_tilt_deg` | 상체 기울기 (0° 서기, 90° 수평) |
| `bbox_aspect_ratio` | 사람 BBox 가로/세로 비율 |
| `hip_floor_height` | 골반과 바닥(BBox 아래 끝)의 상대 높이 / 상체 길이 |
| `left_ankle_speed`, `right_ankle_speed` | 양발목 속도 (상체 길이/초, 0.1초 간격) |
| `left_wrist_speed`, `right_wrist_speed` | 양손목 속도 — 강사님 표의 "양발과 문 사이 거리" 대신 (영상에 문·물체가 없음) |

- 70,159 Frame 중 Pose 성공 69,926 (99.7%), 45-D 성공 68,813
- 측면에서는 눕기·쓰러짐이 BBox 비율 2.4~3.2, 골반-바닥 높이 0.3~0.5로 서서 하는 동작(약 0.5 / 1.6)과 잘 구분됐다. 정면에서는 카메라 쪽으로 누워 잘 구분되지 않았다.
- 손목 속도는 던지기·내려치기, 발목 속도는 냉동고 앞차기 측면에서 높았다.
- 필요한 자료: 정상행동 비교 영상, 문·물체 위치(ROI), VICON 촬영 목록
- 코드: `scripts/team03/` (교재 26·27번 3팀 사본, 구간 계산, 45-D Sequence, 확인 사진)
- 보고서: `reports/team_projects/team03/team03_data_preparation_report.md`

### 오늘 확인한 핵심

- Pose는 한 순간의 자세이다.
- Sequence는 Frame Feature의 시간 흐름이다.
- Action은 한 Frame이 아니라 시간에 따른 자세 변화로 이해해야 한다.
- Action마다 중요하게 변하는 Feature가 다를 수 있다.
- 대표 Sample만으로 동작이 잘 보이지 않으면 더 많은 시점을 확인해야 한다.

### 다음 수업

Day 05에서는 여러 Subject와 반복 Take를 이용하여
Sequence 데이터를 학습용 Dataset 구조로 확장하고
품질검수와 데이터 분리를 진행한다.

## Day 05 - Keypoint Dataset QA & Subject-wise Split

> 본 실습은 교재 제공 영상 18개, Mini Challenge는 우리 팀(3팀) 영상으로 진행했습니다.

### Dataset

```text
3 Subjects
×
6 Actions
×
1 Take
=
18 MP4
```

```text
A001 bend_return
A002 leg_raise_lower
A003 walk_turn_walk
A004 drink_return
A005 sit_stand
A006 wave
```

### Pipeline

```text
18 MP4
        ↓
COCO 17 Keypoint Dataset
        ↓
자동 QA
        ↓
AUTO_PASS 13
CHECK 5
        ↓
Human QA
        ↓
Human PASS 18
        ↓
Approved 18 Clips
        ↓
Subject-wise Split
        ↓
S01 → Train
S02 → Validation
S03 → Test
        ↓
Leakage check: OK
```

### 주요 결과

- Keypoint CSV: 18개 (Pixel 좌표 + Image Normalized 좌표 + Confidence, Frame당 95개 Column)
- AUTO_PASS: 13개
- CHECK: 5개 (S02 A001~A005, single_person_ratio)
  - 원본 확인 결과 **옷걸이에 걸린 외투**를 두 번째 사람으로 오검출한 것이라 모두 PASS
- 교재 28·29번 코드는 모든 Clip의 Human QA 입력을 요구해서 AUTO_PASS 13개도 직접 확인하고 PASS로 기록했다
- Approved Clips: 18개
- Train: S01 / 6 Clips
- Validation: S02 / 6 Clips
- Test: S03 / 6 Clips
- Subject overlap: 없음
- Clip overlap: 없음

### Random Frame Split 비교

```text
Train Frames: 1946
Test Frames : 487

Subject overlap:
S01, S02, S03

Clip overlap:
18
```

따라서 최종 평가에는 Random Frame Split을 사용하지 않고
Subject-wise Split을 사용한다.

### 주요 실행 파일

```text
configs/
├── day05_subject_split.json
└── day05_challenge_subject_split.json

scripts/
├── 25_check_day05_data.py
├── 26_extract_keypoint_dataset.py
├── 27_keypoint_qa.py
├── 28_validate_human_review.py
├── 29_build_approved_manifest.py
├── 30_build_subject_split.py
├── 31_check_split_leakage.py
└── 32_demo_frame_split_leakage.py
```

- 교재 코드 8개는 교재 원본 그대로 사용했다.

### Mini Challenge - 3팀 A016 뒤로 기절하기 (S01·S02·S03 측면)

같은 Action 하나를 서로 다른 세 사람이 찍은 3팀 영상으로:

```text
Keypoint Dataset
→ 자동 QA (3개 모두 AUTO_PASS)
→ Human QA (3개 PASS)
→ Approved Manifest (3개)
→ Subject-wise Split (S01 Train / S02 Validation / S03 Test)
→ Leakage check: OK
```

를 다시 수행했다.

- 교재 26·27번은 Action 기준(A001~A006)과 중요 관절 표가 코드에 고정되어 있어, 3팀 동작표로 바꾼 사본(`scripts/team03/team_extract_keypoint_dataset.py`, `team_keypoint_qa.py`)으로 실행했다. 28~31번은 교재 원본에 옵션만 바꿔 사용했다.
- 교재 26번은 `--video-root`에 상대 경로를 주면 `relative_to(ROOT)`에서 멈춰서, 3팀 사본에서는 ROOT 기준 경로로 바꿨다.
- 같은 흐름을 3팀 전체 170개에도 적용했다: Approved 170, Train S01·S02·S03 (102) / Validation S04 (34) / Test S05 (34), Leakage check: OK
- 결과 정리: `reports/day05/challenge/mini_challenge_result.md`, `reports/team_projects/team03/team03_data_preparation_report.md`

### Next

Day 06에서는 Day 05의 Keypoint Dataset과 Subject-wise Split을 그대로 사용하여
Body Normalization과 Joint Angle Feature를 만들고
Action Recognition Baseline 학습으로 연결한다.

## Day 06 - Action Recognition Baseline

> 본 실습은 교재 18개 Dataset, Mini Challenge는 우리 팀(3팀) 170개 Dataset으로 진행했습니다.

### Input

Day 05 Approved Keypoint Dataset

- S01 → Train
- S02 → Validation
- S03 → Test
- 6 Actions
- 18 Clips

### Pipeline

```text
Keypoint CSV
→ 38-D Frame Feature
→ Sequence Resampling (30 Step)
→ Sequence Feature (1,140)
→ Random Forest
→ Validation
→ Misclassification Analysis
→ Feature Comparison
→ Mini Challenge
→ Final Test
```

### Action Classes

- bend_return
- leg_raise_lower
- walk_turn_walk
- drink_return
- sit_stand
- wave

### Baseline

- Sequence Steps: 30
- Feature Count: 1140
- n_estimators: 300
- max_depth: null
- random_state: 42

### Baseline Validation

- Accuracy: 0.5000 (교재 0.6667)
- Macro F1: 0.4444 (교재 0.5556)
- Correct: 3 / 6

Misclassified:

- walk_turn_walk → leg_raise_lower (교재와 같음, Confidence 0.2133)
- drink_return → wave (교재와 같음)
- wave → drink_return (교재에서는 정답)

입력 Keypoint·Sequence·Static·좌표만 결과는 교재와 같았다. wave 한 Clip은 두 Action의 투표가 0.22 근처로 거의 같아,
Train 6개에서 라이브러리 버전(scikit-learn 1.9.1) 같은 작은 차이로 결과가 뒤집힌 것으로 보인다.

### Feature Experiments

Static VS Sequence

- Static: Accuracy 0.6667 / Macro F1 0.5556
- Sequence: Accuracy 0.5000 / Macro F1 0.4444

Coordinates + Angles VS Coordinates Only

- Coordinates + Angles: Accuracy 0.5000 / Macro F1 0.4444
- Coordinates Only: Accuracy 1.0000 / Macro F1 1.0000

※ 현재 데이터가 매우 작기 때문에
이 결과를 일반적인 성능으로 해석하지 않습니다.

### Mini Challenge - 3팀 17개 이상행동 Baseline

3팀 170개 Dataset (Train S01~S03 102 / Validation S04 34 / Test S05 34)으로 같은 흐름을 반복하고,
Baseline에서 한 가지 조건만 바꿔 Validation으로 비교했다. (`scripts/team03/team_day06_baseline.py`)

| 조건 | Feature | Validation Accuracy | Validation Macro F1 |
|---|---:|---:|---:|
| Baseline (38-D) | 1,140 | 0.7353 | 0.7373 |
| Static (가운데 Frame) | 38 | 0.5882 | 0.5941 |
| 좌표만 | 1,020 | 0.6765 | 0.6728 |
| **3팀 45-D** | 1,350 | 0.7647 | **0.7725** |
| 동작 구간만 | 1,140 | 0.7941 | 0.7588 |
| 정면만 / 측면만 | 1,140 | 0.6471 / 0.6471 | 0.5882 / 0.5529 |

- 교재 18개와 달리 3팀 데이터에서는 Sequence가 Static보다, 각도 포함이 좌표만보다 높았다.
- 남은 혼동은 물체 없이 팔로 하는 동작(밀치기 ↔ 흔들기, 내려치기·던지기·휘두르기)이었다.
- Feature를 한 단계씩 늘려 보니 38-D 0.7373 → +자세 3개 0.7647 → +발목 속도 0.7922 → +손목 속도 0.7725 (Validation Macro F1). 늘릴수록 항상 좋아지지는 않았다.
- 결과 정리: `reports/day06/challenge/mini_challenge_result.md`
- Feature 확장 과정 기록: `reports/team_projects/team03/feature_expansion_log.md` (3일차 문제 발견 → 4일차 45-D 설계 → 6일차 단계별 효과)

### Final Selected Config

- 본수업: Baseline 그대로 (Sequence Steps 30, Feature 1140, n_estimators 300, max_depth null, random_state 42)
- 3팀: 45-D (Feature 1,350, 나머지 Baseline과 같음) — Test 전에 정한 기준(Validation Macro F1 최고)으로 선택

### Final Test

- 본수업 Subject S03: Accuracy 1.0000 / Macro F1 1.0000 (6 / 6)
- 3팀 Subject S05: Accuracy 0.7647 / Macro F1 0.7529 (26 / 34), 참고 38-D Baseline 0.6765 / 0.6630

Test 결과는 최종 확인에만 사용하고
Test 결과를 이용하여 다시 모델 조건을 수정하지 않았습니다.

### Main Files

- configs/day06_baseline.json
- src/sequence_utils.py
- scripts/33_check_day06_inputs.py
- scripts/34_build_sequence_dataset.py
- scripts/35_train_rf_baseline.py
- scripts/36_evaluate_validation.py
- scripts/37_analyze_misclassified.py
- scripts/38_compare_static_vs_sequence.py
- scripts/39_compare_feature_sets.py
- scripts/40_evaluate_test.py
- scripts/team03/team_day06_baseline.py (3팀 Mini Challenge)
- reports/day06/day06_summary.md
- reports/day06/challenge/mini_challenge_result.md

교재 코드 9개와 설정 파일은 교재 원본 그대로 사용했다.

### Next

Day 07에서는 실시간 Frame을 Sliding Window로 묶어
실시간 Action Recognition으로 확장합니다.

### Important Note

현재 데이터는
3 Subjects × 6 Actions × 1 Take의
작은 교육용 Dataset입니다.

이번 실습의 목적은 높은 성능을 증명하는 것이 아니라
Pose Dataset을 Action Recognition 학습과 평가까지 연결하는
전체 Pipeline을 이해하는 것입니다.
