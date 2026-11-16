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

> 진행 범위: 교재 1~23번 (관절각도, 정규화, Near / Far 비교, Person A / B 비교, 좌우 반전, 38-D Frame Feature)
> 14~15번은 모션캡처 촬영 때 직접 찍은 영상에서 캡처한 장면을, 16번부터는 교재 제공 이미지를 사용했습니다.

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

### Day 03 핵심 흐름 (1~23번)

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
└── 16_make_frame_feature.py
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

### 오늘 이해한 핵심

관절각도는 세 관절로 "관절이 얼마나 굽혀졌는지"를 표현하고,
정규화 좌표는 골반 중심과 어깨 폭을 기준으로 "관절이 몸 기준 어디에 있는지"를 표현한다.
두 값은 서로 다른 정보를 담고 있어서 함께 사용해야 자세를 더 잘 설명할 수 있다.

또한 Confidence가 높아도 관절 위치가 틀릴 수 있으므로,
특히 측면 자세에서는 결과 이미지를 확인한 뒤 각도를 사용해야 한다.

### 다음 진행

24번(`17_arms_up_rule.py`, Arms Up 규칙 판정)부터 이어서 진행합니다.
