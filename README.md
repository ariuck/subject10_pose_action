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
