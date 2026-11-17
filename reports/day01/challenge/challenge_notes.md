# Day01 Team Mini Challenge

## 사용한 데이터

- COCO8-Pose 중 첫 실습에서 사용하지 않은 이미지
- 직접 촬영 데이터: 모션캡처 촬영 때 우리 팀(3팀 무인 상점 이상행동) 17개 동작을 정면·측면으로 찍은 영상에서 장면을 캡처하여 사용

| 파일 | 원본 영상 | 3팀 동작 (카메라) | 시간 | 캡처한 장면 |
|---|---|---|---|---|
| `data/day01/challenge/side.jpg` | `20261113_153015.mp4` | 4 뽑기기계 흔들기 (측면) | 0.10초 | 동작 시작 전 옆으로 서 있음 |
| `data/day01/challenge/crouch.jpg` | `20261113_154206.mp4` | 7 매트 위에 앉기 (정면) | 3.80초 | 매트에 앉으려고 쪼그려 앉는 중 |
| `data/day01/challenge/prone.jpg` | `20261113_161202.mp4` | 9 앞으로 누워서 폰보기 (측면) | 4.85초 | 바닥에 엎드려 누움 |
| `data/day01/challenge/curl_wall.jpg` | `20261113_161645.mp4` | 10 뒤로 눕고 옆으로 눕기 (정면) | 15.50초 | 벽 쪽 바닥에 몸을 말고 옆으로 누움 |

## 요구사항 1 — 사용하지 않은 COCO8-Pose 이미지 분석

| 입력 | 조건 | 사람 검출 | BBox | Keypoint | Skeleton | 판단 |
|---|---|---|---|---|---|---|
| 000000000049 (`coco_result01.jpg`) | 기존 데이터 | 1명 | 정상 | 정상 | 자연스러움 | 잘 됨 |
| 000000000110 (`coco_result02.jpg`) | 사람 많은 식당 장면 | 8명 | 겹친 BBox 다수 | 가려진 사람의 관절이 엉뚱한 곳에 찍힘 | 선이 서로 교차 | 불안정 |
| 000000000077 / 113 | 여러 사람 | 각 3명 | 정상 | 일부 겹침 | 대체로 자연스러움 | 보통 |

## 요구사항 2 — 새로운 촬영 조건과 실행 전 예상

| 촬영 조건 | 실행 전 예상 |
|---|---|
| 옆으로 서기 (한쪽 팔을 뒤로 뻗음) | 반대쪽 팔과 눈·귀가 몸에 가려져서 그쪽 관절이 불안정할 것 같다 |
| 쪼그려 앉기 | 몸이 접히면서 무릎과 팔이 겹쳐 Skeleton이 엉킬 것 같다 |
| 바닥에 엎드리기 | 얼굴이 바닥을 향해 얼굴 관절은 못 잡고, 팔 위치도 불안정할 것 같다 |
| 벽에 붙어 웅크리기 | 몸이 작게 말려 있어서 관절 대부분이 가려질 것 같다 |

## 요구사항 3 — Pose 실행 결과

```bash
python scripts/03_pose_image.py --source data/day01/challenge/side.jpg
python scripts/03_pose_image.py --source data/day01/challenge/crouch.jpg
python scripts/03_pose_image.py --source data/day01/challenge/prone.jpg
python scripts/03_pose_image.py --source data/day01/challenge/curl_wall.jpg
```

| 촬영 조건 | 실제 결과 |
|---|---|
| 옆으로 서기 | 사람 1명 검출(0.85). 다리와 몸통은 정상. 반대쪽 팔꿈치 0.262, 손목 0.182, 왼쪽 귀 0.021로 예상대로 낮음. 예상하지 못한 점은 뒤로 뻗은 팔의 손목이 어깨 바로 옆에 잡혀서 팔이 실제보다 훨씬 짧게 그려진 것 |
| 쪼그려 앉기 | 사람 1명 검출(0.92). 예상과 달리 몸이 접혀도 Skeleton이 실제 자세와 거의 맞음. 아래를 향한 쪽 오른쪽 눈 0.095, 오른쪽 귀 0.013만 낮음 |
| 바닥에 엎드리기 | 사람 1명 검출(0.70). 다리·골반은 0.71~0.80으로 잡혔지만 어깨 0.26~0.31, 팔꿈치·손목 0.07~0.19, 얼굴 0.01 수준이라 상체 Skeleton이 거의 그려지지 않음 |
| 벽에 붙어 웅크리기 | 사람을 검출하지 못함 (0명) |

## 조건별 비교표

| 입력 | 조건 | 사람 검출 | BBox | Keypoint | Skeleton | 판단 |
|---|---|---|---|---|---|---|
| COCO 000000000049 | 기존 데이터 | O | 정상 | 정상 | 자연스러움 | 잘 됨 |
| COCO 000000000110 | 여러 사람 겹침 | O (8명) | 겹침 | 일부 엉뚱한 위치 | 교차 | 불안정 |
| `side_result.jpg` | 옆으로 서기 | O | 정상 | 반대쪽 팔·얼굴 낮음, 뒤로 뻗은 팔 손목 위치 틀림 | 팔이 짧게 그려짐 | 부분적으로 불안정 |
| `crouch_result.jpg` | 쪼그려 앉기 | O | 정상 | 대부분 정상 | 자연스러움 | 잘 됨 |
| `prone_result.jpg` | 엎드리기 | O | 정상 | 하체만 정상 | 상체 없음 | 불안정 |
| `curl_wall_result.jpg` | 벽에 붙어 웅크리기 | X | 없음 | 없음 | 없음 | 실패 |

## 가장 잘 나온 결과

- 파일: `crouch_result.jpg`
- 관찰: 몸을 크게 접은 자세인데도 어깨-팔꿈치-손목, 골반-무릎-발목이 실제 위치에 맞게 연결됨. 아래를 향한 얼굴 한쪽만 Confidence가 낮음

## 가장 불안정한 결과

- 파일: `curl_wall_result.jpg`
- 관찰: 사람이 화면에 분명히 있는데 BBox조차 나오지 않음
- 예상 원인: 벽에 붙어 몸을 공처럼 말고 있어서 얼굴·팔·다리가 거의 보이지 않고, 카메라에서 멀리 떨어져 위에서 내려다보는 각도라 사람의 형태로 인식하기 어려웠던 것 같음

두 번째로 불안정한 결과는 `prone_result.jpg`. 얼굴이 바닥을 향하고 팔이 몸 앞쪽으로 붙어 있어서 상체 관절이 카메라에서 거의 보이지 않았음.

## MP4 입력 확인

```bash
python scripts/04_pose_video.py \
  --source data/day01/video/solution_ci_pose_demo.mp4 \
  --max-frames 150
```

- JPG는 Frame 1개를 처리, MP4는 60개 Frame을 하나씩 반복 처리
- IP Webcam은 카메라에서 Frame이 계속 들어오는 것
- 세 입력 모두 Frame을 얻은 뒤에는 같은 YOLO Pose 모델을 사용

## 오늘 확인한 Pipeline

JPG / MP4 / IP Webcam
→ Frame
→ YOLO Pose
→ Person Bounding Box
→ COCO 17 Keypoint
→ Skeleton
