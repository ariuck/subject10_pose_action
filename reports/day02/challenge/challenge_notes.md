# Day02 Team Mini Challenge

## 1. 우리 팀 실험

- 선택한 조건: 방향
- 조건 A: 정면 촬영
- 조건 B: 측면 촬영
- 바꾼 조건 한 가지: 카메라 방향 (같은 사람, 같은 장소, 차렷 자세는 그대로)

사용한 사진 (모션캡처 촬영 때 우리 팀(3팀) 17개 동작을 정면·측면으로 찍은 영상에서 장면 캡처)

| 파일 | 원본 영상 | 3팀 동작 | 시간 | 조건 |
|---|---|---|---|---|
| `challenge_A_01.jpg` | `20261113_152953.mp4` | 4 뽑기기계 흔들기 | 0.10초 | 정면 |
| `challenge_A_02.jpg` | `20261113_162728.mp4` | 13 테이블 내려치기 | 0.10초 | 정면 |
| `challenge_B_01.jpg` | `20261113_153015.mp4` | 4 뽑기기계 흔들기 | 0.80초 | 측면 |
| `challenge_B_02.jpg` | `20261113_162736.mp4` | 13 테이블 내려치기 | 0.10초 | 측면 |

A_01과 B_01, A_02와 B_02는 같은 동작을 정면 카메라와 측면 카메라로 찍은 영상이고, 동작을 시작하기 전 서 있는 장면을 사용함.

## 2. 결과

### 대표 사진 상세 확인

```bash
python scripts/07_pose_image_detail.py --source data/day02/images/challenge_A_01.jpg
python scripts/07_pose_image_detail.py --source data/day02/images/challenge_B_01.jpg
```

| 확인 항목 | 조건 A (A_01) | 조건 B (B_01) |
|---|---|---|
| Person BBox Confidence | 0.927 | 0.861 |
| 가장 낮은 Keypoint | right_ear | left_ear |
| 가장 낮은 Keypoint Confidence | 0.816 | 0.027 |
| LOW 여부 (`< 0.50`) | X | O |

### 4장 비교

```bash
python scripts/09_compare_pose_images.py
```

| 이미지 | 조건 | BBox Area Ratio | 평균 KP Conf | LOW 관절 수 |
|---|---|---|---|---|
| `challenge_A_01.jpg` | A | 0.2951 | 0.9789 | 0 |
| `challenge_A_02.jpg` | A | 0.2050 | 0.9763 | 0 |
| `challenge_B_01.jpg` | B | 0.1747 | 0.7941 | 4 |
| `challenge_B_02.jpg` | B | 0.1438 | 0.7846 | 4 |

- 조건 A 평균 Keypoint Confidence: 0.9789 / 0.9763
- 조건 B 평균 Keypoint Confidence: 0.7941 / 0.7846

- 조건 A LOW 관절 수: 0 / 0
- 조건 B LOW 관절 수: 4 / 4

- 가장 불안정했던 이미지: `challenge_B_02.jpg`
- 가장 낮았던 Keypoint: left_ear
- 해당 Confidence: 0.029

B 두 장 모두 LOW 관절이 같았음: left_eye, left_ear (0.50 미만 LOW), left_elbow, left_wrist (0.30~0.50 WARNING)
→ 전부 카메라 반대쪽(왼쪽) 관절

BBox Area Ratio가 B에서 작아진 것은 거리가 바뀐 것이 아니라, 옆으로 서면 몸의 폭이 좁게 보여서 BBox 가로 길이가 줄어든 것.

### Threshold 확인

`challenge_B_01.jpg`의 left_elbow(0.377)와 left_wrist(0.416)가 0.30과 0.50 사이라 두 기준으로 다시 그려 봄.

```bash
python scripts/08_draw_skeleton_quality.py --source data/day02/images/challenge_B_01.jpg --kp-conf 0.30
python scripts/08_draw_skeleton_quality.py --source data/day02/images/challenge_B_01.jpg --kp-conf 0.50
```

- Threshold 0.30 → 왼쪽 팔꿈치·손목이 사용되어, 몸 뒤에 가려진 팔이 앞쪽 팔과 겹친 위치에 그려짐
- Threshold 0.50 → 왼쪽 팔꿈치·손목이 제외되어 보이는 쪽 팔만 남음

## 3. 우리 팀의 해석

- 어떤 조건에서 Pose 품질이 더 불안정했는가?
  → 측면(B). 평균 Keypoint Confidence가 약 0.98에서 0.79로 떨어지고 LOW 관절이 0개에서 4개로 늘어남. 사람 검출 Confidence도 0.92~0.93에서 0.86으로 낮아짐
- 실제 사진을 보았을 때 그 이유는 무엇이라고 생각하는가?
  → 옆으로 서면 카메라 반대쪽 눈·귀·팔이 머리와 몸통에 가려져 사진에 거의 나오지 않음. 모델은 보이지 않는 관절도 위치를 추정해서 내놓지만, 그 위치를 확신하지 못해 Confidence가 낮게 나옴
- Confidence가 낮은 관절을 이후 분석에 사용할 때 무엇을 주의해야 하는가?
  → 측면 사진의 왼쪽 손목(0.416)처럼 Confidence가 낮은 관절은 실제로 보이지 않는 위치를 추정한 값이라 좌표를 믿기 어려움. 3일차 관절각도 계산에 그대로 넣으면 팔꿈치 각도가 틀리게 나올 수 있으므로 Threshold로 먼저 걸러야 함

## 참고 — 본수업 6장 결과

| 이미지 | BBox Area Ratio | 평균 KP Conf | LOW 관절 수 | 가장 낮은 관절 |
|---|---|---|---|---|
| front.jpg | 0.300 | 0.9754 | 0 | right_ear 0.743 |
| arms_up.jpg | 0.675 | 0.9599 | 0 | right_ear 0.654 |
| distance_far.jpg | 0.149 | 0.9710 | 0 | right_ear 0.738 |
| occlusion.jpg | 0.265 | 0.9533 | 1 | right_ear 0.480 |
| partial_body.jpg | 0.339 | 0.9073 | 2 | left_ankle 0.491 |
| side.jpg | 0.199 | 0.7541 | 4 | right_ear 0.028 |

본수업의 `side.jpg`도 평균 0.754, LOW 4개로 이번 측면 사진과 거의 같은 결과였음. 측면 촬영에서 반대쪽 관절이 불안정해지는 것이 우연이 아니라 반복해서 나타나는 경향이라는 것을 확인함.
