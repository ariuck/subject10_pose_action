# Day 03 Team Mini Challenge

- 선택한 미션: **미션 A — Arms Up VS Arms Down (정면)**
- 데이터: 모션캡처 촬영 때 같은 팀원을 정면 카메라로 직접 찍은 영상에서 장면을 캡처했다.

| 파일 | 원본 영상 | 시점 | 사람 키(BBox) | 장면 |
|---|---|---|---|---|
| `target_near.jpg` | `20261113_152953.mp4` | 1.53초 | 약 1330px | 가까이에서 양손을 어깨 위로 올림 |
| `target_far.jpg` | `20261113_162728.mp4` | 1.18초 | 약 1105px | 조금 멀리에서 양손(주먹)을 어깨 위로 올림 |
| `contrast_near.jpg` | `20261113_152953.mp4` | 0.10초 | 약 1330px | 가까이에서 팔을 내리고 섬 |
| `contrast_far.jpg` | `20261113_162728.mp4` | 0.10초 | 약 1105px | 조금 멀리에서 팔을 내리고 섬 |
| `challenge_action.mp4` | `20261113_152953.mp4` | 4.5초 | | 팔 내림 → 양팔 올림 → 유지 → 다시 내림 |

- 같은 사람, 같은 정면 카메라, 같은 방향이다.
- Near / Far는 같은 영상 안에서 Target과 Contrast를 골라 짝을 맞췄다.
- 두 영상의 팔 든 모양은 조금 다르다. Near는 양손을 벌려 머리 옆으로, Far는 주먹을 쥐고 머리 위로 들었다.
- 영상은 교재 기준(8~10초)보다 짧은 4.5초지만, Contrast → Target → Contrast 흐름이 모두 들어 있다.

## 1. 선택한 동작

- Target: Arms Up (양손을 어깨 위로 올림)
- Contrast: Arms Down (팔을 내리고 섬)

## 2. 관절각도 비교

`12_joint_angle.py` 결과

| 관절 | Target Near | Contrast Near |
|---|---|---|
| Left Elbow | 130.26° | 162.88° |
| Right Elbow | 99.93° | 155.39° |
| Left Knee | 173.66° | 175.99° |
| Right Knee | 174.75° | 176.50° |

- 가장 차이가 컸던 관절: 오른팔꿈치 (99.93° vs 155.39°, 약 55° 차이)
- Target: 팔을 올리면서 팔꿈치를 굽혔다 (130° / 100°)
- Contrast: 팔을 내리고 거의 펴고 있었다 (163° / 155°)
- 이 결과를 어떻게 해석했는가?:
  - 무릎은 두 자세 모두 174~177°로 거의 같았다. 두 자세 모두 서 있으므로 하체는 변하지 않았다.
  - 이번에는 팔을 굽혀서 들었기 때문에 팔꿈치 각도가 달라졌다. 하지만 팔을 쭉 펴고 들었다면 Arms Up / Arms Down 모두 팔꿈치가 펴져 있어 각도가 비슷했을 것이다.
  - 실제로 Far의 Arms Up은 팔꿈치가 52° / 69°로, 같은 Arms Up인데도 Near(130° / 100°)와 많이 달랐다.
  - 관절각도는 "얼마나 굽혔는지"만 설명하고, **팔이 어깨보다 위에 있는지(위치)는 설명하지 못한다.**
- 필요한 Keypoint의 Confidence는 모두 0.96 이상으로 충분했다.

## 3. 정규화 비교

`14_compare_normalization.py --a target_near.jpg --b target_far.jpg`

- Near Shoulder Width: 235.24 px
- Far Shoulder Width: 208.94 px
- Pixel Mean Distance: 108.2566
- Normalized Mean Distance: 0.2103
- 정규화 후 확인한 점:
  - Near와 Far에서 어깨 폭이 달라져(약 0.89배), Pixel 좌표에서는 같은 관절이 평균 108px 떨어져 있었다.
  - Hip Center를 빼고 Shoulder Width로 나누자 평균 차이가 0.210으로 크게 줄었다.
  - 0으로 줄지 않은 이유는 두 사진의 팔 든 모양이 달라서(손을 벌림 vs 주먹을 머리 위로), 손목·팔꿈치 위치가 실제로 다르기 때문이다.
  - 이번 Near / Far는 거리 차이가 크지 않았다(사람 키 약 1330px vs 1105px). 그래도 Pixel 차이 108 → 정규화 0.21로 위치·크기 영향이 줄어드는 것을 확인했다.

## 4. 우리가 선택한 Feature

1. `left_wrist_y - left_shoulder_y`
2. `right_wrist_y - right_shoulder_y`
3. `right_elbow_angle`

선택한 이유:

- Arms Up은 "손목이 어깨보다 위에 있는가?"가 핵심이다. 정규화 좌표에서는 위쪽이 음수이므로, 손목 y − 어깨 y < 0이면 손목이 어깨보다 위다.
- 팔꿈치 각도는 팔을 굽혀서 들었는지, 펴서 들었는지를 보조로 알려 준다.

## 34.1 선택한 Feature 비교

`16_make_frame_feature.py` 결과로 계산

| 이미지 | 왼손목 − 왼어깨 y | 오른손목 − 오른어깨 y | 오른팔꿈치 각도 | 평균 Confidence | LOW 수 |
|---|---|---|---|---|---|
| Target Near | -0.297 | -0.628 | 99.93° | 0.966 | 0 |
| Target Far | -0.312 | -0.497 | 68.52° | 0.970 | 0 |
| Contrast Near | 1.538 | 1.570 | 155.39° | 0.979 | 0 |
| Contrast Far | 1.495 | 1.561 | 168.11° | 0.976 | 0 |

- Target과 Contrast에서 실제로 차이가 큰 Feature는 무엇인가?:
  - 손목 − 어깨 y. Target은 -0.3 ~ -0.6(어깨보다 위), Contrast는 +1.5 ~ +1.6(어깨보다 한참 아래)으로 부호부터 다르다.
- Near와 Far가 달라도 비슷하게 유지되는 Feature가 있는가?:
  - 손목 − 어깨 y. Contrast는 Near 1.538 / Far 1.495로 거의 같았고, Target도 둘 다 음수였다.
  - 반면 오른팔꿈치 각도는 Target Near 99.9° / Far 68.5°로 같은 Arms Up 안에서도 크게 달랐다.
- Confidence가 낮은 경우 그 Feature를 그대로 믿어도 되는가?:
  - 이번 4장은 LOW Keypoint가 0개, 평균 Confidence 0.966 이상이라 그대로 사용했다.
  - Confidence가 낮은 손목이 있다면 17번 코드처럼 SKIP 해야 한다.

17번 코드(`17_arms_up_rule.py`)로도 확인했다.

| 이미지 | Min Confidence | 왼손목 − 어깨 | 오른손목 − 어깨 | Arms Up |
|---|---|---|---|---|
| Target Near | 0.960 | -0.297 | -0.628 | True |
| Target Far | 0.985 | -0.312 | -0.497 | True |
| Contrast Near | 0.990 | 1.538 | 1.570 | False |
| Contrast Far | 0.982 | 1.495 | 1.561 | False |

## 5. Confidence

- LOW Keypoint가 있었는가?: 4장 모두 없었다 (LOW 0개, 평균 Confidence 0.966~0.979).
- 결과 해석에 주의가 필요한 부분:
  - 정면에서 찍은 서 있는 자세라 모든 관절이 잘 보였다.
  - 3일차 본 실습의 bend_pose처럼 측면이나 숙인 자세에서는 Confidence가 높아도 위치가 틀릴 수 있다. 그래서 Mini Challenge는 교재 기준대로 정면에서 확인했다.

## 6. 영상에서 확인한 Feature 변화

`18_video_frame_features.py --source data/day03/challenge/videos/challenge_action.mp4 --max-frames 0`

```text
Total frames: 271
Valid pose frames: 271
Feature count per frame: 38
```

- 확인한 Feature: `left_wrist_y - left_shoulder_y`, `right_wrist_y - right_shoulder_y`
- 시작 자세의 값: 0~0.5초 왼쪽 1.52~1.53 / 오른쪽 1.54~1.58 (팔을 내림)
- 동작 중 변화: 0.75초 1.34 / 1.30 → 1.0초 0.32 / 0.25로 빠르게 줄어들었다. 손목이 어깨 쪽으로 올라왔다.
- Target 자세의 값: 1.17~2.68초 사이에 양쪽 모두 음수(약 -0.5 ~ 0)인 Frame이 74개 나왔다. 손목이 어깨보다 위에 있었다. 중간에 0 근처로 오르내린 것은 손을 흔들면서 높이가 조금씩 바뀌었기 때문이다.
- 다시 Contrast로: 3.25초 0.44 / 0.24 → 3.75초 1.45 / 1.22 → 4.0초 이후 1.55 / 1.48로 처음과 비슷하게 돌아왔다.
- 팔꿈치 각도도 시작 163° / 155° → 팔을 든 동안 57~128° → 끝 167° / 155°로 굽혔다가 다시 펴졌다.

## 7. 팀의 결론

- 이 동작을 가장 잘 설명한 Feature: **손목 y − 어깨 y (정규화 좌표)**
- 그 이유:
  - Target / Contrast에서 부호가 반대로 나뉘었고(-0.3 ~ -0.6 vs +1.5 ~ +1.6), Near / Far에서도 거의 같은 값이 나왔다.
  - 팔꿈치 각도는 팔을 어떻게 굽혀 들었는지에 따라 같은 Arms Up 안에서도 52°~130°로 달라져, 혼자서는 Arms Up을 설명하지 못했다.
- 여러 Frame을 시간 순서로 보는 것이 필요한 이유:
  - 한 Frame만 보면 "지금 팔이 위에 있다"는 것만 안다. 영상에서는 1.5 → 0.3 → 음수 → 다시 1.5로 바뀌는 흐름이 보여서, 팔을 올렸다가 유지하고 다시 내리는 **동작**으로 설명할 수 있었다.
  - Target 구간 안에서도 값이 0 근처로 오르내렸다. 한 Frame만 잘라 보면 이 순간은 "Arms Up이 아니다"로 보일 수 있다. 그래서 일정 시간 동안의 흐름을 함께 봐야 한다.
  - 이 시간 흐름이 4일차 Sequence의 재료가 된다.

## 37. 완료 체크

- [x] Target과 Contrast를 정했다. (Arms Up / Arms Down)
- [x] 이미지 4장을 준비했다. (직접 찍은 영상에서 캡처)
- [x] 영상 1개를 준비했다.
- [x] Target과 Contrast의 관절각도를 비교했다.
- [x] Confidence를 확인했다.
- [x] Target Near와 Far의 정규화 전·후를 비교했다.
- [x] 이미지 4장을 38-D Frame Feature로 만들었다.
- [x] 동작을 설명할 Feature 3개를 선택했다.
- [x] 선택한 이유를 설명할 수 있다.
- [x] 영상을 Frame별 38-D Feature로 변환했다.
- [x] Feature가 시간에 따라 변하는 것을 확인했다.
- [x] 팀 결과를 `feature_design.md`에 정리했다.

> 우리 팀(3팀 무인 상점 이상행동) 주제로 따로 진행한 추가 과제는 `reports/day03/team3/feature_design_team3.md`에 있다.
