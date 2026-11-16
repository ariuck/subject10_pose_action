# Day02 Team Mini Challenge

> 상태: 촬영 사진 4장(`challenge_A_01/02.jpg`, `challenge_B_01/02.jpg`)이 아직 없어 미완료입니다.
> 사진을 `data/day02/images/`에 넣고 `07_pose_image_detail.py`, `09_compare_pose_images.py`를 실행한 뒤 아래 TODO를 채웁니다.

## 1. 우리 팀 실험

- 선택한 조건: TODO (거리 / 방향 / 가림 / 화면 이탈 중 하나)
- 조건 A: TODO
- 조건 B: TODO
- 바꾼 조건 한 가지: TODO

## 2. 결과

- 조건 A 평균 Keypoint Confidence: TODO
- 조건 B 평균 Keypoint Confidence: TODO

- 조건 A LOW 관절 수: TODO
- 조건 B LOW 관절 수: TODO

- 가장 불안정했던 이미지: TODO
- 가장 낮았던 Keypoint: TODO
- 해당 Confidence: TODO

## 참고 — 본수업 6장 결과 (실제 실행값)

| 이미지 | BBox Area Ratio | 평균 KP Conf | LOW 관절 수 | 가장 낮은 관절 |
|---|---|---|---|---|
| front.jpg | 0.300 | 0.9754 | 0 | right_ear 0.743 |
| arms_up.jpg | 0.675 | 0.9599 | 0 | right_ear 0.654 |
| distance_far.jpg | 0.149 | 0.9710 | 0 | right_ear 0.738 |
| occlusion.jpg | 0.265 | 0.9533 | 1 | right_ear 0.480 |
| partial_body.jpg | 0.339 | 0.9073 | 2 | left_ankle 0.491 |
| side.jpg | 0.199 | 0.7541 | 4 | right_ear 0.028 |

- side.jpg의 LOW 관절: right_eye 0.089, right_ear 0.028, right_elbow 0.167, right_wrist 0.250
  → 몸 반대편(오른쪽) 관절이 몸에 가려져, 오른팔 Skeleton이 실제로는 보이지 않는 몸통 위에 그려짐
- MP4: 60 Frame 모두 사람 검출. 25~26번 Frame(약 0.80초)에서 눈·귀 4개가 0.50 미만 → 바닥 자세로 얼굴이 작고 아래·옆을 향함

## 3. 우리 팀의 해석

- 어떤 조건에서 Pose 품질이 더 불안정했는가? TODO
- 실제 사진을 보았을 때 그 이유는 무엇이라고 생각하는가? TODO
- Confidence가 낮은 관절을 이후 분석에 사용할 때 무엇을 주의해야 하는가?
  → side.jpg의 right_wrist(0.250)처럼 Confidence가 낮은 관절은 좌표 자체가 틀렸을 수 있으므로,
    3일차 관절각도 계산에 그대로 넣지 말고 Threshold로 걸러야 함
