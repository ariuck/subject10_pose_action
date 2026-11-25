"""7일차 Mini Challenge용 영상을 폰 카메라 Stream에서 직접 녹화한다.

3초 카운트다운 후 정해진 시간만큼 녹화하고 MP4로 저장한다.
저장 FPS는 실제로 받은 Frame 수 / 녹화 시간으로 계산해서, 재생 속도가 실제와 같게 만든다.
녹화한 영상은 학습 데이터가 아니라 Runtime 평가용 입력이다.

사용 예:
  python scripts/team03/team_record_clip.py --url "http://<휴대폰IP>:4747/video" \
      --output data/day07/challenge/team03/videos/S93_A007_T01.mp4 --seconds 10
"""

import argparse
import time
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[2]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seconds", type=float, default=10.0)
    parser.add_argument("--countdown", type=float, default=3.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = Path(args.output)
    if not output.is_absolute():
        output = ROOT / output
    if output.exists():
        raise FileExistsError(f"이미 있는 파일입니다. T02처럼 Take 번호를 바꿔 주세요: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)

    capture = cv2.VideoCapture(args.url)
    if not capture.isOpened():
        raise RuntimeError(f"Stream을 열 수 없습니다: {args.url}")

    window = "Team03 Record (Q: cancel)"
    frames = []
    start = time.time()
    recording_start = None

    while True:
        ok, frame = capture.read()
        if not ok:
            print("[WARN] Frame을 받지 못했습니다. 다시 시도합니다.")
            time.sleep(0.1)
            continue

        now = time.time()
        elapsed = now - start
        view = frame.copy()

        if elapsed < args.countdown:
            text = f"READY {args.countdown - elapsed:.1f}s"
            color = (0, 255, 255)
        else:
            if recording_start is None:
                recording_start = now
                print("[REC] 녹화 시작")
            frames.append(frame)
            rec_time = now - recording_start
            text = f"REC {rec_time:.1f} / {args.seconds:.0f}s"
            color = (0, 0, 255)
            if rec_time >= args.seconds:
                break

        cv2.putText(view, text, (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.6, color, 4)
        cv2.imshow(window, view)
        if cv2.waitKey(1) & 0xFF in (ord("q"), ord("Q")):
            print("[CANCEL] 녹화를 취소했습니다. 파일은 저장하지 않습니다.")
            capture.release()
            cv2.destroyAllWindows()
            return

    duration = time.time() - recording_start
    capture.release()
    cv2.destroyAllWindows()

    fps = len(frames) / duration
    height, width = frames[0].shape[:2]
    writer = cv2.VideoWriter(str(output), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    for frame in frames:
        writer.write(frame)
    writer.release()

    print(f"[SAVED] {output}")
    print(f"  Frames: {len(frames)} / Duration: {duration:.2f}s / FPS: {fps:.2f} / Size: {width}x{height}")


if __name__ == "__main__":
    main()
