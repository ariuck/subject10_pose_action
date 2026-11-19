"""4~6일차 Action 학습 파이프라인 (요약형, QA 단계 제외).

파일·코드 연결 + 상자마다 한두 줄 요약.
출력: 3팀_Action_학습파이프라인_요약.png
실행: python reports/day06/pipeline_assignment/draw_pipeline_simple.py
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import font_manager  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle  # noqa: E402

OUT = Path(__file__).resolve().parent / "3팀_Action_학습파이프라인_요약.png"
W, H = 1500, 1800
RED, GRAY = "#dc2626", "#4b5563"

KIND = {  # 채움, 테두리, 테두리 모양
    "config": ("#dbeafe", "#2563eb", "-"),
    "raw": ("#ede9fe", "#7c3aed", "-"),
    "code": ("#f3f4f6", "#374151", "-"),
    "check": ("#ffffff", "#6b7280", "--"),
    "csv": ("#dcfce7", "#16a34a", "-"),
    "model": ("#f3e8ff", "#9333ea", "-"),
    "result": ("#fef3c7", "#b45309", "-"),
}

# id: (중심 x, 중심 y, 너비, 높이, 종류, 제목, 요약)
N = {
    "project": (260, 150, 400, 84, "config", "configs/project.json", "1일차 직접 작성 · YOLO 모델·imgsz·conf 기준"),
    "action": (750, 150, 400, 84, "config", "configs/action_classes.csv", "4일차 직접 작성 · 정답표 A001~A006 → action_label"),
    "mp4": (1240, 150, 400, 84, "raw", "data/day05/videos/S0x/*.mp4", "강사 제공 원본 18개 · 3명 × 6동작 (예: S03_A006_T01)"),
    "c26": (750, 300, 520, 72, "code", "26  26_extract_keypoint_dataset.py", "Frame마다 YOLO Pose로 관절 17개 추출 · 파일명으로 정답 붙임"),
    "kp": (750, 435, 640, 84, "csv", "data/day05/keypoints/S0x/<label>/<clip_id>.csv × 18",
           "원본 Keypoint · 1행 = Frame 1개 · 95열 (x_px·y_px·norm·conf + action_label)"),
    "appr": (750, 570, 560, 84, "csv", "data/day05/manifests/approved_clip_manifest.csv",
             "승인된 Clip 18개 목록 · keypoint_csv 경로 + action_label (QA 과정은 생략)"),
    "split_cfg": (200, 705, 330, 84, "config", "configs/day05_subject_split.json", "5일차 직접 작성 · S01 / S02 / S03 배정"),
    "c30": (750, 705, 460, 72, "code", "30  30_build_subject_split.py", "사람(Subject) 단위로 Train · Val · Test 분리"),
    "c31": (1290, 705, 330, 72, "check", "31  31_check_split_leakage.py", "Subject·Clip 겹침 검사 → OK"),
    "s_train": (290, 845, 400, 84, "csv", "data/day05/splits/train.csv", "S01 · 6 Clip · 1행 = Clip 1개 + keypoint_csv 경로"),
    "s_val": (750, 845, 400, 84, "csv", "data/day05/splits/val.csv", "S02 · 6 Clip · 1행 = Clip 1개 + keypoint_csv 경로"),
    "s_test": (1210, 845, 400, 84, "csv", "data/day05/splits/test.csv", "S03 · 6 Clip · 1행 = Clip 1개 + keypoint_csv 경로"),
    "base_cfg": (200, 1000, 330, 84, "config", "configs/day06_baseline.json", "6일차 직접 작성 · 30 Step · 300 Tree · seed 42"),
    "c34": (750, 1000, 500, 72, "code", "34  34_build_sequence_dataset.py", "Keypoint → 38-D(좌표 34 + 각도 4) → 30 Step × 38 = 1,140"),
    "lib": (1300, 1000, 330, 84, "config", "src/sequence_utils.py", "공통 함수 · 38-D Frame Feature · 30 Step 보간"),
    "q_train": (290, 1140, 400, 84, "csv", "data/day06/sequences/train.csv", "S01 · 6행 × (메타 8 + 1,140 Feature)"),
    "q_val": (750, 1140, 400, 84, "csv", "data/day06/sequences/val.csv", "S02 · 6행 × (메타 8 + 1,140 Feature)"),
    "q_test": (1210, 1140, 400, 84, "csv", "data/day06/sequences/test.csv", "S03 · 6행 × (메타 8 + 1,140 Feature)"),
    "c35": (290, 1285, 400, 72, "code", "35  35_train_rf_baseline.py", "Random Forest 학습 · model.fit(X_train, y_train)"),
    "c36": (750, 1430, 400, 72, "code", "36  36_evaluate_validation.py", "S02 예측·채점 · 조건 비교용"),
    "c40": (1210, 1430, 400, 72, "code", "40  40_evaluate_test.py", "S03 예측·채점 · 조건 확정 후 마지막 1회"),
    "model": (290, 1430, 400, 84, "model", "artifacts/day06/rf_action_baseline.joblib",
              "학습된 모델 + rf_feature_columns.json (Feature 순서)"),
    "vout": (750, 1575, 440, 92, "result", "reports/day06/metrics · confusion",
             "validation_predictions.csv · validation_metrics.txt\nvalidation_confusion_matrix.png  (Acc 0.50)"),
    "tout": (1210, 1575, 440, 92, "result", "reports/day06/metrics · confusion",
             "test_predictions.csv · test_metrics.txt\ntest_confusion_matrix.png  (Acc 1.00)"),
    "c37": (290, 1575, 400, 72, "code", "37  37_analyze_misclassified.py", "틀린 Clip + 원본 MP4 경로 → misclassified.csv"),
}

# (출발, 도착, 글자, 정답 흐름?, 경유점)
E = [
    ("project", "c26", "YOLO 설정", False, []),
    ("action", "c26", "A006 → wave", True, []),
    ("mp4", "c26", "MP4 18개", False, []),
    ("c26", "kp", "자동 생성", True, []),
    ("kp", "appr", "Clip 목록", True, []),
    ("appr", "c30", "승인 Clip 18개", True, []),
    ("split_cfg", "c30", "Subject 규칙", False, []),
    ("c30", "c31", "", False, []),
    ("c30", "s_train", "", True, [(290, 770)]),
    ("c30", "s_val", "", True, []),
    ("c30", "s_test", "", True, [(1210, 770)]),
    ("s_train", "c34", "", True, [(290, 925)]),
    ("s_val", "c34", "keypoint_csv 경로 + 정답", True, []),
    ("s_test", "c34", "", True, [(1210, 925)]),
    ("base_cfg", "c34", "Step 30", False, []),
    ("lib", "c34", "함수", False, []),
    ("c34", "q_train", "", True, [(290, 1070)]),
    ("c34", "q_val", "1,140 Feature", True, []),
    ("c34", "q_test", "", True, [(1210, 1070)]),
    ("q_train", "c35", "X_train · y_train", True, []),
    ("base_cfg", "c35", "", False, [(70, 1000), (70, 1285)]),
    ("c35", "model", "joblib.dump", False, []),
    ("q_val", "c36", "X_val · y_true", True, []),
    ("q_test", "c40", "X_test · y_true", True, []),
    ("model", "c36", "모델", False, []),
    ("model", "c40", "모델", False, [(490, 1500), (1000, 1500)]),
    ("c36", "vout", "예측 결과", True, []),
    ("c40", "tout", "예측 결과", True, []),
    ("vout", "c37", "예측", False, []),
]


def shift_y(y):
    """구역 제목 자리를 만들려고 구역마다 상자를 아래로 내린다."""
    return y + (20 if y < 225 else 36 if y < 935 else 52)


N = {k: (cx, shift_y(cy), *rest) for k, (cx, cy, *rest) in N.items()}
E = [(a, b, t, r, [(x, shift_y(y)) for x, y in w]) for a, b, t, r, w in E]


def box(k):
    cx, cy, w, h, *_ = N[k]
    return cx - w / 2, cy - h / 2, w, h


def clip(k, p):
    cx, cy, w, h, *_ = N[k]
    dx, dy = p[0] - cx, p[1] - cy
    if dx == 0 and dy == 0:
        return cx, cy
    s = min((w / 2) / abs(dx) if dx else 1e9, (h / 2) / abs(dy) if dy else 1e9)
    return cx + dx * s, cy + dy * s


def main():
    for name in ("Noto Sans CJK KR", "Noto Sans KR", "NanumGothic"):
        if any(f.name == name for f in font_manager.fontManager.ttflist):
            plt.rcParams["font.family"] = name
            break
    fig = plt.figure(figsize=(W / 100, H / 100), dpi=100)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, W)
    ax.set_ylim(H, 0)
    ax.axis("off")

    ax.text(W / 2, 38, "교과10 4~6일차 Action 학습 파이프라인", fontsize=22, weight="bold", ha="center", color="#1e3a8a")
    ax.text(W / 2, 74, "원본 영상 → Keypoint → 사람 기준 Split → 1,140 Feature → Random Forest → Validation / Test",
            fontsize=12.5, ha="center", color="#4b5563")
    # 구역
    zones = [(92, 236, "준비", "#f8fafc", "#cbd5e1"),
             (244, 943, "5일차 Dataset", "#f0fdf4", "#bbf7d0"),
             (951, 1700, "6일차 학습 · 평가", "#fffbeb", "#fde68a")]
    for y0, y1, title, fc, ec in zones:
        ax.add_patch(Rectangle((20, y0), W - 40, y1 - y0 - 8, fc=fc, ec=ec, lw=1.2, zorder=0))
        ax.text(34, y0 + 15, title, fontsize=12.5, weight="bold", color="#334155", va="center", zorder=1)

    for src, dst, label, red, way in E:
        first = way[0] if way else N[dst][:2]
        last = way[-1] if way else N[src][:2]
        pts = [clip(src, first)] + list(way) + [clip(dst, last)]
        color = RED if red else GRAY
        lw = 2.2 if red else 1.4
        ls = (0, (5, 3)) if red else "-"
        ax.plot([p[0] for p in pts[:-1]], [p[1] for p in pts[:-1]], color=color, lw=lw, ls=ls, zorder=2)
        ax.add_patch(FancyArrowPatch(pts[-2], pts[-1], arrowstyle="-|>", mutation_scale=15, lw=lw, color=color,
                                     linestyle=ls, zorder=2))
        if label:
            segs = list(zip(pts[:-1], pts[1:]))
            a, b = max(segs, key=lambda s: (s[0][0] - s[1][0]) ** 2 + (s[0][1] - s[1][1]) ** 2)
            ax.text((a[0] + b[0]) / 2 + 6, (a[1] + b[1]) / 2, label, fontsize=9.5, color=color, ha="left",
                    va="center", zorder=7, bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.9))

    for k, (cx, cy, w, h, kind, title, desc) in N.items():
        x, y, _, _ = box(k)
        fc, ec, ls = KIND[kind]
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=8", fc=fc, ec=ec, lw=2,
                                    ls=ls, zorder=5))
        ax.text(cx, y + 22, title, fontsize=11.2, weight="bold", ha="center", va="center", color="#111827", zorder=6)
        ax.text(cx, y + 22 + (h - 22) / 2 + 2, desc, fontsize=9.6, ha="center", va="center", color="#374151",
                linespacing=1.35, zorder=6)

    # 범례
    ly = 1738
    items = [("config", "직접 작성 · 공통 함수"), ("raw", "원본 영상"), ("code", "Python 코드 (번호)"),
             ("check", "검사만 (출력 없음)"), ("csv", "자동 생성 CSV"), ("model", "학습 모델"), ("result", "평가 결과")]
    x = 40
    for kind, text in items:
        fc, ec, ls = KIND[kind]
        ax.add_patch(FancyBboxPatch((x, ly - 9), 30, 18, boxstyle="round,pad=0,rounding_size=4", fc=fc, ec=ec,
                                    lw=1.6, ls=ls, zorder=5))
        ax.text(x + 38, ly, text, fontsize=10, va="center", zorder=6)
        x += 38 + len(text) * 11.5 + 28
    ax.plot([x, x + 34], [ly, ly], color=RED, lw=2.2, ls=(0, (5, 3)))
    ax.text(x + 42, ly, "action_label(정답) 흐름", fontsize=10, color=RED, va="center")
    ax.text(W / 2, 1778,
            "정답은 YOLO가 예측하지 않는다. action_classes.csv에서 26번이 붙인 action_label이 승인 목록 · Split · "
            "Sequence를 거쳐 학습의 y_train, 평가의 y_true가 된다.",
            fontsize=10.5, ha="center", va="center", color="#374151")

    fig.savefig(OUT, dpi=100, facecolor="white")
    print("[SAVED]", OUT)


if __name__ == "__main__":
    main()
