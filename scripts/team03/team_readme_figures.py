"""README에 넣을 3팀 결과 그림을 만든다.

1. 조건별 / Feature 단계별 Validation Macro F1 막대그래프
2. 최종 45-D 모델 Test Confusion Matrix (6일차 결과 이미지를 docs/images로 복사)

사용: python scripts/team03/team_readme_figures.py
"""

import shutil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib import font_manager  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
DAY06 = ROOT / "reports" / "team_projects" / "team03" / "day06"
OUT = ROOT / "docs" / "images"

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
GRID = "#e4e3df"
CHOSEN = "#2a78d6"   # 최종으로 고른 조건
OTHER = "#a9c8ee"    # 나머지 조건 (같은 파랑의 밝은 단계)

CONDITION_NAMES = {
    "baseline_38d": "기본 38-D",
    "static_middle_frame": "가운데 Frame 1장",
    "coordinates_only": "좌표만 (각도 제외)",
    "team_45d": "3팀 45-D",
    "action_segment_only": "동작 구간만",
    "front_only": "정면만",
    "side_only": "측면만",
}
STEP_NAMES = {
    "38d_textbook": "38-D",
    "41d_+posture": "+ 자세 3개 (41-D)",
    "43d_+ankle_speed": "+ 발목 속도 (43-D)",
    "45d_+wrist_speed": "+ 손목 속도 (45-D)",
}


def set_korean_font() -> None:
    for name in ("Noto Sans CJK KR", "Noto Sans KR", "NanumGothic"):
        if any(f.name == name for f in font_manager.fontManager.ttflist):
            plt.rcParams["font.family"] = name
            return


def draw_bars(ax, labels, values, chosen, title):
    y = range(len(labels))[::-1]
    colors = [CHOSEN if label == chosen else OTHER for label in labels]
    ax.barh(list(y), values, color=colors, height=0.62, edgecolor=SURFACE, linewidth=2)
    for yi, value, label in zip(y, values, labels):
        weight = "bold" if label == chosen else "normal"
        ax.text(value + 0.012, yi, f"{value:.3f}", va="center", fontsize=10, color=INK, weight=weight)
    ax.set_yticks(list(y))
    ax.set_yticklabels(labels, fontsize=10.5, color=INK)
    ax.set_xlim(0, 1.0)
    ax.set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.tick_params(axis="x", colors=INK_2, labelsize=9)
    ax.tick_params(axis="y", length=0)
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.set_title(title, fontsize=12, color=INK, loc="left", pad=10, weight="bold")
    ax.set_xlabel("Validation Macro F1 (S04)", fontsize=9.5, color=INK_2)
    ax.set_facecolor(SURFACE)


def main() -> None:
    set_korean_font()
    OUT.mkdir(parents=True, exist_ok=True)

    conditions = pd.read_csv(DAY06 / "validation" / "condition_comparison.csv", encoding="utf-8-sig")
    steps = pd.read_csv(DAY06 / "feature_steps" / "feature_steps.csv", encoding="utf-8-sig")
    steps = steps[steps["step"].isin(STEP_NAMES)].set_index("step").loc[list(STEP_NAMES)].reset_index()

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), dpi=150, gridspec_kw={"width_ratios": [1.15, 1]})
    fig.patch.set_facecolor(SURFACE)

    draw_bars(
        axes[0],
        [CONDITION_NAMES[c] for c in conditions["condition"]],
        conditions["macro_f1"].tolist(),
        CONDITION_NAMES["team_45d"],
        "한 번에 한 조건만 바꾼 비교",
    )
    draw_bars(
        axes[1],
        [STEP_NAMES[s] for s in steps["step"]],
        steps["macro_f1"].tolist(),
        STEP_NAMES["45d_+wrist_speed"],
        "3팀 Feature를 한 단계씩 추가",
    )
    fig.suptitle(
        "3팀 17개 이상행동 분류 - Random Forest (진한 막대: 최종 선택 45-D)",
        fontsize=13.5, color=INK, x=0.01, ha="left", weight="bold",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    bar_path = OUT / "team03_validation_comparison.png"
    fig.savefig(bar_path, facecolor=SURFACE)
    print(f"[SAVED] {bar_path.relative_to(ROOT)}")

    confusion_src = DAY06 / "test" / "test_confusion_team_45d.png"
    confusion_dst = OUT / "team03_test_confusion_45d.png"
    shutil.copyfile(confusion_src, confusion_dst)
    print(f"[COPIED] {confusion_dst.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
