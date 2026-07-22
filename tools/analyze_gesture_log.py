"""
Script for my evaluation section — reads the CSV logs from mac/main.py and
spits out confusion matrices + accuracy/precision/recall/F1 for both hands.

Compares the target_* columns (what I told it I was actually doing) against
the *_stable columns (what the classifier actually decided, after
debouncing). Right and left hands get their own separate confusion matrix
since they're different classifiers with different possible labels.

Needs mac/requirements-analysis.txt installed (sklearn/matplotlib/pandas) —
kept these out of the main requirements.txt since you don't need them just
to fly the drone, only for this report stuff.

Usage:
    python tools/analyze_gesture_log.py --logs logs/*.csv
    python tools/analyze_gesture_log.py --logs logs/session1.csv logs/session2.csv
    python tools/analyze_gesture_log.py --logs logs/*.csv --out-dir report/
"""

import argparse
import glob
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import (
    ConfusionMatrixDisplay, classification_report, confusion_matrix, f1_score,
    precision_recall_fscore_support,
)

NONE_LABEL = "NONE"
BLUES_CMAP = "Blues"
# light -> dark blue for precision/recall/f1 bars, keeps the whole report on one palette
SCORE_COLORS = {"precision": "#9ecae1", "recall": "#4292c6", "f1": "#08519c"}

RIGHT_LABELS = ["THROTTLE_UP", "THROTTLE_DOWN", "THROTTLE_HOLD", "RIGHT_UNKNOWN", "RIGHT_MISSING"]
LEFT_LABELS = [
    "ROLL_LEFT", "ROLL_RIGHT", "PITCH_FORWARD", "PITCH_BACKWARD",
    "DIRECTION_NEUTRAL", "LEFT_UNKNOWN", "LEFT_MISSING",
]


def resolve_log_paths(patterns):
    """Turns --logs args into an actual file list. Runs each one through
    glob, but if a pattern doesn't match anything, just keep it as a plain
    path (so passing a single non-glob file still works normally)."""
    paths = []
    for pattern in patterns:
        matches = sorted(glob.glob(pattern))
        paths.extend(matches if matches else [pattern])
    seen = set()
    unique_paths = []
    for p in paths:
        if p not in seen:
            seen.add(p)
            unique_paths.append(p)
    return unique_paths


def load_logs(paths):
    """Reads a bunch of log CSVs and mashes them into one big DataFrame."""
    frames = []
    for path in paths:
        if not os.path.isfile(path):
            print(f"warning: log file not found, skipping: {path}", file=sys.stderr)
            continue
        frames.append(pd.read_csv(path))
    if not frames:
        raise SystemExit("no valid log files found")
    return pd.concat(frames, ignore_index=True)


def compute_hand_metrics(df, target_col, stable_col, labels, min_settle_frames=0):
    """Does the actual metric computation for one hand. Only looks at rows
    where I actually set a ground-truth target (skips the untagged NONE
    ones). Returns None if there weren't any tagged rows at all.

    min_settle_frames (default 0, i.e. no filtering) additionally drops rows
    where frames_since_target_change is below the threshold — those are
    likely still "moving into" the gesture rather than holding it cleanly,
    right after a ground-truth key press."""
    subset = df[df[target_col] != NONE_LABEL]
    if subset.empty:
        return None

    excluded_by_settle = 0
    if min_settle_frames > 0 and "frames_since_target_change" in subset.columns:
        before = len(subset)
        subset = subset[subset["frames_since_target_change"] >= min_settle_frames]
        excluded_by_settle = before - len(subset)
        if subset.empty:
            return None

    y_true = subset[target_col].astype(str)
    y_pred = subset[stable_col].astype(str)

    cm = confusion_matrix(y_true, y_pred, labels=labels)
    report = classification_report(y_true, y_pred, labels=labels, zero_division=0)
    accuracy = float((y_true.values == y_pred.values).mean())
    macro_f1 = float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0))
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=0)

    return {
        "labels": labels,
        "confusion_matrix": cm,
        "report": report,
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "n": len(subset),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "support": support,
        "excluded_by_settle_filter": excluded_by_settle,
    }


def plot_confusion_matrix(metrics, hand_name, out_path):
    """Saves the confusion matrix as a PNG heatmap (blue theme, for the report)."""
    disp = ConfusionMatrixDisplay(confusion_matrix=metrics["confusion_matrix"], display_labels=metrics["labels"])
    fig, ax = plt.subplots(figsize=(6, 5))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    disp.plot(ax=ax, cmap=BLUES_CMAP, colorbar=True, xticks_rotation=45, values_format="d")

    # text stays black/dark so it's still readable against the blue squares
    ax.set_title(f"{hand_name} hand confusion matrix", color="black")
    ax.xaxis.label.set_color("black")
    ax.yaxis.label.set_color("black")
    for tick_label in ax.get_xticklabels() + ax.get_yticklabels():
        tick_label.set_color("black")

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, facecolor="white")
    plt.close(fig)


def plot_class_scores(metrics, hand_name, out_path):
    """Bar chart of precision/recall/F1 for each gesture that was actually
    tested (skips labels with zero support so the chart isn't full of empty
    bars for gestures that never showed up as ground truth)."""
    labels = metrics["labels"]
    tested = [i for i, s in enumerate(metrics["support"]) if s > 0]
    if not tested:
        return

    names = [labels[i] for i in tested]
    precision = [metrics["precision"][i] for i in tested]
    recall = [metrics["recall"][i] for i in tested]
    f1 = [metrics["f1"][i] for i in tested]

    x = list(range(len(names)))
    width = 0.25
    fig, ax = plt.subplots(figsize=(max(6, len(names) * 1.4), 5))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    ax.bar([i - width for i in x], precision, width, label="Precision", color=SCORE_COLORS["precision"])
    ax.bar(x, recall, width, label="Recall", color=SCORE_COLORS["recall"])
    ax.bar([i + width for i in x], f1, width, label="F1", color=SCORE_COLORS["f1"])

    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=30, ha="right", color="black")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score", color="black")
    ax.set_title(f"{hand_name} hand — per-gesture score", color="black")
    ax.tick_params(colors="black")
    for spine in ax.spines.values():
        spine.set_color("black")
    ax.legend(facecolor="white", edgecolor="black", labelcolor="black")

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, facecolor="white")
    plt.close(fig)


def print_summary(hand_name, metrics, target_col, stable_col):
    if metrics is None:
        print(f"\n=== {hand_name} hand ===")
        print(f"No ground-truth-tagged rows found ({target_col} was always {NONE_LABEL}); skipping.")
        return
    print(f"\n=== {hand_name} hand: {metrics['n']} ground-truth-tagged frames "
          f"({target_col} vs {stable_col}) ===")
    if metrics["excluded_by_settle_filter"] > 0:
        print(f"(excluded {metrics['excluded_by_settle_filter']} rows below --min-settle-frames)")
    print(f"Accuracy: {metrics['accuracy']:.4f}   Macro F1: {metrics['macro_f1']:.4f}")
    print(metrics["report"])


def print_perf_summary(df):
    """Prints avg/median/max FPS and round-trip latency across everything loaded."""
    print("\n=== Performance ===")
    if "fps" in df.columns and df["fps"].notna().any():
        fps = df["fps"].dropna()
        print(f"FPS           — avg: {fps.mean():.1f}  median: {fps.median():.1f}  max: {fps.max():.1f}")
    else:
        print("FPS           — no data in these logs")

    if "round_trip_ms" in df.columns and df["round_trip_ms"].notna().any():
        rt = df["round_trip_ms"].dropna()
        print(f"Round-trip ms — avg: {rt.mean():.1f}  median: {rt.median():.1f}  max: {rt.max():.1f}")
    else:
        print("Round-trip ms — no data (dry-run session, or the Pi never replied)")


def print_condition_breakdown(df, target_col, stable_col, labels, hand_name, min_settle_frames):
    """Per-test_condition accuracy/F1 table, so different lighting/background/
    distance runs can be compared side by side. Only called when there's more
    than one distinct test_condition in the loaded logs."""
    print(f"\n--- {hand_name} hand by test_condition ---")
    print(f"{'condition':<20} {'n':>6} {'accuracy':>10} {'macro_f1':>10}")
    for condition in sorted(df["test_condition"].dropna().unique()):
        condition_df = df[df["test_condition"] == condition]
        metrics = compute_hand_metrics(condition_df, target_col, stable_col, labels, min_settle_frames)
        if metrics is None:
            print(f"{condition:<20} {'--':>6} {'--':>10} {'--':>10}")
            continue
        print(f"{condition:<20} {metrics['n']:>6} {metrics['accuracy']:>10.4f} {metrics['macro_f1']:>10.4f}")


def main():
    parser = argparse.ArgumentParser(
        description="Confusion matrices + accuracy/precision/recall/F1 from mac/main.py gesture logs.")
    parser.add_argument("--logs", nargs="+", required=True,
                         help="CSV log file path(s) and/or glob pattern(s), e.g. logs/*.csv")
    parser.add_argument("--out-dir", default=".", help="directory to write confusion matrix PNGs into")
    parser.add_argument("--min-settle-frames", type=int, default=0,
                         help="drop rows within this many frames of a ground-truth target switch "
                              "(still 'moving into' the gesture); 0 = no filtering (default)")
    args = parser.parse_args()

    paths = resolve_log_paths(args.logs)
    if not paths:
        raise SystemExit("no log files matched --logs")

    df = load_logs(paths)
    os.makedirs(args.out_dir, exist_ok=True)

    print_perf_summary(df)

    hands = [
        ("Right", "target_right_gesture", "right_stable", RIGHT_LABELS,
         "confusion_matrix_right.png", "scores_right.png"),
        ("Left", "target_left_gesture", "left_stable", LEFT_LABELS,
         "confusion_matrix_left.png", "scores_left.png"),
    ]

    multiple_conditions = "test_condition" in df.columns and df["test_condition"].nunique(dropna=True) > 1

    for hand_name, target_col, stable_col, labels, cm_png_name, score_png_name in hands:
        metrics = compute_hand_metrics(df, target_col, stable_col, labels, args.min_settle_frames)
        print_summary(hand_name, metrics, target_col, stable_col)
        if metrics is not None:
            cm_path = os.path.join(args.out_dir, cm_png_name)
            plot_confusion_matrix(metrics, hand_name, cm_path)
            print(f"Saved: {cm_path}")

            score_path = os.path.join(args.out_dir, score_png_name)
            plot_class_scores(metrics, hand_name, score_path)
            print(f"Saved: {score_path}")

        if multiple_conditions:
            print_condition_breakdown(df, target_col, stable_col, labels, hand_name, args.min_settle_frames)


if __name__ == "__main__":
    main()
