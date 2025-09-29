import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.constants import (
    GPC_PATH,
    DATASET_PATH,
    FULL_CONFIDENCE_DISTRIBUTION_GRAPH_PATH,
    CORRECT_CONFIDENCE_DISTRIBUTION_GRAPH_PATH,
    INCORRECT_CONFIDENCE_DISTRIBUTION_GRAPH_PATH,
    MODEL_PERFORMANCE_SEGMENT_GRAPH_PATH,
    MODEL_PERFORMANCE_FAMILY_GRAPH_PATH,
    MODEL_PERFORMANCE_CLASS_GRAPH_PATH,
    ENSEMBLE_MODEL_OUTPUT_DATASET_PATH,
    ANALYSIS_DIR,
    GPC_COVERAGE_GRAPH_PATH,
    ENSEMBLE_PIPELINE_OUTPUT_PATH,
    PRODUCT_DISTRIBUTION_ACCROSS_SEGMENTS_GRAPH_PATH,
    PRODUCT_DISTRIBUTION_ACCROSS_FAMILIES_GRAPH_PATH,
    PRODUCT_DISTRIBUTION_ACCROSS_CLASSES_GRAPH_PATH,
)

def plot_level_distribution(
    df: pd.DataFrame, 
    level: str,
    box_pos_x: float,
    box_pos_y: float,
    img_path: str
) -> None:
    fig, ax = plt.subplots(figsize=(18, 10))

    counts = df[level].value_counts()

    colors = plt.cm.viridis(np.linspace(0, 1, len(counts)))

    bars = ax.bar(range(len(counts)), counts.values, 
                color=colors, alpha=0.8, edgecolor='white', linewidth=0.7)

    x_smooth = np.linspace(0, len(counts)-1, 300)

    def thousands_formatter(x, pos):
        return f'{x/1000:.0f}K' if x >= 1000 else f'{x:.0f}'

    ax.yaxis.set_major_formatter(FuncFormatter(thousands_formatter))

    ax.set_xlabel(level.capitalize(), fontsize=14, fontweight='600', color='#34495e')
    ax.set_ylabel('Number of Products', fontsize=14, fontweight='600', color='#34495e')

    cum_counts = counts.cumsum()
    total = counts.sum()

    # Remove 100% cutoff - only keep 70% and 90%
    cutoffs = {
        "70%": cum_counts[cum_counts <= total * 0.7].index[-1],
        "90%": cum_counts[cum_counts <= total * 0.9].index[-1]
    }

    for pct, seg in cutoffs.items():
        idx = counts.index.get_loc(seg)
        ax.axvline(idx + 0.5, color="red", linestyle="--", linewidth=1.5)
        ax.text(idx + 0.5, ax.get_ylim()[1]*0.95, pct,
                color="red", fontsize=12, fontweight="bold", ha="right")

    labels = []
    for label in counts.index:
        if len(label) > 11:
            words = label.split()
            if len(words) > 1:
                mid = len(words) // 2
                label = ' '.join(words[:mid]) + ' \\ ' + ' '.join(words[mid:])
            else:
                label = label[:8] + '\\' + label[8:]
        labels.append(label)

    ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5, axis='y')
    ax.set_axisbelow(True)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('#bdc3c7')
    ax.spines['bottom'].set_color('#bdc3c7')

    if level == "segment":
        ax.set_xticks(range(len(counts)))
        ax.set_xticklabels(labels, rotation=50, ha='right', fontsize=10, 
                        verticalalignment='top')
        total = sum(counts.values)
        for i, (bar, value) in enumerate(zip(bars, counts.values)):
            if i < 10:
                ratio = value / total * 100
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + max(counts.values) * 0.01,
                    f'{ratio:.1f}%', 
                    ha='center',
                    va='bottom',
                    fontsize=9.5,
                    fontweight='bold',
                    color='#2c3e50'
                )
        top_3_indices = counts.head(3).index
        for i, lvl in enumerate(counts.index):
            if lvl in top_3_indices:
                bars[i].set_edgecolor('#e74c3c')
                bars[i].set_linewidth(3)
                ax.get_xticklabels()[i].set_fontweight('bold')

    # Create statistics box similar to reference code
    top5 = counts.head(5)
    bottom5 = counts.tail(5)

    textstr = f"Top 5 {level.capitalize()}s by Product Count:\n"
    textstr += "\n".join([f"{idx}: {val:,}" for idx, val in top5.items()])
    textstr += f"\n\nLowest 5 {level.capitalize()}s by Product Count:\n"
    textstr += "\n".join([f"{idx}: {val:,}" for idx, val in bottom5.items()])

    props = dict(boxstyle="round", facecolor="white", alpha=0.8)
    ax.text(box_pos_x, box_pos_y, textstr, transform=ax.transAxes,
            fontsize=17, verticalalignment="center", bbox=props, color="black")

    ax.set_yscale("log")

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.15)

    plt.savefig(img_path, bbox_inches='tight')
    plt.close()

def plot_gpc_coverage(df: pd.DataFrame, df_gpc: pd.DataFrame, img_path: str):
    totals = {
        "Segment": df_gpc["SegmentTitle"].nunique(),
        "Family": df_gpc["FamilyTitle"].nunique(),
        "Class": df_gpc["ClassTitle"].nunique(),
    }

    actuals = {
        "Segment": df["segment"].nunique(),
        "Family": df["family"].nunique(),
        "Class": df["class"].nunique(),
    }

    levels = list(totals.keys())
    coverage = [actuals[l] / totals[l] * 100 for l in levels]

    plt.figure(figsize=(8,6))
    bars = plt.bar(levels, coverage, color=["steelblue", "orange", "green"])

    for i, bar in enumerate(bars):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, height + 1,
                f"{height:.1f}%", ha='center', va='bottom', fontsize=11, fontweight="bold")
        plt.text(bar.get_x() + bar.get_width()/2, -7,
                f"{actuals[levels[i]]}/{totals[levels[i]]}", ha='center', va='top', fontsize=10, color="black")

    plt.ylim(0, 110)
    plt.ylabel("Coverage (%)")
    
    plt.savefig(img_path, bbox_inches='tight')
    plt.close()

def plot_confidence_distribution(df: pd.DataFrame, img_path: str) -> None:
    levels = ["segment", "family", "class"]
    conf_levels = ["Low", "Medium", "High"]
    ratios = {}
    for level in levels:
        counts = df[df[level].notna()]["confidence_level"].value_counts(normalize=True) * 100
        ratios[level] = counts.reindex(conf_levels, fill_value=0)
    ratios_df = pd.DataFrame(ratios).T
    colors = {"Low": "#d73027", "Medium": "#fc8d59", "High": "#1a9850"}
    fig, ax = plt.subplots(figsize=(10, 6))
    ratios_df.plot(
        kind="bar",
        stacked=True,
        ax=ax,
        color=[colors[c] for c in conf_levels],
        edgecolor="black",
        linewidth=0.6
    )
    ax.set_ylabel("Percentage (%)", fontsize=12)
    ax.set_xlabel("Hierarchy Level", fontsize=12)
    # ax.set_title("Confidence Level Distribution by Hierarchy", fontsize=14, weight="bold", pad=15)
    ax.set_ylim(0, 100)
    ax.set_xticklabels([lbl.capitalize() for lbl in ratios_df.index], rotation=0, fontsize=11)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{int(y)}%"))
    ax.legend(
        title="Confidence Level", 
        fontsize=10, 
        title_fontsize=11, 
        loc="upper center", 
        bbox_to_anchor=(0.5, -0.15),
        ncol=3,
        frameon=True,
        fancybox=True,
        shadow=True
    )
    for container in ax.containers:
        for bar, label in zip(container, container.datavalues):
            ax.text(
                bar.get_x() + bar.get_width() + 0.02,
                bar.get_y() + bar.get_height() / 2,
                f"{label:.1f}%",
                va="center", ha="left",
                fontsize=9, color="black", weight="bold"
            )
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.2)
    plt.savefig(img_path, bbox_inches='tight')
    plt.close()

def plot_classification_results(df: pd.DataFrame, level: str, img_path: str, mode: str = "absolute_value") -> None:
    df = df.copy()
    col_true = level
    col_pred = f"pred_{level}"
    df["correct"] = df[col_true] == df[col_pred]

    results = []
    for label in df[col_true].dropna().unique():
        label_df = df[df[col_true] == label]
        total = len(label_df)
        correct = (label_df["correct"]).sum()
        incorrect = total - correct
        if mode == "percentage":
            results.append({
                "label": label,
                "correct": (correct / total) * 100 if total > 0 else 0,
                "incorrect": (incorrect / total) * 100 if total > 0 else 0,
                "total": total
            })
        else:
            results.append({
                "label": label,
                "correct": correct,
                "incorrect": incorrect,
                "total": total
            })

    res_df = pd.DataFrame(results).set_index("label").sort_values("total", ascending=False)

    fig, ax = plt.subplots(figsize=(14, 6))
    x_pos = np.arange(len(res_df.index))

    bars_correct = ax.bar(x_pos, res_df["correct"], color="green", alpha=0.7, label="Correct")
    bars_incorrect = ax.bar(
        x_pos,
        -res_df["incorrect"],
        color="red",
        alpha=0.7,
        label="Incorrect"
    )

    ax.set_xlabel(level.title())
    ax.legend(title="Prediction")

    if mode == "percentage":
        ax.set_ylabel("Percentage (%)")
        # ax.set_title(f"Percentage Distribution of Correctly vs Incorrectly Classified {level.title()}")
        ax.set_ylim(-105, 105)
    else:
        ax.set_ylabel("Number of Predictions")
        # ax.set_title(f"Distribution of Correctly vs Incorrectly Classified {level.title()}")
        ax.set_yscale("symlog")

    ax.set_xticks(x_pos)
    if level == "segment":
        ax.set_xticklabels(res_df.index, rotation=45, ha="right")
        for i, (c, ic) in enumerate(zip(res_df["correct"], res_df["incorrect"])):
            if c > 0:
                text = f"{c:.0f}%" if mode == "percentage" else f"{int(c)}"
                ax.text(i, c + (2 if mode == "percentage" else c * 0.1), text,
                        ha="center", va="bottom", fontsize=6, color="green", weight="bold")
            if ic > 0:
                text = f"{ic:.0f}%" if mode == "percentage" else f"{int(ic)}"
                ax.text(i, -(ic + (2 if mode == "percentage" else ic * 0.1)), text,
                        ha="center", va="top", fontsize=6, color="red", weight="bold")
    else:
        ax.set_xticklabels([])
        ax.set_xlabel(f"{level}")

    ax.axhline(y=0, color="black", linestyle="-", linewidth=0.5)
    ax.spines['top'].set_visible(False)

    metric = res_df["correct"]
    top5 = metric.sort_values(ascending=False).head(5)
    bottom5 = metric.sort_values(ascending=False).tail(5)
    textstr = "Top 5 Correctly Classified:\n" + "\n".join([f"{idx}: {val}" for idx, val in top5.items()])
    textstr += "\n\nLowest 5 Correctly Classified:\n" + "\n".join([f"{idx}: {val}" for idx, val in bottom5.items()])

    props = dict(boxstyle="round", facecolor="white", alpha=0.8)
    ax.text(1.02, 0.5, textstr, transform=ax.transAxes,
            fontsize=15, verticalalignment="center", bbox=props, color="black")

    base, ext = os.path.splitext(img_path)
    img_path = f"{base}_{mode}{ext}"
    plt.savefig(img_path, bbox_inches="tight")
    plt.close()

def plot_classification_by_sublevel(df: pd.DataFrame, upper_level: str, sub_level: str, base_path: str) -> None:
    df = df.copy()
    col_true = upper_level
    col_pred = f"pred_{upper_level}"
    df["correct"] = df[col_true] == df[col_pred]

    upper_levels = df[upper_level].dropna().unique()
    save_dir = os.path.join(base_path, f"model_performance_by_each_{upper_level}")
    os.makedirs(save_dir, exist_ok=True)

    for upper_idx, upper_label in enumerate(upper_levels):
        upper_df = df[df[upper_level] == upper_label].copy()
        if len(upper_df) == 0:
            continue

        counts = (
            upper_df.groupby([sub_level, "correct"])
            .size()
            .unstack(fill_value=0)
        )
        if len(counts) == 0:
            continue

        total_counts = counts.sum(axis=1).sort_values(ascending=False)
        counts = counts.reindex(total_counts.index)

        fig, ax = plt.subplots(figsize=(12, 6))
        correct_counts = counts.get(True, pd.Series(0, index=counts.index))
        incorrect_counts = counts.get(False, pd.Series(0, index=counts.index))

        x_pos = np.arange(len(counts.index))
        
        ax.bar(x_pos, correct_counts, color="green", alpha=0.7, label="Correct", width=0.6)
        ax.bar(x_pos, -incorrect_counts, color="red", alpha=0.7, label="Incorrect", width=0.6)

        ax.set_ylabel("Number of Predictions")
        ax.set_xlabel(sub_level.title())
        # ax.set_title(f"Classification Results for {upper_level.title()}: {upper_label}\n"
        #              f"Performance across all {sub_level}")
        ax.legend(title="Prediction")
        ax.set_yscale("symlog")
        ax.set_xticks(x_pos)
        ax.set_xticklabels(counts.index, rotation=45, ha="right")
        
        ax.axhline(y=0, color='black', linewidth=0.8, alpha=0.3)
        ax.spines['top'].set_visible(False)

        for i, (correct, incorrect) in enumerate(zip(correct_counts, incorrect_counts)):
            if correct > 0:
                ax.text(i, correct + correct * 0.05, f"{int(correct)}", ha="center", va="bottom",
                        fontsize=8, color="black", weight="bold") 
            if incorrect > 0:
                ax.text(i, -incorrect - incorrect * 0.05, f"{int(incorrect)}", ha="center", va="top",
                        fontsize=8, color="black", weight="bold") 

        total_correct = correct_counts.sum()
        total_incorrect = incorrect_counts.sum()
        total_predictions = total_correct + total_incorrect
        accuracy = (total_correct / total_predictions * 100) if total_predictions > 0 else 0

        textstr = (
            f"{upper_level}: {upper_label}\n"
            f"Number of {sub_level}s: {len(counts)}\n"
            f"Total Predictions: {total_predictions}\n"
            f"Correct: {total_correct}\n"
            f"Incorrect: {total_incorrect}\n"
            f"Overall Accuracy: {accuracy:.1f}%"
        )
        props = dict(boxstyle="round", facecolor="white", alpha=0.8)
        ax.text(1.05, 0.5, textstr, transform=ax.transAxes,
                fontsize=10, verticalalignment="center",
                bbox=props, color="black")

        plt.tight_layout()

        safe_upper_label = str(upper_label).replace("/", "_").replace("\\", "_").replace(" ", "_")
        img_path = os.path.join(save_dir, f"{safe_upper_label}.png")
        plt.savefig(img_path, bbox_inches="tight")
        plt.close()

        print(f"Saved plot for {upper_level} '{upper_label}' showing performance across all {sub_level}s to {img_path}")

def draw_eda(df: pd.DataFrame, df_full: pd.DataFrame, df_gpc: pd.DataFrame) -> None:
    plot_gpc_coverage(df_full, df_gpc, GPC_COVERAGE_GRAPH_PATH)
    plot_level_distribution(df_full, "segment", 0.65, 0.72, PRODUCT_DISTRIBUTION_ACCROSS_SEGMENTS_GRAPH_PATH)
    plot_level_distribution(df_full, "family", 0.52, 0.75, PRODUCT_DISTRIBUTION_ACCROSS_FAMILIES_GRAPH_PATH)
    plot_level_distribution(df_full, "class", 0.7, 0.6, PRODUCT_DISTRIBUTION_ACCROSS_CLASSES_GRAPH_PATH)
    df_correct = df[df["is_correct_segment"]==True]
    df_incorrect = df[df["is_correct"]==False]
    plot_confidence_distribution(df, FULL_CONFIDENCE_DISTRIBUTION_GRAPH_PATH)
    plot_confidence_distribution(df_correct, CORRECT_CONFIDENCE_DISTRIBUTION_GRAPH_PATH)
    plot_confidence_distribution(df_incorrect, INCORRECT_CONFIDENCE_DISTRIBUTION_GRAPH_PATH)

    plot_classification_results(df, "segment", MODEL_PERFORMANCE_SEGMENT_GRAPH_PATH, "absolute_value")
    plot_classification_results(df, "family", MODEL_PERFORMANCE_FAMILY_GRAPH_PATH, "absolute_value")
    plot_classification_results(df, "class", MODEL_PERFORMANCE_CLASS_GRAPH_PATH, "absolute_value")
    plot_classification_results(df, "segment", MODEL_PERFORMANCE_SEGMENT_GRAPH_PATH, "percentage")
    plot_classification_results(df, "family", MODEL_PERFORMANCE_FAMILY_GRAPH_PATH, "percentage")
    plot_classification_results(df, "class", MODEL_PERFORMANCE_CLASS_GRAPH_PATH, "percentage")

    plot_classification_by_sublevel(df, "segment", "family", ANALYSIS_DIR)
    plot_classification_by_sublevel(df, "family", "class", ANALYSIS_DIR)


def main():
    df_gpc = pd.read_csv(GPC_PATH)
    df_full = pd.read_csv(DATASET_PATH)
    df = pd.read_csv(ENSEMBLE_MODEL_OUTPUT_DATASET_PATH)
    draw_eda(df, df_full, df_gpc)

if __name__ == "__main__":
    main()