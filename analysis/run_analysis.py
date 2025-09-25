import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.constants import (
    FULL_CONFIDENCE_DISTRIBUTION_GRAPH_PATH,
    CORRECT_CONFIDENCE_DISTRIBUTION_GRAPH_PATH,
    INCORRECT_CONFIDENCE_DISTRIBUTION_GRAPH_PATH,
    MODEL_PERFORMANCE_SEGMENT_GRAPH_PATH,
    MODEL_PERFORMANCE_FAMILY_GRAPH_PATH,
    MODEL_PERFORMANCE_CLASS_GRAPH_PATH,
    ENSEMBLE_MODEL_OUTPUT_DATASET_PATH,
    ANALYSIS_DIR,
   ENSEMBLE_PIPELINE_OUTPUT_PATH
)


def draw_eda(df: pd.DataFrame) -> None:
    df["is_correct"] = df.apply(lambda x: x["segment"]==x["pred_segment"], axis=1)
    df_correct = df[df["is_correct"]==True]
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
    ax.set_title("Confidence Level Distribution by Hierarchy", fontsize=14, weight="bold", pad=15)
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
        ax.set_title(f"Percentage Distribution of Correctly vs Incorrectly Classified {level.title()}")
        ax.set_ylim(-105, 105)
    else:
        ax.set_ylabel("Number of Predictions")
        ax.set_title(f"Distribution of Correctly vs Incorrectly Classified {level.title()}")
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
        ax.set_xlabel("")

    ax.axhline(y=0, color="black", linestyle="-", linewidth=0.5)
    ax.spines['top'].set_visible(False)

    metric = res_df["correct"]
    top5 = metric.sort_values(ascending=False).head(5)
    bottom5 = metric.sort_values(ascending=False).tail(5)
    textstr = "Top 5 Correctly Classified:\n" + "\n".join([f"{idx}: {val}" for idx, val in top5.items()])
    textstr += "\n\nLowest 5 Correctly Classified:\n" + "\n".join([f"{idx}: {val}" for idx, val in bottom5.items()])

    props = dict(boxstyle="round", facecolor="white", alpha=0.8)
    ax.text(1.05, 0.5, textstr, transform=ax.transAxes,
            fontsize=9, verticalalignment="center", bbox=props, color="black")

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
        ax.set_title(f"Classification Results for {upper_level.title()}: {upper_label}\n"
                     f"Performance across all {sub_level}")
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

def main():
    df = pd.read_csv(ENSEMBLE_PIPELINE_OUTPUT_PATH)
    draw_eda(df)

if __name__ == "__main__":
    main()