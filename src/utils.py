import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

from typing import List

from constants import (
    RANDOM_STATE,
    FULL_CONFIDENCE_DISTRIBUTION_GRAPH_PATH,
    CORRECT_CONFIDENCE_DISTRIBUTION_GRAPH_PATH,
    INCORRECT_CONFIDENCE_DISTRIBUTION_GRAPH_PATH,
    MODEL_PERFORMANCE_SEGMENT_GRAPH_PATH,
    MODEL_PERFORMANCE_FAMILY_GRAPH_PATH,
    MODEL_PERFORMANCE_CLASS_GRAPH_PATH,
)

def split_dataset(dataset_path: str, train_dataset_path: str, test_dataset_path: str):
    df = pd.read_csv(dataset_path)
    df.dropna(subset=["class"], inplace=True)
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=RANDOM_STATE)

    train_df.to_csv(train_dataset_path, index=False)
    test_df.to_csv(test_dataset_path, index=False)

def draw_eda(df: pd.DataFrame) -> None:
    df["is_correct"] = df.apply(lambda x: x["segment"]==x["pred_segment"], axis=1)
    df_correct = df[df["is_correct"]==True]
    df_incorrect = df[df["is_correct"]==False]
    plot_confidence_distribution(df, FULL_CONFIDENCE_DISTRIBUTION_GRAPH_PATH)
    plot_confidence_distribution(df_correct, CORRECT_CONFIDENCE_DISTRIBUTION_GRAPH_PATH)
    plot_confidence_distribution(df_incorrect, INCORRECT_CONFIDENCE_DISTRIBUTION_GRAPH_PATH)

    plot_classification_results(df, "segment", MODEL_PERFORMANCE_SEGMENT_GRAPH_PATH)
    plot_classification_results(df, "family", MODEL_PERFORMANCE_FAMILY_GRAPH_PATH)
    plot_classification_results(df, "class", MODEL_PERFORMANCE_CLASS_GRAPH_PATH)

def plot_confidence_distribution(df: pd.DataFrame, img_path: str) -> None:
    levels = ["segment", "family", "class"]
    conf_levels = ["Low", "Medium", "High"]

    ratios = {}
    for level in levels:
        counts = df[df[level].notna()]["confidence_level"].value_counts(normalize=True) * 100
        ratios[level] = counts.reindex(conf_levels, fill_value=0)

    ratios_df = pd.DataFrame(ratios).T
    colors = {"Low": "#d73027", "Medium": "#fc8d59", "High": "#1a9850"}
    _, ax = plt.subplots(figsize=(10, 6))
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

    ax.legend(title="Confidence Level", fontsize=10, title_fontsize=11, loc="upper center", bbox_to_anchor=(0.5, -0.08), ncol=3, frameon=False)
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

    plt.savefig(img_path)
    plt.close()

def plot_classification_results(df: pd.DataFrame, level: str, img_path: str) -> None:
    df = df.copy()
    col_true = level
    col_pred = f"pred_{level}"

    df["correct"] = df[col_true] == df[col_pred]

    counts = (
        df.groupby([col_true, "correct"])
        .size()
        .unstack(fill_value=0)
    )

    colors = {False: "red", True: "green"}
    ax = counts.plot(
        kind="bar",
        stacked=False,
        figsize=(14, 6),
        color=[colors[c] for c in counts.columns]
    )

    ax.set_ylabel("Number of Predictions")
    ax.set_title(f"Distribution of Correctly Classified {level.title()}s")
    ax.legend(["Incorrect", "Correct"], title="Prediction")
    ax.set_yscale("log")

    if level == "segment":
        plt.xticks(rotation=45, ha="right")
        for p in ax.patches:
            height = p.get_height()
            if height > 0:
                ax.text(
                    p.get_x() + p.get_width() / 2,
                    height * 1.1,
                    f"{int(height)}",
                    ha="center",
                    va="bottom",
                    fontsize=7,
                    color=p.get_facecolor(),
                    weight="bold",
                )
    else:
        ax.set_xticklabels([])
        ax.set_xlabel("")

    correct_counts = counts.get(True, pd.Series(dtype=int)).sort_values(ascending=False)
    top5 = correct_counts.head(5)
    bottom5 = correct_counts.tail(5)

    textstr = "Top 5 Correctly Classified:\n"
    for idx, val in top5.items():
        textstr += f"{idx}: {val}\n"

    textstr += "\nLowest 5 Correctly Classified:\n"
    for idx, val in bottom5.items():
        textstr += f"{idx}: {val}\n"

    props = dict(boxstyle="round", facecolor="white", alpha=0.8)
    ax.text(
        1.05,
        0.5,
        textstr,
        transform=ax.transAxes,
        fontsize=9,
        verticalalignment="center",
        bbox=props,
        color="black"
    )

    ymin, ymax = ax.get_ylim()
    ax.set_ylim(ymin, ymax * 1.2)

    plt.tight_layout()
    plt.savefig(img_path)
    plt.close()

def get_confidence_level(confidence_rates: List[float]) -> List[str]:
    confidence_levels = []
    for rate in confidence_rates:
        rate *= 100
        if 0 <= rate <= 50:
            confidence_levels.append("Low")
        elif 50 < rate <= 75:
            confidence_levels.append("Medium")
        elif 75 < rate <= 100:
            confidence_levels.append("High")
    
    return confidence_levels