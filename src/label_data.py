import pandas as pd
from tqdm.auto import tqdm

from gpc_agent import GpcAgent
from utils import evaluation_score
from constants import LABELED_GPC_PATH, LABELED_GPC_PRED_PATH


def label_products(df: pd.DataFrame, agent: GpcAgent, product_col: str):
    segment_lst = []
    family_lst = []
    class_lst = []
    brick_lst = []
    for _, row in tqdm(df.iterrows(), total=len(df)):
        result = agent.classify(row[product_col])
        segment_lst.append(result["segment"])
        family_lst.append(result["family"])
        class_lst.append(result["class"])
        brick_lst.append(result["brick"])

    df["segment_pred"] = segment_lst
    df["family_pred"] = family_lst
    df["class_pred"] = class_lst
    df["brick_pred"] = brick_lst

    return df

def main():
    agent = GpcAgent()

    df_sample = pd.read_csv(LABELED_GPC_PATH)
    df_labeled = label_products(df_sample, agent, "translated_name")

    df_labeled.to_csv(LABELED_GPC_PRED_PATH, index=False, encoding="utf-8-sig")

if __name__ == "__main__":
    main()