import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

from typing import List

from constants import RANDOM_STATE

def split_dataset(dataset_path: str, train_dataset_path: str, test_dataset_path: str):
    df = pd.read_csv(dataset_path)
    df.dropna(subset=["class"], inplace=True)
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=RANDOM_STATE)

    train_df.to_csv(train_dataset_path, index=False)
    test_df.to_csv(test_dataset_path, index=False)

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