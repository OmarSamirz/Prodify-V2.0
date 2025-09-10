import pandas as pd
import numpy as np
import re, warnings, os
from typing import List, Optional, Dict
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder as SklearnLabelEncoder
from sklearn.metrics import f1_score, accuracy_score
from sklearn.svm import LinearSVC
from constants import JIO_MART_DATASET_MAPPED, TD_DB
warnings.filterwarnings('ignore')
from teradataml import *
from teradataml.dataframe.copy_to import copy_to_sql

from modules.db import TeradataDatabase
hierarchy = ['segment','family','class','brick']

td_db = TeradataDatabase() 
td_db.connect()

TRAIN_FRAC = 0.70
VAL_FRAC = 0.20
TEST_FRAC = 0.10
RANDOM_SEED = 42

EXCLUDE_SOURCES_FOR_BRICK_TRAIN = {'MWPD_FULL'}

def vec_builder():
    return TfidfVectorizer(
        max_features=15000,
        ngram_range=(1,2),
        min_df=2,
        max_df=0.95,
        stop_words='english',
        sublinear_tf=True,
        norm='l2'
    )

def model_builder():
    return LinearSVC(C=1.0, class_weight='balanced')

def train_single_level(X_train, y_series):
    le = SklearnLabelEncoder()
    y = y_series.astype(str).fillna("NA_LBL")
    y_enc = le.fit_transform(y)
    if len(np.unique(y_enc)) < 2:
        model = ('const', int(y_enc[0]))
    else:
        clf = model_builder()
        clf.fit(X_train, y_enc)
        model = ('svm', clf)
    return model, le

def predict_single_level(model_tuple, le, X):
    kind, obj = model_tuple
    if kind == 'const':
        y_pred_enc = np.full(X.shape[0], obj, dtype=int)
    else:
        y_pred_enc = obj.predict(X)
    y_pred = pd.Series(le.inverse_transform(y_pred_enc)).replace("NA_LBL", np.nan)
    return y_pred

def eval_metrics(y_true, y_pred) -> Dict[str, float]:
    yt = y_true.astype(str)
    yp = y_pred.astype(str)
    return {
        'accuracy': accuracy_score(yt, yp),
        'f1_macro': f1_score(yt, yp, average='macro', zero_division=0),
        'f1_weighted': f1_score(yt, yp, average='weighted', zero_division=0),
        'n': len(yt)
    }

def split_by_key(df_all: pd.DataFrame, seed=RANDOM_SEED) -> pd.DataFrame:
    keys = df_all['dedup_key'].unique().tolist()
    rng = np.random.RandomState(seed)
    perm = rng.permutation(len(keys))
    keys = np.array(keys)[perm]
    n = len(keys)
    n_train = int(round(TRAIN_FRAC * n))
    n_val = int(round(VAL_FRAC * n))
    n_train = max(1, min(n_train, n - 2))
    n_val   = max(1, min(n_val, n - n_train - 1))
    train_keys = set(keys[:n_train])
    val_keys   = set(keys[n_train:n_train+n_val])
    test_keys  = set(keys[n_train+n_val:])
    df_all['split'] = np.where(df_all['dedup_key'].isin(train_keys), 'train',
                        np.where(df_all['dedup_key'].isin(val_keys), 'val', 'test'))
    assert train_keys.isdisjoint(val_keys) and train_keys.isdisjoint(test_keys) and val_keys.isdisjoint(test_keys)
    return df_all

def clean_products_in_db(): 

    tdf = DataFrame.from_table("full_dataset", schema_name=TD_DB)
    
    cleaning_query = """
    UPDATE demo_user.full_dataset
    SET product_name =
                    TRIM(
                        REGEXP_REPLACE(
                        REGEXP_REPLACE(
                            REGEXP_REPLACE(product_name, '[[:digit:]]+', ''), 
                            '[-_/\\|]', ''),                              
                        '[[:punct:]]', ''                              
                        )
                    )
                    ;
    """

    td_db.execute_query(cleaning_query)
    
def load_data_set_from_db():
    clean_products_in_db()
    tdf = td_db.execute_query("Select * from demo_user.full_dataset")
    df = pd.DataFrame(tdf)
    return df

    
def run():
    df_all = load_data_set_from_db()
    df_all = split_by_key(df_all, seed=RANDOM_SEED)
    print("\nSplit sizes (rows):")
    print(df_all['split'].value_counts(dropna=False).to_string())
    vec = vec_builder()
    X_train_texts = df_all.loc[df_all['split']=='train','text'].tolist()
    vec.fit(X_train_texts)
    models: Dict[str, tuple] = {}
    encoders: Dict[str, SklearnLabelEncoder] = {}
    val_metrics_all: Dict[str, Dict[str,float]] = {}
    test_metrics_all: Dict[str, Dict[str,float]] = {}
    for layer in hierarchy:
        tr = df_all[(df_all['split']=='train') & (~df_all[layer].isna())]
        if layer == 'brick':
            tr = tr[~tr['source'].isin(EXCLUDE_SOURCES_FOR_BRICK_TRAIN)]
        va = df_all[(df_all['split']=='val')   & (~df_all[layer].isna())]
        te = df_all[(df_all['split']=='test')  & (~df_all[layer].isna())]

        print(f"\n[{layer.upper()}] Train/Val/Test sizes (rows): {len(tr):,} / {len(va):,} / {len(te):,}")

        X_tr = vec.transform(tr['text'])
        X_va = vec.transform(va['text'])
        X_te = vec.transform(te['text'])

        model, le = train_single_level(X_tr, tr[layer])
        models[layer] = model
        encoders[layer] = le
        y_va_pred = predict_single_level(model, le, X_va)
        y_te_pred = predict_single_level(model, le, X_te)
        val_metrics = eval_metrics(va[layer], y_va_pred)
        test_metrics = eval_metrics(te[layer], y_te_pred)
        val_metrics_all[layer] = val_metrics
        test_metrics_all[layer] = test_metrics
        print(f"  Val  | n={val_metrics['n']:5d} | Acc={val_metrics['accuracy']:.4f}  | F1_weighted={val_metrics['f1_weighted']:.4f}")
        print(f"  Test | n={test_metrics['n']:5d} | Acc={test_metrics['accuracy']:.4f} | F1_weighted={test_metrics['f1_weighted']:.4f}")
    def avg(metrics: Dict[str, Dict[str,float]]):
        acc = np.mean([metrics[l]['accuracy'] for l in hierarchy])
        f1m = np.mean([metrics[l]['f1_macro'] for l in hierarchy])
        f1w = np.mean([metrics[l]['f1_weighted'] for l in hierarchy])
        return acc, f1m, f1w
    v_acc, v_f1m, v_f1w = avg(val_metrics_all)
    t_acc, t_f1m, t_f1w = avg(test_metrics_all)
    print("\nAverage metrics across layers:")
    print(f"  Validation | Acc={v_acc:.4f} | F1_weighted={v_f1w:.4f}")
    print(f"  Test       | Acc={t_acc:.4f} | F1_weighted={t_f1w:.4f}")
    te_full = df_all[df_all['split']=='test'].copy().dropna(subset=hierarchy)
    if len(te_full):
        X_te_full = vec.transform(te_full['text'])
        preds = {}
        for layer in hierarchy:
            preds[layer] = predict_single_level(models[layer], encoders[layer], X_te_full).values
        preds_df = pd.DataFrame(preds)
        y_true_df = te_full[hierarchy].reset_index(drop=True)
        def build_prediction_output_df(item_name_series, y_true_df, y_pred_df, item_name_col='item_name'):
            out_cols = {item_name_col: item_name_series}
            level_correct_cols = []
            for lvl in hierarchy:
                truth_col = f'{lvl}_truth'
                pred_col = f'{lvl}_pred'
                corr_col = f'{lvl}_correct'
                out_cols[truth_col] = y_true_df[lvl]
                out_cols[pred_col] = y_pred_df[lvl]
                correct = (y_true_df[lvl].astype(str) == y_pred_df[lvl].astype(str)).astype(int)
                out_cols[corr_col] = correct
                level_correct_cols.append(corr_col)
            df_out = pd.DataFrame(out_cols)
            df_out['all_levels_correct'] = (df_out[level_correct_cols].sum(axis=1) == len(hierarchy)).astype(int)
            return df_out
        item_name_series = te_full['product_name'].reset_index(drop=True)
        prediction_output_df = build_prediction_output_df(
            item_name_series=item_name_series,
            y_true_df=y_true_df,
            y_pred_df=preds_df.reset_index(drop=True),
            item_name_col='item_name'
        )
        os.makedirs('data', exist_ok=True)
        prediction_output_df.to_csv('data/prediction_output.csv', index=False)
        print("\nSaved detailed test predictions to data/prediction_output.csv")
    else:
        print("\n[INFO] No complete test rows with all labels to export detailed predictions.")

if __name__ == "__main__":
    run()

