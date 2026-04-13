import os
import pickle
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from catboost import CatBoostClassifier, Pool
import warnings
import sys

from src.dataset_preprocess import build_full_dataset
from src.utils import reduce_mem_usage, load_data, mapk

sys.path.append(os.path.abspath("../"))
import consts

warnings.filterwarnings('ignore')

class CFG:
    train = True
    output_dir = "./output"
    exp = "1"
    seed = 42
    fold = 5
    used_fold = [0, 1, 2, 3, 4]
    val_start_date = '2020-09-16'
    catboost_params = {
        'loss_function': 'Logloss',
        'learning_rate': 0.02,
        'max_depth': 6,
        'random_seed': seed,
        'thread_count': 4,
        'task_type': 'GPU',
        'scale_pos_weight': 100,
        'iterations': 5000,
        'early_stopping_rounds': 100,
        'verbose': 200,
        'fold_permutation_block': 32,
        'gpu_cat_features_storage': 'CpuPinnedMemory',
        "train_dir": "./output"
    }


os.makedirs(CFG.output_dir, exist_ok=True)
os.makedirs(os.path.join(CFG.output_dir, CFG.exp), exist_ok=True)


def train_model(df, feature_cols):
    print("Обучение CatBoost с кросс-валидацией...")
    # Разбиение по пользователям
    unique_customers = df['customer_id'].unique()
    kf = KFold(n_splits=CFG.fold, shuffle=True, random_state=CFG.seed)
    fold_assign = np.zeros(len(df), dtype=int)
    for i, (_, val_idx) in enumerate(kf.split(unique_customers)):
        val_cust = unique_customers[val_idx]
        fold_assign[df['customer_id'].isin(val_cust)] = i

    oof_pred = np.zeros(len(df))
    scores = []
    for fold in range(CFG.fold):
        if fold not in CFG.used_fold:
            continue
        print(f"Fold {fold}")
        tr_idx = fold_assign != fold
        va_idx = fold_assign == fold
        X_train = df.loc[tr_idx, feature_cols]
        y_train = df.loc[tr_idx, 'target']
        X_val = df.loc[va_idx, feature_cols]
        y_val = df.loc[va_idx, 'target']

        # Категориальные признаки (укажем колонки, которые являются категориями)
        cat_cols = [c for c in feature_cols if c in ['product_code', 'product_type_no', 'graphical_appearance_no',
                                                     'colour_group_code', 'index_group_no', 'section_no',
                                                     'garment_group_no']]
        train_pool = Pool(X_train, y_train, cat_features=cat_cols)
        val_pool = Pool(X_val, y_val, cat_features=cat_cols)

        model = CatBoostClassifier(**CFG.catboost_params)
        model.fit(train_pool, eval_set=val_pool, use_best_model=True, verbose=True)
        joblib.dump(model, os.path.join(CFG.output_dir, CFG.exp, f'model_fold{fold}.pkl'))

        val_pred = model.predict_proba(X_val)[:, 1]
        oof_pred[va_idx] = val_pred
        # Считаем MAP@12 на валидации
        val_df = df.loc[va_idx, ['customer_id', 'article_id', 'target']].copy()
        val_df['pred'] = val_pred
        val_group = val_df.sort_values('pred', ascending=False).groupby('customer_id')['article_id'].apply(
            list).reset_index()
        target_group = target_df.groupby('customer_id')['article_id'].apply(list).reset_index()
        val_group = pd.merge(val_group, target_group, on='customer_id', how='left')
        score = mapk(val_group['article_id_y'], val_group['article_id_x'], k=12)
        scores.append(score)
        print(f"Fold {fold} MAP@12 = {score:.6f}")
    print(f"\nСредний MAP@12 по фолдам: {np.mean(scores):.6f}")
    return oof_pred


if __name__ == "__main__":
    transactions, articles, customers, sample_sub = load_data(consts.WORKING_DATASET_DIRECTORY, CFG.output_dir)
    target_df = transactions[transactions['t_dat'] >= CFG.val_start_date].reset_index(drop=True)
    transactions = transactions[transactions['t_dat'] < CFG.val_start_date].reset_index(drop=True)
    df, feature_cols = build_full_dataset(transactions, articles, customers, target_df, train=True)
    oof = train_model(df, feature_cols)
    with open(os.path.join(CFG.output_dir, CFG.exp, 'df.pickle'), 'wb') as f:
        pickle.dump(df, f)
    print("Обучение завершено.")