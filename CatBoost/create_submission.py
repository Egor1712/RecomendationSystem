import os
import pickle

import joblib
import numpy as np
import warnings
import sys

import consts
from CatBoost.train_cat_boost import CFG
from src.dataset_preprocess import build_full_dataset
from src.utils import load_data

sys.path.append(os.path.abspath("../"))

warnings.filterwarnings('ignore')


def inference(test_df, feature_cols):
    preds = np.zeros(len(test_df))
    for fold in CFG.used_fold:
        model_path = os.path.join(CFG.output_dir, CFG.exp, f'model_fold{fold}.pkl')
        if not os.path.exists(model_path):
            print(f"Модель {model_path} не найдена")
            continue
        model = joblib.load(model_path)
        preds += model.predict_proba(test_df[feature_cols])[:, 1]
    preds /= len(CFG.used_fold)
    return preds


if __name__ == "__main__":
    transactions, articles, customers, sample_sub, _ = load_data(consts.WORKING_DATASET_DIRECTORY, CFG.output_dir)
    with open(os.path.join(CFG.output_dir, CFG.exp, 'df.pickle'), 'rb') as f:
        df_train = pickle.load(f)
    feature_cols = [c for c in df_train.columns if c not in ['customer_id', 'article_id', 'target']]
    # Подготавливаем тестовый датасет (нужно загрузить target_df = sample_sub)
    print("Построение тестового датасета...")
    test_df, _ = build_full_dataset(transactions, articles, customers, sample_sub)
    # Предсказания
    test_df['pred'] = inference(test_df, feature_cols)
    # Формируем submission
    sub = test_df.sort_values('pred', ascending=False).groupby('customer_id')['article_id'].apply(
        list).reset_index()
    sub['prediction'] = sub['article_id'].apply(lambda x: ' '.join(map(str, x[:12])))
    # Декодируем customer_id
    le = joblib.load(os.path.join(CFG.output_dir, "customer_id_label_encoder.joblib"))
    sub['customer_id'] = le.inverse_transform(sub['customer_id'])
    sub[['customer_id', 'prediction']].to_csv(os.path.join(CFG.output_dir, CFG.exp, 'submission.csv'), index=False)
    print("Сабмит сохранён.")
