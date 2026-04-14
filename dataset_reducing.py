import glob
from pathlib import Path

import pandas as pd
import numpy as np
import os

import consts


def reduce_hm_rows(input_dir, output_dir, sample_frac=0.1, random_state=42):
    os.makedirs(output_dir, exist_ok=True)

    print("1. Выбор подмножества пользователей...")
    cust_path = os.path.join(input_dir, 'customers.csv')
    trans_path = os.path.join(input_dir, 'transactions_train.csv')
    sub_path = os.path.join(input_dir, 'sample_submission.csv')
    customers = pd.read_csv(cust_path)
    all_cust_ids = customers['customer_id'].unique()

    rng = np.random.default_rng(random_state)
    selected_cust_ids = rng.choice(all_cust_ids,
                                   size=int(len(all_cust_ids) * sample_frac),
                                   replace=False)
    selected_cust_ids_set = set(selected_cust_ids)
    print(f"   Выбрано {len(selected_cust_ids)} пользователей из {len(all_cust_ids)}.")

    print("\n2. Фильтрация customers.csv...")
    cust_filtered = customers[customers['customer_id'].isin(selected_cust_ids_set)]
    cust_filtered.to_csv(os.path.join(output_dir, 'customers.csv'), index=False)
    print(f"   Сохранено строк: {len(cust_filtered)}")
    del customers, cust_filtered

    print("\n3. Фильтрация transactions_train.csv...")
    output_trans_path = os.path.join(output_dir, 'transactions_train.csv')
    output_sub_path = os.path.join(output_dir, 'sample_submission.csv')

    transaction_df = pd.read_csv(trans_path)
    filtered_df = transaction_df[transaction_df['customer_id'].isin(selected_cust_ids_set)]
    filtered_df.to_csv(output_trans_path, index=False)
    sample_df = pd.read_csv(sub_path)
    filtered_df = sample_df[sample_df['customer_id'].isin(selected_cust_ids_set)]
    filtered_df.to_csv(output_sub_path, index=False)

    print(f"Отфильтрованные транзакции сохранены в {output_trans_path}")

    art_path = os.path.join(input_dir, 'articles.csv')
    print("4. Сохранение articles.csv (все строки)...")
    articles = pd.read_csv(art_path)

    available_article_ids = articles['article_id'].unique()
    article_filtered = articles[articles['article_id'].isin(available_article_ids)]
    article_filtered.to_csv(os.path.join(output_dir, 'articles.csv'), index=False)
    print(f"Сохранено строк: {len(articles)}")
    images_path = os.path.join(input_dir, 'images')
    new_images_path = os.path.join(output_dir, 'images')
    os.makedirs(new_images_path, exist_ok=True)
    for path in glob.glob(f'{images_path}/**/*.jpg', recursive=True):
        file = Path(path)
        if int(file.stem[1:]) in available_article_ids:
            os.replace(path, os.path.join(new_images_path, file.name))

    del articles, article_filtered


if __name__ == "__main__":
    SAMPLE_FRACTION = 0.1

    reduce_hm_rows(
        input_dir=consts.DATASET_DIRECTORY,
        output_dir=consts.WORKING_DATASET_DIRECTORY,
        sample_frac=SAMPLE_FRACTION
    )