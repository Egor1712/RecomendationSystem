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

    # 3. Фильтруем transactions_train.csv чанками
    print("\n3. Фильтрация transactions_train.csv...")
    output_trans_path = os.path.join(output_dir, 'transactions_train.csv')
    output_sub_path = os.path.join(output_dir, 'sample_submission.csv')

    # Используем tqdm для отображения прогресса (примерное количество чанков можно оценить)
    # Для этого узнаем общее число строк (быстро через wc -l или приблизительно)
    # Здесь для простоты прогресс-бар будет без общего количества, только по чанкам.
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
    articles.to_csv(os.path.join(output_dir, 'articles.csv'), index=False)
    print(f"Сохранено строк: {len(articles)}")
    del articles


if __name__ == "__main__":
    SAMPLE_FRACTION = 0.1

    reduce_hm_rows(
        input_dir=consts.DATASET_DIRECTORY,
        output_dir=consts.WORKING_DATASET_DIRECTORY,
        sample_frac=SAMPLE_FRACTION
    )