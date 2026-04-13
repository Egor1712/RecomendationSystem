import os
from sklearn.preprocessing import LabelEncoder

from scipy.sparse import csr_matrix
import joblib
import pandas as pd
import numpy as np


def save_model(model, path):
    joblib.dump(model, path)


def load_model(path):
    return joblib.load(path)


def save_preprocessor(obj, path):
    joblib.dump(obj, path)


def load_preprocessor(path):
    return joblib.load(path)


def create_user_item_matrix(transactions_df, user_ids, item_ids):
    """Создаёт разреженную матрицу взаимодействий (пользователи x товары)."""
    user_index = {uid: i for i, uid in enumerate(user_ids)}
    item_index = {iid: j for j, iid in enumerate(item_ids)}
    rows = [user_index[uid] for uid in transactions_df['customer_id']]
    cols = [item_index[iid] for iid in transactions_df['article_id']]
    data = np.ones(len(rows))
    return csr_matrix((data, (rows, cols)), shape=(len(user_ids), len(item_ids)))


def reduce_mem_usage(df):
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    start_mem = df.memory_usage().sum() / 1024 ** 2
    for col in df.columns:
        col_type = df[col].dtypes
        if col_type in numerics:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                else:
                    df[col] = df[col].astype(np.int64)
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
    end_mem = df.memory_usage().sum() / 1024 ** 2
    print(f'Memory: {start_mem:.2f} Mb -> {end_mem:.2f} Mb ({100 * (start_mem - end_mem) / start_mem:.1f}% reduction)')
    return df


def mapk(actual, predicted, k=12):
    """Mean Average Precision @ k"""

    def apk(a, p, k):
        if len(p) > k:
            p = p[:k]
        score = 0.0
        hits = 0.0
        for i, pred in enumerate(p):
            if pred in a and pred not in p[:i]:
                hits += 1.0
                score += hits / (i + 1.0)
        return score / min(len(a), k) if a else 0.0

    return np.mean([apk(a, p, k) for a, p in zip(actual, predicted)])


def load_data(data_dir, encoder_output_dir):
    print("Загрузка данных...")
    transactions = reduce_mem_usage(pd.read_csv(os.path.join(data_dir, "transactions_train.csv")))
    articles = reduce_mem_usage(pd.read_csv(os.path.join(data_dir, "articles.csv")))
    customers = reduce_mem_usage(pd.read_csv(os.path.join(data_dir, "customers.csv")))
    sample_sub = reduce_mem_usage(pd.read_csv(os.path.join(data_dir, "sample_submission.csv")))

    le = LabelEncoder()
    all_customers = pd.concat(
        [transactions['customer_id'], customers['customer_id'], sample_sub['customer_id']]).unique()
    le.fit(all_customers)
    transactions['customer_id'] = le.transform(transactions['customer_id']).astype('int32')
    customers['customer_id'] = le.transform(customers['customer_id']).astype('int32')
    sample_sub['customer_id'] = le.transform(sample_sub['customer_id']).astype('int32')
    joblib.dump(le, os.path.join(encoder_output_dir, "customer_id_label_encoder.joblib"))

    transactions['t_dat'] = pd.to_datetime(transactions['t_dat'])
    transactions['day'] = (transactions['t_dat'].max() - transactions['t_dat']).dt.days.astype('int16')
    transactions['week'] = (transactions['day'] // 7).astype('int8')

    return transactions, articles, customers, sample_sub
