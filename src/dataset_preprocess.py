import os.path

import joblib
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.preprocessing import LabelEncoder

from src.utils import save_preprocessor


def make_candidate_df(transactions, articles, target_df, val_start_date = '2020-01-01', train=False):
    if train:
        top300 = target_df.groupby('article_id')['customer_id'].nunique().sort_values(ascending=False).head(300).index
    else:
        top300 = transactions.query(f"t_dat >= '{val_start_date}'").groupby('article_id')[
            'customer_id'].nunique().sort_values(ascending=False).head(300).index

    if train:
        customers_list = target_df['customer_id'].unique()
    else:
        customers_list = target_df['customer_id'].unique()
    # Создаём все комбинации с топ-300 товарами
    df = pd.DataFrame({'customer_id': customers_list})
    df['tmp'] = 0
    tmp2 = pd.DataFrame({'article_id': top300, 'tmp': 0})
    df = pd.merge(df, tmp2, on='tmp', how='outer').drop('tmp', axis=1)

    # Шаг 2: товары с тем же product_code, что уже купленные пользователем
    user_products = transactions[['customer_id', 'article_id']].drop_duplicates()
    user_products = user_products[user_products['customer_id'].isin(customers_list)]
    user_products = pd.merge(user_products, articles[['article_id', 'product_code']], on='article_id', how='left')
    user_products = user_products.drop_duplicates(['customer_id', 'product_code'])[['customer_id', 'product_code']]

    # Все товары, имеющие такие product_code
    articles_with_codes = articles[articles['product_code'].isin(user_products['product_code'].unique())][
        ['article_id', 'product_code']]
    extra = pd.merge(user_products, articles_with_codes, on='product_code', how='outer')[['customer_id', 'article_id']]
    extra = extra.dropna(subset=['customer_id', 'article_id']).astype({'customer_id': 'int32', 'article_id': 'int32'})

    # Объединяем и убираем дубликаты
    df = pd.concat([df, extra]).drop_duplicates(['customer_id', 'article_id']).reset_index(drop=True)

    if train:
        # Добавляем метку target (1 если покупка в целевой период)
        target_pairs = target_df[['customer_id', 'article_id']].drop_duplicates()
        target_pairs['target'] = 1
        df = pd.merge(df, target_pairs, on=['customer_id', 'article_id'], how='left')
        df['target'] = df['target'].fillna(0).astype('int8')
    else:
        df['target'] = 0

    return df

def add_article_features(df, articles):
    article_cols = ['article_id', 'product_code', 'product_type_no', 'graphical_appearance_no',
                    'colour_group_code', 'index_group_no', 'section_no', 'garment_group_no']
    articles_sub = articles[article_cols].copy()
    df = pd.merge(df, articles_sub, on='article_id', how='left')
    return df


def add_customer_features(df, customers, output_dir):
    customers['age'] = customers['age'].fillna(customers['age'].median()).astype(int)
    df = pd.merge(df, customers[['customer_id', 'age', 'club_member_status']], on='customer_id', how='left')
    df['age'] = pd.cut(df['age'], bins=[0, 18, 25, 35, 45, 55, 65, 120],
                                  labels=[0, 1, 2, 3, 4, 5, 6]).astype(int)
    le = LabelEncoder()
    df['club_member_status'] = le.fit_transform(df['club_member_status'])
    save_preprocessor(le, os.path.join(output_dir, 'club_code_encoder.pkl'))
    return df


def preprocess_customers(customers, output_dir):
    customers = customers[['customer_id', 'age', 'club_member_status']].copy()
    customers['age'] = customers['age'].fillna(customers['age'].median()).astype(int)
    customers['age_bin'] = pd.cut(customers['age'], bins=[0, 18, 25, 35, 45, 55, 65, 120],
                                  labels=[0, 1, 2, 3, 4, 5, 6]).astype(int)
    customers['club_member_status'] = customers['club_member_status'].fillna('unknown')
    le = LabelEncoder()
    customers['club_code'] = le.fit_transform(customers['club_member_status'])
    save_preprocessor(le, os.path.join(output_dir, 'club_encoder.pkl'))
    return customers[['customer_id', 'age_bin', 'club_code']], ['age_bin', 'club_code']



def repeat_features(transactions, df):
    # Повторы по паре customer, article
    rep = transactions.drop_duplicates(['customer_id', 'article_id', 't_dat']).groupby(
        ['customer_id', 'article_id']).size().reset_index(name='repeat_count')

    # По пользователю
    user_stats = rep.groupby('customer_id')['repeat_count'].agg(['mean', 'max', 'std', 'count', 'sum']).reset_index()
    user_stats.columns = ['customer_id', 'mean_repeat_by_cust', 'max_repeat_by_cust', 'std_repeat_by_cust',
                          'unique_items_by_cust', 'total_purchases_by_cust']
    user_stats['repeat_rate_by_cust'] = (user_stats['total_purchases_by_cust'] - user_stats['unique_items_by_cust']) / \
                                        user_stats['total_purchases_by_cust']
    # По товару
    item_stats = rep.groupby('article_id')['repeat_count'].agg(['mean', 'max', 'std', 'count', 'sum']).reset_index()
    item_stats.columns = ['article_id', 'mean_repeat_by_item', 'max_repeat_by_item', 'std_repeat_by_item',
                          'unique_customers_by_item', 'total_sales_by_item']
    item_stats['repeat_rate_by_item'] = (item_stats['total_sales_by_item'] - item_stats['unique_customers_by_item']) / \
                                        item_stats['total_sales_by_item']

    df = pd.merge(df, user_stats, on='customer_id', how='left')
    df = pd.merge(df, item_stats, on='article_id', how='left')
    # Заполняем NaN
    for col in df.columns:
        if 'repeat' in col or 'unique' in col or 'total' in col:
            df[col] = df[col].fillna(0)
    return df


def last_purchase_date(transactions, df):
    for col in ['article_id', 'product_code', 'product_type_no', 'graphical_appearance_no',
                'colour_group_code', 'index_group_no', 'section_no', 'garment_group_no']:
        if col not in df.columns or col not in transactions.columns:
            continue
        tmp = transactions.groupby(['customer_id', col])['day'].min().reset_index().rename(
            columns={'day': f'last_{col}'})
        df = pd.merge(df, tmp, on=['customer_id', col], how='left')
        df[f'last_{col}'] = df[f'last_{col}'].fillna(9999).astype('int16')
    return df


def weekly_count_features(transactions, df, weeks=5):
    for col in ['article_id']:
        if col not in df.columns:
            continue
        for w in range(weeks):
            cnt = transactions[transactions['week'] == w].groupby(col)['customer_id'].nunique().reset_index().rename(
                columns={'customer_id': f'{col}_week{w}_count'})
            df = pd.merge(df, cnt, on=col, how='left')
            df[f'{col}_week{w}_count'] = df[f'{col}_week{w}_count'].fillna(0).astype('int16')
        # изменение между неделями
        df[f'{col}_week0_ratio'] = (df[f'{col}_week0_count'] / (df[f'{col}_week1_count'] + 1e-5)).astype('float32')
    return df


def daily_count_features(transactions, df, days=7):
    for col in ['article_id']:
        if col not in df.columns:
            continue
        for d in range(days):
            cnt = transactions[transactions['day'] == d].groupby(col)['customer_id'].nunique().reset_index().rename(
                columns={'customer_id': f'{col}_day{d}_count'})
            df = pd.merge(df, cnt, on=col, how='left')
            df[f'{col}_day{d}_count'] = df[f'{col}_day{d}_count'].fillna(0).astype('int16')
        df[f'{col}_day0_ratio'] = (df[f'{col}_day0_count'] / (df[f'{col}_day1_count'] + 1e-5)).astype('float32')
    return df


def price_features(transactions, df):
    user_price = transactions.groupby('customer_id')['price'].agg(['max', 'mean']).reset_index()
    user_price.columns = ['customer_id', 'max_price_by_cust', 'mean_price_by_cust']
    df = pd.merge(df, user_price, on='customer_id', how='left')
    # средняя цена товара за последние 4 недели
    recent = transactions[transactions['week'] < 4].groupby('article_id')['price'].mean().reset_index().rename(
        columns={'price': 'mean_price_4w'})
    df = pd.merge(df, recent, on='article_id', how='left')
    df['higher_than_max'] = (df['mean_price_4w'] > df['max_price_by_cust']).astype('int8')
    df['higher_than_mean'] = (df['mean_price_4w'] > df['mean_price_by_cust']).astype('int8')
    df[['max_price_by_cust', 'mean_price_by_cust', 'mean_price_4w']] = df[
        ['max_price_by_cust', 'mean_price_by_cust', 'mean_price_4w']].fillna(0).astype('float32')
    return df


def channel_features(transactions, df):
    for w in [999, 3, 0]:
        sub = transactions[transactions['week'] <= w] if w != 999 else transactions
        # по пользователю
        user_ch = sub.groupby('customer_id')['sales_channel_id'].agg(['mean', 'max']).reset_index()
        user_ch.columns = ['customer_id', f'mean_channel_{w}w_cust', f'max_channel_{w}w_cust']
        df = pd.merge(df, user_ch, on='customer_id', how='left')
        # по товару
        item_ch = sub.groupby('article_id')['sales_channel_id'].agg(['mean', 'max']).reset_index()
        item_ch.columns = ['article_id', f'mean_channel_{w}w_item', f'max_channel_{w}w_item']
        df = pd.merge(df, item_ch, on='article_id', how='left')
        # заполнение
        for col in [f'mean_channel_{w}w_cust', f'max_channel_{w}w_cust', f'mean_channel_{w}w_item',
                    f'max_channel_{w}w_item']:
            df[col] = df[col].fillna(0)
    return df



def preprocess_articles(articles, output_dir):
    cat_cols = ['product_type_name', 'graphical_appearance_name',
                'colour_group_name', 'perceived_colour_value_name',
                'department_name', 'index_name', 'section_name']
    articles = articles[['article_id'] + cat_cols].copy()
    for col in cat_cols:
        articles[col] = articles[col].fillna('unknown')
    encoders = {}
    for col in cat_cols:
        le = LabelEncoder()
        articles[f'{col}_code'] = le.fit_transform(articles[col])
        encoders[col] = le
    feature_cols = [f'{col}_code' for col in cat_cols]
    save_preprocessor(encoders, os.path.join(output_dir, 'article_encoders.pkl'))
    return articles[['article_id'] + feature_cols], feature_cols



def build_dataset_with_matrix(articles, customers, transactions, output_dir):
    popular_items = transactions['article_id'].value_counts().index
    active_users = transactions['customer_id'].value_counts().index
    transactions = transactions[transactions['article_id'].isin(popular_items) &
                                transactions['customer_id'].isin(active_users)]

    user_ids = transactions['customer_id'].unique()
    item_ids = transactions['article_id'].unique()
    user_to_idx = {uid: i for i, uid in enumerate(user_ids)}
    item_to_idx = {iid: j for j, iid in enumerate(item_ids)}

    # Матрица взаимодействий
    rows = [user_to_idx[uid] for uid in transactions['customer_id']]
    cols = [item_to_idx[iid] for iid in transactions['article_id']]
    data = np.ones(len(rows), dtype=np.float32)  # вместо np.ones(len(rows))
    interaction = csr_matrix((data, (rows, cols)), shape=(len(user_ids), len(item_ids)), dtype=np.float32)

    # Признаки пользователей
    users_df, user_feat_cols = preprocess_customers(customers, output_dir)
    users_df = users_df[users_df['customer_id'].isin(user_ids)]
    users_df = users_df.set_index('customer_id').reindex(user_ids).fillna(0)
    user_features = csr_matrix(users_df[user_feat_cols].values.astype(np.float32), dtype=np.float32)

    # Признаки товаров
    items_df, item_feat_cols = preprocess_articles(articles, output_dir)
    items_df = items_df[items_df['article_id'].isin(item_ids)]
    items_df = items_df.set_index('article_id').reindex(item_ids).fillna(0)
    item_features = csr_matrix(items_df[item_feat_cols].values.astype(np.float32), dtype=np.float32)

    joblib.dump({'user_to_idx': user_to_idx, 'item_to_idx': item_to_idx,
                 'user_ids': user_ids, 'item_ids': item_ids},
                os.path.join(output_dir, 'mappings.pkl'))

    return interaction, user_features, item_features, user_to_idx, item_to_idx


def build_full_dataset(transactions, articles, customers, target_df, train=False, output_dir='./output'):
    print("Формирование пар (customer, article)...")
    df_copy = make_candidate_df(transactions, articles, target_df, train=train)
    print("Добавление признаков товаров...")
    df_copy = add_article_features(df_copy, articles)
    print("Добавление признаков пользователей...")
    df_copy = add_customer_features(df_copy, customers, output_dir)
    print("Статистика повторных покупок...")
    df_copy = repeat_features(transactions, df_copy)
    print("Дата последней покупки...")
    df_copy = last_purchase_date(transactions, df_copy)
    print("Недельные популярности...")
    df_copy = weekly_count_features(transactions, df_copy)
    print("Дневные популярности...")
    df_copy = daily_count_features(transactions, df_copy)
    print("Ценовые признаки...")
    df_copy = price_features(transactions, df_copy)
    print("Признаки каналов...")
    df_copy = channel_features(transactions, df_copy)

    drop_cols = ['customer_id', 'article_id', 'target']
    if train:
        drop_cols.append('target')
    feature_cols = [c for c in df_copy.columns if c not in drop_cols]
    return df_copy, feature_cols



