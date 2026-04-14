import sys

import pandas as pd

import consts
from src.inference import HmRecommender
import joblib
import os
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed


sys.path.append(os.path.abspath("../"))
sys.path.append(os.path.abspath("../../"))
sys.path.append(os.path.abspath("../../../"))
from src.utils import load_data
from src.dataset_preprocess import build_dataset_with_matrix


def get_popular_items(transactions, top_k=100):
    pop = transactions['article_id'].value_counts().head(top_k).index.tolist()
    return pop


def predict_for_user(user_id, recommender, top_k, popular_items):
    try:
        recs = recommender.recommend(user_id, top_k=top_k)
        article_ids = [str(rec[0]) for rec in recs]
        return user_id, ' '.join(article_ids), None
    except Exception as e:
        fallback = [str(pid) for pid in popular_items[:top_k]]
        return user_id, ' '.join(fallback), str(e)


def create_submission(model_path='./models/lightfm_best.pkl',
                      output_path='submission.csv',
                      top_k=12,
                      max_workers=None):
    # 1. Загружаем данные
    print("Загрузка данных...")
    articles, customers, transactions, sample_sub = load_data(consts.WORKING_DATASET_DIRECTORY, './output')
    target_users = sample_sub['customer_id'].values
    print(f"Загружено {len(target_users)} пользователей из sample_submission")

    # 3. Строим признаки (требуется для модели)
    print("Построение признаков...")
    interaction, user_features, item_features, _, _ = build_dataset_with_matrix(articles, customers, transactions, './output')

    # 4. Загружаем маппинги и популярные товары
    print("Загрузка маппингов...")
    mappings = joblib.load('./models/mappings.pkl')
    popular_items = get_popular_items(transactions, top_k=100)

    # 5. Инициализируем рекомендателя
    print("Инициализация модели...")
    recommender = HmRecommender(model_path=model_path)
    recommender.set_features(user_features, item_features)
    recommender.popular_items = popular_items

    if max_workers is None:
        max_workers = os.cpu_count()

    print(f"Запуск в {max_workers} потоков...")
    results = {}
    errors = 0

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(predict_for_user, uid, recommender, top_k, popular_items): uid
                   for uid in target_users}

        with tqdm(total=len(target_users), desc="Генерация рекомендаций", unit="user") as pbar:
            for future in as_completed(futures):
                uid, pred, err = future.result()
                results[uid] = pred
                if err:
                    errors += 1
                    tqdm.write(f"Ошибка для {uid}: {err}")
                pbar.update(1)

    if errors > 0:
        print(f"Предупреждение: {errors} пользователей получили fallback (популярные товары)")

    # 7. Сохраняем результат (сохраняем порядок из target_users)
    predictions = [results[uid] for uid in target_users]
    submission_df = pd.DataFrame({
        'customer_id': target_users,
        'prediction': predictions
    })
    submission_df.to_csv(output_path, index=False)
    print(f"Submission сохранён в {output_path}")


if __name__ == "__main__":
    create_submission()