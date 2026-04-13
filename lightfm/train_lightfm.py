import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from lightfm import LightFM
from sklearn.model_selection import train_test_split

import consts
from src.dataset_preprocess import build_dataset_with_matrix
from src.utils import save_model, load_data
import os

class CFG:
    data_output = './output'
    lightfm_params = {
        'loss': 'warp',
        "learning_rate": 0.05,
        'no_components': 256,
        'user_alpha': 1e-6,
        'item_alpha': 1e-6,
        'random_state': 42
    }


os.makedirs(CFG.data_output, exist_ok=True)

def map_at_k(model, test_interactions, user_features, item_features, k=12):
    n_users = test_interactions.shape[0]
    n_items = test_interactions.shape[1]
    user_indices = range(n_users)

    all_item_ids = np.arange(n_items)
    ap_scores = []

    for uid in user_indices:
        true_items = test_interactions[uid].indices
        if len(true_items) == 0:
            continue

        # Оценки для всех товаров для данного пользователя
        # model.predict ожидает user_ids и item_ids одинаковой длины
        user_ids = np.full(n_items, uid, dtype=np.int32)
        scores = model.predict(user_ids, all_item_ids,
                               user_features=user_features,
                               item_features=item_features)
        top_k_indices = np.argsort(-scores)[:k]

        hits = 0
        precision_sum = 0.0
        for pos, idx in enumerate(top_k_indices, start=1):
            if idx in true_items:
                hits += 1
                precision_sum += hits / pos
        ap = precision_sum / min(len(true_items), k)
        ap_scores.append(ap)

    return np.mean(ap_scores) if ap_scores else 0.0


def train_lightfm_with_logging(interaction, user_features, item_features,
                               epochs=30, num_threads=24, validation_size=0.2,
                               k=12, log_dir='./logs', save_best=True):
    train_interaction, val_interaction = train_test_split(interaction, test_size=validation_size, random_state=42)

    model = LightFM(**CFG.lightfm_params)

    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, 'training_log.csv')

    best_map = -1.0
    history = {'epoch': [], 'map@12': []}

    print("Начинаем обучение...")
    for epoch in range(1, epochs + 1):
        # Первая эпоха – fit, последующие – fit_partial
        if epoch == 1:
            model.fit(train_interaction,
                      user_features=user_features,
                      item_features=item_features,
                      epochs=1,
                      num_threads=num_threads,
                      verbose=False)
        else:
            model.fit_partial(train_interaction,
                              user_features=user_features,
                              item_features=item_features,
                              epochs=1,
                              num_threads=num_threads,
                              verbose=False)

        # Вычисляем MAP@12 на валидации (подвыборка для скорости)
        map_score = map_at_k(model, val_interaction, user_features, item_features, k=k)
        history['epoch'].append(epoch)
        history['map@12'].append(map_score)

        print(f"Epoch {epoch:3d}/{epochs} | MAP@{k}: {map_score:.6f}")

        if save_best and map_score > best_map:
            best_map = map_score
            save_model(model, os.path.join(CFG.data_output, f'lightfm_best_{epoch}'))
            print(f"  -> Новая лучшая модель сохранена (MAP@{k}={map_score:.6f})")

    # Сохраняем историю
    log_df = pd.DataFrame(history)
    log_df.to_csv(log_file, index=False)
    print(f"Лог сохранён в {log_file}")

    # Рисуем график
    plt.figure(figsize=(10, 6))
    plt.plot(history['epoch'], history['map@12'], marker='o', linestyle='-', color='b')
    plt.xlabel('Epoch')
    plt.ylabel(f'MAP@{k}')
    plt.title(f'Learning Curve: MAP@{k} on Validation')
    plt.grid(True)
    plt.savefig(os.path.join(log_dir, 'map_curve.png'), dpi=150)
    plt.show()

    return model, history


if __name__ == "__main__":
    print("=== Загрузка данных ===")
    transactions, articles, customers, sample_sub = load_data(consts.WORKING_DATASET_DIRECTORY, CFG.data_output)
    print("=== Подготовка признаков ===")

    interaction, user_features, item_features, _, _ = build_dataset_with_matrix(articles, customers, transactions, CFG.data_output)
    print("=== Обучение модели ===")
    train_lightfm_with_logging(interaction, user_features, item_features, epochs=20)
    print("=== Обучение завершено. Запустите API: python api.py ===")
