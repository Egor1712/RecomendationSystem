import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from fontTools.misc.cython import returns
from lightfm import LightFM
from sklearn.model_selection import train_test_split
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

import os
from lightfm.evaluation import precision_at_k

sys.path.append(os.path.abspath("../"))
sys.path.append(os.path.abspath("../../"))
sys.path.append(os.path.abspath("../../../"))

import consts
from src.dataset_preprocess import build_dataset_with_matrix
from src.utils import save_model, load_data

class CFG:
    data_output = './output'
    num_threads = 24
    lightfm_params = {'no_components': 60,
   'learning_schedule': 'adagrad',
   'loss': 'bpr',
   'learning_rate': 0.024169871051368697,
   'item_alpha': 1.5038748403807634e-08,
   'user_alpha': 5.035922243112898e-09,
   'max_sampled': 11
                      }


os.makedirs(CFG.data_output, exist_ok=True)

def map_at_k_v2(model, test_interactions, user_features, item_features, k=12):
    precision_value = precision_at_k(model=model,
                                     test_interactions=test_interactions,
                                     num_threads=CFG.num_threads,
                                     k= k,
                                     check_intersections=True,
                                     user_features=user_features,
                                     item_features=item_features)
    return precision_value.mean()

def train_lightfm_with_logging(interaction, user_features, item_features,
                               epochs=30, num_threads=CFG.num_threads, validation_size=0.2,
                               k=12, log_dir='./logs', save_best=True):
    train_interaction, val_interaction = train_test_split(interaction, test_size=validation_size, random_state=42)

    model = LightFM(**CFG.lightfm_params)

    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, 'training_log.csv')

    best_map = -1.0
    history = {'epoch': [], 'map@12': []}

    print("Начинаем обучение...")
    for epoch in range(1, epochs + 1):
        if epoch == 1:
            model.fit(train_interaction,
                      user_features=user_features,
                      item_features=item_features,
                      epochs=1,
                      num_threads=num_threads,
                      verbose=True)
        else:
            model.fit_partial(train_interaction,
                              user_features=user_features,
                              item_features=item_features,
                              epochs=1,
                              num_threads=num_threads,
                              verbose=True)

        # Вычисляем MAP@12 на валидации (подвыборка для скорости)
        map_score = map_at_k_v2(model, val_interaction, user_features, item_features, k=k)
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
    transactions, articles, customers, sample_sub, _ = load_data(consts.WORKING_DATASET_DIRECTORY, CFG.data_output)
    print("=== Подготовка признаков ===")

    interaction, user_features, item_features, _, _ = build_dataset_with_matrix(articles, customers, transactions, CFG.data_output)
    print("=== Обучение модели ===")
    train_lightfm_with_logging(interaction, None, None, epochs=32)
    print("=== Обучение завершено. Запустите API: python api.py ===")
