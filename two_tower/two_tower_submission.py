import os.path

import pandas as pd
import torch
import faiss
from tqdm import tqdm

import consts
from src.two_tower_model import TwoTowerModel
from src.preprocess import load_hm_data, build_user_item_features

# ========== CONFIG ==========
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = "./models/two_tower/two_tower_best.pth"
EMBEDDING_DIM = 128
TOP_K = 12
BATCH_SIZE = 1024


def load_model(num_users, num_items):
    model = TwoTowerModel(num_users, num_items, embedding_dim=EMBEDDING_DIM)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    return model


def get_popular_items(transactions, top_k=100):
    pop = transactions['article_id'].value_counts().head(top_k).index.tolist()
    return pop


def create_submission():
    print("Загрузка данных...")
    articles, customers, transactions = load_hm_data()

    interaction, _, _, user_to_idx, item_to_idx = build_user_item_features(
        articles, customers, transactions)

    num_users = interaction.shape[0]
    num_items = interaction.shape[1]
    print(f"Пользователей: {num_users}, товаров: {num_items}")

    # Загружаем модель
    model = load_model(num_users, num_items)

    # Предвычисляем эмбеддинги всех товаров
    item_ids = torch.arange(num_items, device=DEVICE)
    with torch.no_grad():
        item_emb = model.item_embedding(item_ids).cpu().numpy().astype('float32')

    faiss.normalize_L2(item_emb)
    index = faiss.IndexFlatIP(item_emb.shape[1])
    index.add(item_emb)

    sample_sub = pd.read_csv(os.path.join(consts.WORKING_DATASET_DIRECTORY, 'sample_submission.csv'))
    target_users = sample_sub['customer_id'].values

    popular_items = get_popular_items(transactions, top_k=TOP_K)
    # Преобразуем популярные article_id в индексы
    popular_indices = [[item_to_idx.get(pid, 0) for pid in popular_items if pid in item_to_idx]]
    if not popular_indices:
        popular_indices = list(range(min(TOP_K, num_items)))

    predictions = []
    print("Генерация рекомендаций...")
    for target_user in tqdm(target_users):
        if target_user not in user_to_idx:
            rec_indices = popular_indices[:TOP_K]
        else:
            cust_idx = user_to_idx[target_user]
            user_tensor = torch.tensor([cust_idx], device=DEVICE)
            with torch.no_grad():
                user_emb = model.user_embedding(user_tensor).cpu().numpy().astype('float32')
            faiss.normalize_L2(user_emb)
            scores, rec_indices = index.search(user_emb, TOP_K)
        # Преобразуем индексы обратно в article_id
        inv_item_to_idx = {v: k for k, v in item_to_idx.items()}
        rec_articles = [str(inv_item_to_idx.get(idx, idx)) for idx in rec_indices[0]]
        predictions.append(' '.join(rec_articles))

    # Сохраняем submission
    sub_df = pd.DataFrame({'customer_id': target_users, 'prediction': predictions})
    sub_df.to_csv('submission_two_tower.csv', index=False)
    print("Сабмит сохранён в submission_two_tower.csv")


if __name__ == "__main__":
    create_submission()