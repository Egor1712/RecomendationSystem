import json
import os
import pickle

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from matplotlib import pyplot as plt
from torch.utils.data import Dataset, DataLoader
import optuna
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm
import warnings

from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner
warnings.filterwarnings("ignore")

RESULT_PREPROCESSED_PATH = r'C:\Users\Egor\Desktop\study\diploma\project\dataset_preprocessed'


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


def load_data(data_dir):
    print("Загрузка данных...")
    transactions = reduce_mem_usage(pd.read_csv(os.path.join(data_dir, "transactions.csv")))
    articles = reduce_mem_usage(pd.read_csv(os.path.join(data_dir, "articles.csv")))
    customers = reduce_mem_usage(pd.read_csv(os.path.join(data_dir, "customers.csv")))
    rfm_features = reduce_mem_usage(pd.read_csv(os.path.join(data_dir, "rfm_features.csv")))
    return transactions, articles, customers, rfm_features


class HMDataset(Dataset):
    def __init__(self, df, user_features_dict, item_features_dict, user_cat_cols, item_cat_cols, user_id_to_idx, item_id_to_idx):

        self.user_idx = df['user_idx'].values
        self.item_idx = df['item_idx'].values
        self.weights = df['weight'].values.astype(np.float32)

        self.user_features = []
        self.item_features = []

        idx_to_user = {v: k for k, v in user_id_to_idx.items()}
        idx_to_item = {v: k for k, v in item_id_to_idx.items()}

        for ui, ii in zip(self.user_idx, self.item_idx):
            uid = idx_to_user[ui]
            iid = idx_to_item[ii]
            self.user_features.append(user_features_dict.get(uid, [0]*len(user_cat_cols)))
            self.item_features.append(item_features_dict.get(iid, [0]*len(item_cat_cols)))

    def __len__(self):
        return len(self.user_idx)

    def __getitem__(self, idx):
        user = torch.tensor(self.user_idx[idx], dtype=torch.long)
        item = torch.tensor(self.item_idx[idx], dtype=torch.long)
        weight = torch.tensor(self.weights[idx], dtype=torch.float32)

        user_feats = torch.tensor(self.user_features[idx], dtype=torch.long)
        item_feats = torch.tensor(self.item_features[idx], dtype=torch.long)

        return user, item, user_feats, item_feats, weight


class TwoTowerModel(nn.Module):
    def __init__(self, num_users, num_items,
                 user_cat_sizes,
                 item_cat_sizes,
                 emb_dim=64, hidden_dims=[128, 64], dropout=0.2):
        super().__init__()
        self.user_id_emb = nn.Embedding(num_users, emb_dim)
        self.item_id_emb = nn.Embedding(num_items, emb_dim)

        self.user_cat_embs = nn.ModuleList([nn.Embedding(size, emb_dim) for size in user_cat_sizes])
        self.item_cat_embs = nn.ModuleList([nn.Embedding(size, emb_dim) for size in item_cat_sizes])

        user_input_dim = emb_dim * (1 + len(user_cat_sizes))
        item_input_dim = emb_dim * (1 + len(item_cat_sizes))

        user_layers = []
        in_dim = user_input_dim
        for hdim in hidden_dims:
            user_layers.append(nn.Linear(in_dim, hdim))
            user_layers.append(nn.BatchNorm1d(hdim))
            user_layers.append(nn.ReLU())
            user_layers.append(nn.Dropout(dropout))
            in_dim = hdim
        self.user_mlp = nn.Sequential(*user_layers)

        item_layers = []
        in_dim = item_input_dim
        for hdim in hidden_dims:
            item_layers.append(nn.Linear(in_dim, hdim))
            item_layers.append(nn.BatchNorm1d(hdim))
            item_layers.append(nn.ReLU())
            item_layers.append(nn.Dropout(dropout))
            in_dim = hdim
        self.item_mlp = nn.Sequential(*item_layers)

        self.user_proj = nn.Linear(hidden_dims[-1], emb_dim)
        self.item_proj = nn.Linear(hidden_dims[-1], emb_dim)

    def forward(self, user_ids, item_ids, user_cat_feats, item_cat_feats):
        # user_cat_feats: список тензоров [batch_size] или один тензор [batch_size, num_cat_feats]
        # но ожидаем, что приходит тензор формы [batch_size, num_features] – тогда нужно разбить
        # Для удобства будем передавать user_cat_feats как список тензоров (каждый размер [batch_size])
        user_emb = self.user_id_emb(user_ids)  # [batch, emb_dim]
        user_emb_list = [user_emb]
        for i, emb_layer in enumerate(self.user_cat_embs):
            user_emb_list.append(emb_layer(user_cat_feats[i]))  # [batch, emb_dim]
        user_vec = torch.cat(user_emb_list, dim=1)  # [batch, user_input_dim]
        user_vec = self.user_mlp(user_vec)          # [batch, hidden_dims[-1]]
        user_vec = self.user_proj(user_vec)         # [batch, emb_dim]

        item_emb = self.item_id_emb(item_ids)       # [batch, emb_dim]
        item_emb_list = [item_emb]
        for i, emb_layer in enumerate(self.item_cat_embs):
            item_emb_list.append(emb_layer(item_cat_feats[i]))
        item_vec = torch.cat(item_emb_list, dim=1)  # [batch, item_input_dim]
        item_vec = self.item_mlp(item_vec)          # [batch, hidden_dims[-1]]
        item_vec = self.item_proj(item_vec)         # [batch, emb_dim]

        # Скалярное произведение (оценка сходства)
        scores = (user_vec * item_vec).sum(dim=1)
        return scores

    def get_user_embedding(self, user_ids, user_cat_feats):
        """Получение эмбеддинга пользователя для инференса (без товарной части)"""
        user_emb = self.user_id_emb(user_ids)
        user_emb_list = [user_emb]
        for i, emb_layer in enumerate(self.user_cat_embs):
            user_emb_list.append(emb_layer(user_cat_feats[i]))
        user_vec = torch.cat(user_emb_list, dim=1)
        user_vec = self.user_mlp(user_vec)
        user_vec = self.user_proj(user_vec)
        return user_vec

    def get_item_embedding(self, item_ids, item_cat_feats):
        """Получение эмбеддинга товара для инференса"""
        item_emb = self.item_id_emb(item_ids)
        item_emb_list = [item_emb]
        for i, emb_layer in enumerate(self.item_cat_embs):
            item_emb_list.append(emb_layer(item_cat_feats[i]))
        item_vec = torch.cat(item_emb_list, dim=1)
        item_vec = self.item_mlp(item_vec)
        item_vec = self.item_proj(item_vec)
        return item_vec


def train_one_epoch(model, loader, optimizer, device, log_interval=50):
    model.train()
    total_loss = 0
    pbar = tqdm(loader, desc="Training", unit="batch", leave=False)

    for batch_idx, (user_batch, pos_item_batch, user_feats, item_feats, weights) in enumerate(pbar):
        user_batch = user_batch.to(device)
        pos_item_batch = pos_item_batch.to(device)
        user_feats = user_feats.to(device)
        item_feats = item_feats.to(device)
        weights = weights.to(device)

        # Генерация отрицательных товаров (случайных)
        neg_item_batch = torch.randint(0, model.item_id_emb.num_embeddings,
                                       pos_item_batch.shape, device=device)

        # Преобразование фичей в список тензоров (каждый признак отдельно)
        user_feats_list = [user_feats[:, i] for i in range(user_feats.shape[1])]
        item_feats_list = [item_feats[:, i] for i in range(item_feats.shape[1])]
        pos_scores = model(user_batch, pos_item_batch, user_feats_list, item_feats_list)
        neg_scores = model(user_batch, neg_item_batch, user_feats_list, item_feats_list)

        loss = -torch.log(torch.sigmoid(pos_scores - neg_scores) + 1e-8).mean()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        if (batch_idx + 1) % log_interval == 0 or batch_idx == len(loader) - 1:
            avg_loss = total_loss / (batch_idx + 1)
            pbar.set_postfix({'loss': f'{loss.item():.4f}', 'avg': f'{avg_loss:.4f}'})

    return total_loss / len(loader)


def objective(trial, train_dataset, val_dataset, num_users, num_items,
              user_cat_sizes_list, item_cat_sizes_list,
              all_item_features, user_features_dict, device):
    emb_dim = trial.suggest_categorical('emb_dim', [32, 64, 128])
    lr = trial.suggest_float('lr', 1e-4, 1e-2, log=True)
    batch_size = trial.suggest_categorical('batch_size', [256, 512, 1024])
    dropout = trial.suggest_float('dropout', 0.1, 0.5)
    hidden_dims = trial.suggest_categorical('hidden_dims', [[64, 32], [128, 64], [256, 128]])
    weight_decay = trial.suggest_float('weight_decay', 1e-6, 1e-3, log=True)
    num_epochs_fast = 5

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=8)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=8)

    model = TwoTowerModel(
        num_users=num_users,
        num_items=num_items,
        user_cat_sizes=user_cat_sizes_list,
        item_cat_sizes=item_cat_sizes_list,
        emb_dim=emb_dim,
        hidden_dims=hidden_dims,
        dropout=dropout
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    for epoch in range(num_epochs_fast):
        train_loss = train_one_epoch(model, train_loader, optimizer,  device, log_interval=100)
        if epoch % 2 == 0:
            map12 = fast_evaluate_map_at_k(
                model, val_loader, device, num_items,
                all_item_features, user_features_dict,
                k=12
            )
            print(f"MAP@12 {map12}")
            trial.report(map12, epoch)
            if trial.should_prune():
                raise optuna.TrialPruned()

    final_map = fast_evaluate_map_at_k(
        model, val_loader, device, num_items,
        all_item_features, user_features_dict,
        k=12)
    return final_map


def fast_evaluate_map_at_k(model, val_loader, device, num_items, all_item_features, user_features_dict,
                           k=12):
    all_item_features = all_item_features.to(device)
    model.eval()

    user_to_pos = {}
    for batch in val_loader:
        user_ids, item_ids, _, _, _ = batch
        for u, i in zip(user_ids.cpu().numpy(), item_ids.cpu().numpy()):
            user_to_pos.setdefault(u, set()).add(i)
    users = list(user_to_pos.keys())

    item_indices = np.arange(num_items)

    with torch.no_grad():
        all_item_embs = []
        for start in range(0, len(item_indices), 512):
            batch_ids = torch.tensor(item_indices[start:start + 512], device=device)
            batch_feats = all_item_features[batch_ids].to(device)
            feats_list = [batch_feats[:, i] for i in range(batch_feats.shape[1])]
            embs = model.get_item_embedding(batch_ids, feats_list).cpu()
            all_item_embs.append(embs)
        all_item_embs = torch.cat(all_item_embs, dim=0)  # (n_items_selected, emb_dim)

    user_embs = {}
    for user in tqdm(users, desc="Evaluating MAP (fast)", leave=False, position=0):
        u_tensor = torch.tensor([user], device=device)
        u_feats = user_features_dict[user].to(device)  # тензор (len(user_cat_sizes),)
        feats_list = [u_feats[i].unsqueeze(0) for i in range(u_feats.shape[0])]
        with torch.no_grad():
            emb = model.get_user_embedding(u_tensor, feats_list).cpu()
        user_embs[user] = emb  # (1, emb_dim)

    ap_scores = []
    for user in users:
        pos_set = user_to_pos[user]
        if not pos_set:
            continue
        pos_set = {p for p in pos_set if p in item_indices}
        if not pos_set:
            continue
        user_vec = user_embs[user][0]  # (emb_dim,)
        scores = torch.matmul(all_item_embs, user_vec)  # (n_items_selected,)
        top_k_indices = torch.topk(scores, k=k).indices.cpu().numpy()
        top_items = [item_indices[idx] for idx in top_k_indices]
        hits = 0
        prec_sum = 0.0
        for rank, item in enumerate(top_items, 1):
            if item in pos_set:
                hits += 1
                prec_sum += hits / rank
        ap = prec_sum / min(k, len(pos_set))
        ap_scores.append(ap)
    return np.mean(ap_scores) if ap_scores else 0.0


def run_hyperparam_search(train_dataset, val_dataset, num_users, num_items,
                          user_cat_sizes_list, item_cat_sizes_list,
                          all_item_features, user_features_dict, device,
                          n_trials=30, n_jobs=1):
    study = optuna.create_study(
        direction='maximize',
        sampler=TPESampler(seed=42),
        pruner=MedianPruner(n_startup_trials=5, n_warmup_steps=3)
    )

    def objective_wrapper(trial):
        return objective(
            trial, train_dataset, val_dataset, num_users, num_items,
            user_cat_sizes_list, item_cat_sizes_list,
            all_item_features, user_features_dict, device
        )

    study.optimize(objective_wrapper, n_trials=n_trials, n_jobs=n_jobs, show_progress_bar=True)

    print("Best trial:")
    best_trial = study.best_trial
    print(f"  MAP@12: {best_trial.value}")
    print("  Params: ")
    for key, value in best_trial.params.items():
        print(f"    {key}: {value}")

    with open("best_params.json", "w") as f:
        json.dump(best_trial.params, f)

    return study

if __name__ == '__main__':
    train = True
    study = True
    transactions, articles, customers, rfm = load_data(RESULT_PREPROCESSED_PATH)
    transactions = transactions[transactions['t_dat'] > '2020-08-21']
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    user_cat_cols = ['age_group', 'club_member_status', 'fashion_news_frequency']


    user_encoders = {}
    user_cat_sizes = {}

    for col in user_cat_cols:
        le = LabelEncoder()
        customers[col] = customers[col].fillna('Unknown').astype(str)
        customers[col] = le.fit_transform(customers[col])
        user_encoders[col] = le
        user_cat_sizes[col] = len(le.classes_)

    user_features_dict = {}
    for uid, group in customers.groupby('customer_id'):
        user_features_dict[uid] = [group[col].iloc[0] for col in user_cat_cols]

    item_cat_cols = ['product_group_name', 'colour_group_name']

    item_encoders = {}
    item_cat_sizes = {}

    for col in item_cat_cols:
        le = LabelEncoder()
        articles[col] = articles[col].fillna('Unknown').astype(str)
        articles[col] = le.fit_transform(articles[col])
        item_encoders[col] = le
        item_cat_sizes[col] = len(le.classes_)

    item_features_dict = {}
    for aid, group in articles.groupby('article_id'):
        item_features_dict[aid] = [group[col].iloc[0] for col in item_cat_cols]

    all_user_ids = customers['customer_id'].unique()
    all_article_ids = articles['article_id'].unique()

    user_id_to_idx = {uid: i for i, uid in enumerate(all_user_ids)}
    item_id_to_idx = {aid: i for i, aid in enumerate(all_article_ids)}

    num_users = len(user_id_to_idx)
    num_items = len(item_id_to_idx)
    transactions['weight'] = np.log1p(transactions['price'])

    # Преобразуем ID в индексы
    transactions['user_idx'] = transactions['customer_id'].map(user_id_to_idx)
    transactions['item_idx'] = transactions['article_id'].map(item_id_to_idx)

    # Отбросим строки, где ID не найден (на случай несоответствия)
    transactions = transactions.dropna(subset=['user_idx', 'item_idx'])
    transactions['user_idx'] = transactions['user_idx'].astype(int)
    transactions['item_idx'] = transactions['item_idx'].astype(int)
    dataset = HMDataset(transactions, user_features_dict, item_features_dict,
              user_cat_cols, item_cat_cols, user_id_to_idx, item_id_to_idx)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    user_cat_sizes_list = [user_cat_sizes[col] for col in user_cat_cols]
    item_cat_sizes_list = [item_cat_sizes[col] for col in item_cat_cols]

    all_item_features = torch.zeros((num_items, len(item_cat_sizes_list)), dtype=torch.long)
    for aid, idx in item_id_to_idx.items():
        feats = item_features_dict.get(aid, [0] * len(item_cat_sizes_list))
        all_item_features[idx] = torch.tensor(feats)

    user_features_dict_idx = {}
    for uid, feats in user_features_dict.items():
        if uid in user_id_to_idx:
            user_features_dict_idx[user_id_to_idx[uid]] = torch.tensor(feats)
    if study:
        study = run_hyperparam_search(
            train_dataset, val_dataset, num_users, num_items,
            user_cat_sizes_list, item_cat_sizes_list,
            all_item_features, user_features_dict_idx,
            device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
            n_trials=30,
            n_jobs=1
        )

    if train:
        batch_size = 1024
        params = {'emb_dim': 64, 'lr': 0.0015751320499779737, 'dropout': 0.4464704583099741, 'hidden_dims': [128, 64], 'weight_decay': 0.0008123245085588687}

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=22)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=22)

        model = TwoTowerModel(
            num_users=num_users,
            num_items=num_items,
            user_cat_sizes=user_cat_sizes_list,
            item_cat_sizes=item_cat_sizes_list,
            emb_dim=params['emb_dim'],
            hidden_dims=params['hidden_dims'],
            dropout=params["dropout"]
        ).to(device)

        optimizer = optim.Adam(model.parameters(), lr=params['lr'], weight_decay=params["weight_decay"])
        losses = []
        maps = []
        for epoch in range(40):
            train_loss = train_one_epoch(model, train_loader, optimizer, device, log_interval=100)
            losses.append(train_loss)
            map12 = fast_evaluate_map_at_k(
                model, val_loader, device, num_items,
                all_item_features, user_features_dict,
                k=12)
            maps.append(map12)
            print(f"MAP@12 {map12}")

        epochs = np.arange(1, len(losses) + 1)

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

        ax1.plot(epochs, losses, 'b-', label='Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.set_title('Training Loss')
        ax1.grid(True)
        ax1.legend()

        ax2.plot(epochs, maps, 'r-', label='MAP@12')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('MAP@12')
        ax2.set_title('Mean Average Precision @12 on Test Subset')
        ax2.grid(True)
        ax2.legend()

        plt.tight_layout()
        plt.savefig('training_curve.png', dpi=150)
        plt.show()

        final_map12 = maps[-1] if maps else 0.0
        print(f"Финальная MAP@12 на подвыборке: {final_map12:.6f}")
