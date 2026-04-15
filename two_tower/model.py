import torch
import torch.nn as nn
from torch.utils.data import Dataset
import numpy as np
from tqdm import tqdm

class HMDataset(Dataset):
    def __init__(self, interactions_csr, num_negatives=5):
        self.interactions = interactions_csr
        self.num_negatives = num_negatives
        self.users, self.items = self.interactions.nonzero()
        self.n_users = interactions_csr.shape[0]
        self.n_items = interactions_csr.shape[1]

    def __len__(self):
        return len(self.users)

    def __getitem__(self, idx):
        user = self.users[idx]
        pos_item = self.items[idx]
        neg_items = np.random.choice(self.n_items, self.num_negatives, replace=False)
        return torch.tensor(user, dtype=torch.long), torch.tensor(pos_item, dtype=torch.long), torch.tensor(neg_items,
                                                                                                            dtype=torch.long)


class TwoTowerModel(nn.Module):
    def __init__(self, num_users, num_items, embedding_dim=128):
        super(TwoTowerModel, self).__init__()
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.item_embedding = nn.Embedding(num_items, embedding_dim)
        nn.init.xavier_uniform_(self.user_embedding.weight)
        nn.init.xavier_uniform_(self.item_embedding.weight)

    def forward(self, user_ids, item_ids):
        user_vec = self.user_embedding(user_ids)
        item_vec = self.item_embedding(item_ids)
        return (user_vec * item_vec).sum(dim=1)


def train_two_tower_epoch(model, train_loader, optimizer, device):
    """Обучает одну эпоху с отображением прогресса по батчам."""
    model.train()
    total_loss = 0
    # Оборачиваем train_loader в tqdm
    pbar = tqdm(train_loader, desc="Training batches", leave=False)
    for users, pos_items, neg_items in pbar:
        users = users.to(device)
        pos_items = pos_items.to(device)
        neg_items = neg_items.to(device)

        pos_scores = model(users, pos_items)
        # neg_scores: (batch, num_neg)
        neg_scores = model(users.unsqueeze(1).expand(-1, neg_items.size(1)), neg_items)

        loss = -torch.log(torch.sigmoid(pos_scores.unsqueeze(1) - neg_scores)).mean()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        # Обновляем описание прогресс-бара (показываем текущий loss)
        pbar.set_postfix({"loss": f"{loss.item():.4f}"})

    return total_loss / len(train_loader)

def map_at_k_two_tower(model, test_interactions, k=12, batch_size=1024, device='cpu'):
    """Вычисляет MAP@12 для two-tower модели."""
    model.eval()
    n_users = test_interactions.shape[0]
    n_items = test_interactions.shape[1]
    ap_scores = []
    with torch.no_grad():
        # предвычисляем эмбеддинги товаров
        item_ids = torch.arange(n_items, device=device)
        item_emb = model.item_embedding(item_ids)  # (n_items, emb_dim)
        for uid in range(n_users):
            true_items = test_interactions[uid].indices
            if len(true_items) == 0:
                continue
            user_tensor = torch.tensor([uid], device=device)
            user_emb = model.user_embedding(user_tensor)  # (1, emb_dim)
            scores = torch.mv(item_emb, user_emb.squeeze())  # (n_items,)
            top_k = torch.argsort(scores, descending=True)[:k].cpu().numpy()
            hits = 0
            precision_sum = 0.0
            for pos, idx in enumerate(top_k, start=1):
                if idx in true_items:
                    hits += 1
                    precision_sum += hits / pos
            ap = precision_sum / min(len(true_items), k)
            ap_scores.append(ap)
    return np.mean(ap_scores)


def load_two_tower_model(path, num_users, num_items, device, embedding_dim = 128):
    model = TwoTowerModel(num_users, num_items, embedding_dim=embedding_dim)
    model.load_state_dict(torch.load(path, map_location=device))
    model.to(device)
    model.eval()
    return model