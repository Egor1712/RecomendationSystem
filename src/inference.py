import os.path

import numpy as np

import consts
from src.utils import load_model
import joblib
import torch
import faiss


from two_tower.model import load_two_tower_model


class HmRecommender:
    def __init__(self, model_type):
        mappings_path = os.path.join(model_type, 'output', 'mappings.pkl')
        self.mappings = joblib.load(mappings_path)
        self.model_type = model_type

        if model_type == consts.LIGTFM:
            model_path = os.path.join(model_type, 'output', 'model_best')
            self.model = load_model(model_path)
        elif model_type == consts.TWO_TOWER_MODEL:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            num_users = len(self.mappings['user_to_idx'])
            num_items = len(self.mappings['item_to_idx'])
            model_path = os.path.join(model_type, 'output', '1', 'model_best')
            self.model = load_two_tower_model(model_path, num_users, num_items, self.device)
            item_ids = torch.arange(num_items, device=self.device)
            with torch.no_grad():
                item_emb = self.model.item_embedding(item_ids).cpu().numpy().astype('float32')

            faiss.normalize_L2(item_emb)
            self.index = faiss.IndexFlatIP(item_emb.shape[1])
            self.index.add(item_emb)

        self.user_to_idx = self.mappings['user_to_idx']
        self.item_to_idx = self.mappings['item_to_idx']
        self.inv_item_to_idx = {v: k for k, v in self.item_to_idx.items()}

        self.user_ids = self.mappings['user_ids']
        self.item_ids = self.mappings['item_ids']
        self.user_features = None
        self.item_features = None


    def set_features(self, user_features, item_features):
        self.user_features = user_features
        self.item_features = item_features

    def recommend(self, user_id, top_k=12):
        if user_id not in self.user_to_idx:
            return []

        if self.model_type == consts.LIGTFM:
            user_idx = self.user_to_idx[user_id]
            scores = self.model.predict(user_idx, np.arange(len(self.item_ids)),
                                        user_features=self.user_features,
                                        item_features=self.item_features)
            top_indices = np.argsort(-scores)[:top_k]
            recommendations = [(self.item_ids[i], float(scores[i])) for i in top_indices]
            return recommendations
        elif self.model_type == consts.TWO_TOWER_MODEL:
            cust_idx = self.user_to_idx[user_id]
            user_tensor = torch.tensor([cust_idx], device=self.device)
            with torch.no_grad():
                user_emb = self.model.user_embedding(user_tensor).cpu().numpy().astype('float32')
            faiss.normalize_L2(user_emb)
            scores, top_indices = self.index.search(user_emb, top_k)
            recommendations = [(self.inv_item_to_idx[i], float(score)) for i, score in zip(top_indices[0],scores[0])]
            return recommendations

    def _get_popular_items(self, top_k):
        return [(item_id, 0.0) for item_id in self.popular_items[:top_k]]
